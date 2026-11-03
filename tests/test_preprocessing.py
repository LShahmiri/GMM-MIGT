"""EM-GMM colour reduction (thesis Eq. 5.1-5.4) and PCA (Eq. 5.5-5.8, CPV)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from gmm_migt import (
    ConfigurationError,
    PreprocessingConfig,
    em_gmm_colour_reduction,
    pca_channel_reconstruction,
    pca_reconstruction,
    preprocess_image,
    select_n_components,
)


def colour_image(seed=0, h=30, w=40, k=3):
    """Image whose pixels come from ``k`` colour clusters plus noise."""
    rng = np.random.default_rng(seed)
    centres = rng.integers(30, 225, size=(k, 3))
    labels = rng.integers(0, k, size=(h, w))
    img = centres[labels] + rng.normal(0, 12, size=(h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# EM-GMM
# ----------------------------------------------------------------------
def manual_em(X, weights, means, covs, n_iter, reg_covar=1e-6):
    """Literal implementation of thesis Eq. 5.1 (E-step) and Eq. 5.2-5.4 (M-step)."""
    m, d = X.shape
    k = len(weights)
    for _ in range(n_iter):
        # E-step, Eq. 5.1: omega_j^i = g_j(x_i) Phi_j / sum_l g_l(x_i) Phi_l
        g = np.column_stack([multivariate_normal(means[j], covs[j]).pdf(X) for j in range(k)])
        num = g * weights
        omega = num / num.sum(axis=1, keepdims=True)
        # M-step
        s = omega.sum(axis=0)
        weights = s / m                                   # Eq. 5.2
        means = (omega.T @ X) / s[:, None]                # Eq. 5.3
        covs = []
        for j in range(k):                                # Eq. 5.4
            diff = X - means[j]
            covs.append((omega[:, j, None] * diff).T @ diff / s[j] + reg_covar * np.eye(d))
        covs = np.array(covs)
    return weights, means, covs


def test_sklearn_em_equals_thesis_equations():
    X = colour_image(1).reshape(-1, 3).astype(float)
    rng = np.random.default_rng(5)
    k = 3
    w0 = np.full(k, 1 / k)
    mu0 = X[rng.choice(len(X), k, replace=False)]
    cov0 = np.array([np.cov(X.T) for _ in range(k)])
    n_iter = 7

    gmm = GaussianMixture(k, covariance_type="full", tol=0, max_iter=n_iter, weights_init=w0,
                          means_init=mu0, precisions_init=np.linalg.inv(cov0), reg_covar=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gmm.fit(X)
    w, mu, cov = manual_em(X, w0, mu0, cov0, n_iter)
    assert np.allclose(gmm.weights_, w, rtol=1e-6, atol=1e-9)
    assert np.allclose(gmm.means_, mu, rtol=1e-6, atol=1e-6)
    assert np.allclose(gmm.covariances_, cov, rtol=1e-5, atol=1e-5)


def test_colour_reduction_uses_k_cluster_means():
    img = colour_image(2, k=5)
    out, info = em_gmm_colour_reduction(img, n_components=4, random_state=0)
    assert out.shape == img.shape and out.dtype == np.float64
    colours = np.unique(out.reshape(-1, 3), axis=0)
    assert len(colours) <= 4
    # every output colour is one of the fitted component means (Section 5.2.1, step 5)
    for c in colours:
        assert np.min(np.abs(info.means - c).sum(axis=1)) < 1e-9
    assert info.weights.sum() == pytest.approx(1.0)


def test_colour_reduction_assigns_argmax_responsibility():
    img = colour_image(3)
    X = img.reshape(-1, 3).astype(float)
    out, _ = em_gmm_colour_reduction(img, n_components=4, random_state=7)
    gmm = GaussianMixture(4, tol=1e-3, max_iter=100, random_state=7).fit(X)
    expected = gmm.means_[np.argmax(gmm.predict_proba(X), axis=1)].reshape(img.shape)
    assert np.allclose(out, expected)


def test_colour_reduction_reproducible():
    img = colour_image(4, k=6)
    a, _ = em_gmm_colour_reduction(img, random_state=1)
    b, _ = em_gmm_colour_reduction(img, random_state=1)
    assert np.array_equal(a, b)


def test_colour_reduction_grayscale_and_few_colours():
    g = colour_image(5)[:, :, 0]
    out, info = em_gmm_colour_reduction(g, n_components=4, random_state=0)
    assert out.shape == g.shape and len(np.unique(out)) <= 4
    two = np.zeros((6, 6, 3), np.uint8)
    two[:3] = 200
    out2, info2 = em_gmm_colour_reduction(two, n_components=4)
    assert np.array_equal(out2, two.astype(float)) and info2.skipped_reason


def test_colour_reduction_subsampled_fit():
    img = colour_image(6, h=60, w=60)
    out, _ = em_gmm_colour_reduction(img, max_pixels=500, random_state=0)
    assert out.shape == img.shape and len(np.unique(out.reshape(-1, 3), axis=0)) <= 4


# ----------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------
def manual_pca(M, variance):
    """Thesis Section 5.2.2: mean (5.5), covariance with n-1 (5.7/5.8), eigen-decomposition, CPV."""
    mean = M.mean(axis=0)
    D = M - mean
    C = D.T @ D / (M.shape[0] - 1)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    cpv = np.cumsum(vals) / vals.sum()
    k = int(np.argmax(cpv >= variance - 1e-12)) + 1
    V = vecs[:, :k]
    return (D @ V) @ V.T + mean, k, cpv[k - 1]


@pytest.mark.parametrize("variance", [0.5, 0.85, 0.95, 0.99])
def test_pca_matches_thesis_definition(variance):
    rng = np.random.default_rng(0)
    M = rng.normal(size=(40, 12)) @ rng.normal(size=(12, 25)) + rng.normal(0, 0.1, (40, 25))
    rec, k, cpv = pca_channel_reconstruction(M, variance)
    rec2, k2, cpv2 = manual_pca(M, variance)
    assert k == k2
    assert cpv == pytest.approx(cpv2)
    assert cpv >= variance
    assert np.allclose(rec, rec2, atol=1e-8)


def test_cpv_is_smallest_k():
    ratio = np.array([0.5, 0.3, 0.15, 0.05])
    assert select_n_components(ratio, 0.95) == 3  # 0.5+0.3+0.15 = 0.95 reaches the threshold
    assert select_n_components(ratio, 0.96) == 4
    assert select_n_components(ratio, 0.5) == 1
    assert select_n_components(ratio, 1.0) == 4


def test_pca_full_variance_is_identity_and_constant_channel():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(10, 8))
    rec, _, _ = pca_channel_reconstruction(M, 1.0)
    assert np.allclose(rec, M)
    const = np.full((5, 5), 3.0)
    rec, k, _ = pca_channel_reconstruction(const, 0.95)
    assert k == 0 and np.array_equal(rec, const)
    one_row = np.arange(5.0)[None, :]
    assert np.array_equal(pca_channel_reconstruction(one_row)[0], one_row)


def test_pca_is_per_channel():
    img = colour_image(7).astype(float)
    out, info = pca_reconstruction(img, 0.9)
    assert len(info.n_components) == 3
    for c in range(3):
        rec, k, _ = pca_channel_reconstruction(img[:, :, c], 0.9)
        assert np.allclose(out[:, :, c], rec) and k == info.n_components[c]


# ----------------------------------------------------------------------
# full preprocessing
# ----------------------------------------------------------------------
def test_preprocess_pipeline_order_and_dtype():
    img = colour_image(8)
    out, info = preprocess_image(img)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert info.gmm is not None and info.pca is not None
    gm, _ = em_gmm_colour_reduction(img, random_state=42)
    expected, _ = pca_reconstruction(gm, 0.95)
    assert np.array_equal(out, np.clip(np.rint(expected), 0, 255).astype(np.uint8))


def test_preprocess_ablations():
    img = colour_image(9)
    same, info = preprocess_image(img, PreprocessingConfig(use_gmm=False, use_pca=False))
    assert same is img and info.gmm is None and info.pca is None
    g, info = preprocess_image(img, PreprocessingConfig(use_pca=False))
    assert len(np.unique(g.reshape(-1, 3), axis=0)) <= 4 and info.pca is None
    p, info = preprocess_image(img, PreprocessingConfig(use_gmm=False))
    assert info.gmm is None and info.pca is not None


def test_preprocess_uint16_and_float():
    img16 = (colour_image(10).astype(np.uint16) * 257)
    out, _ = preprocess_image(img16)
    assert out.dtype == np.uint16
    f = colour_image(10).astype(np.float32) / 255
    out, _ = preprocess_image(f)
    assert out.dtype == np.float32


def test_defaults_are_thesis_values():
    c = PreprocessingConfig()
    assert (c.n_components, c.gmm_tol, c.gmm_max_iter, c.pca_variance, c.gmm_covariance_type) == (
        4, 1e-3, 100, 0.95, "full")


@pytest.mark.parametrize("kw", [
    {"n_components": 0}, {"n_components": 2.5}, {"gmm_tol": -1}, {"gmm_max_iter": 0},
    {"gmm_covariance_type": "x"}, {"gmm_init": "x"}, {"gmm_reg_covar": -1},
    {"gmm_max_pixels": 2}, {"pca_variance": 0}, {"pca_variance": 1.5},
])
def test_config_validation(kw):
    with pytest.raises(ConfigurationError):
        PreprocessingConfig(**kw)
