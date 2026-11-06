"""GMM-MIGT preprocessing: EM-GMM colour reduction followed by PCA.

This module implements thesis Section 5.2 and lines 1-5 of Algorithm 5.1.
For every image:

1. **EM-GMM colour reduction** (Section 5.2.1, Eq. 5.1-5.4). A Gaussian
   Mixture Model with ``K`` components (default 4) is fitted to the pixel
   colour vectors with the Expectation-Maximisation algorithm (scikit-learn,
   convergence tolerance 0.001, at most 100 iterations). Each pixel is then
   assigned to the component with the largest responsibility and its value
   is replaced by that component's mean colour.
2. **PCA** (Section 5.2.2, Eq. 5.5-5.8). PCA is applied separately to each
   colour channel of the colour-reduced image, treating the rows of the
   channel matrix as observations. The smallest number of principal
   components whose Cumulative Percent Variance (CPV) reaches 95% is kept,
   the data are projected onto them and the channel is reconstructed.

The reconstructed (processed) image is used only to compute the MI
similarity scores; the dataset partition copies the original files.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from .exceptions import ConfigurationError

GMM_COVARIANCE_TYPES = ("full", "tied", "diag", "spherical")
GMM_INIT_METHODS = ("kmeans", "k-means++", "random", "random_from_data")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters of the GMM-MIGT preprocessing (defaults = thesis Algorithm 5.1).

    Attributes
    ----------
    use_gmm:
        Apply EM-GMM colour reduction (step 1).
    n_components:
        Number of Gaussian components ``K`` (thesis: 4, chosen empirically
        from K = 2..6).
    gmm_tol:
        EM convergence threshold on the change of the (per-pixel average)
        log-likelihood lower bound (thesis: 0.001).
    gmm_max_iter:
        Maximum number of EM iterations (thesis: 100).
    gmm_covariance_type:
        Covariance structure of each component. ``"full"`` gives one full
        covariance matrix per component, as in thesis Eq. 5.4.
    gmm_init:
        Initialisation of the EM parameters, passed to scikit-learn's
        ``init_params``. ``"kmeans"`` is the scikit-learn default (a seeded,
        randomly initialised k-means); ``"random"`` draws random
        responsibilities.
    gmm_reg_covar:
        Non-negative value added to the covariance diagonals to keep them
        positive definite (scikit-learn default).
    gmm_max_pixels:
        If set, the GMM is *fitted* on a seeded random subset of at most this
        many pixels and then applied to all pixels. ``None`` (default, as in
        the thesis) fits on every pixel. Use for very large images.
    use_pca:
        Apply per-channel PCA reconstruction (step 2).
    pca_variance:
        CPV threshold in ``(0, 1]`` (thesis: 0.95).
    random_state:
        Seed for the GMM initialisation and pixel subsampling.
    """

    use_gmm: bool = True
    n_components: int = 4
    gmm_tol: float = 1e-3
    gmm_max_iter: int = 100
    gmm_covariance_type: str = "full"
    gmm_init: str = "kmeans"
    gmm_reg_covar: float = 1e-6
    gmm_max_pixels: Optional[int] = None
    use_pca: bool = True
    pca_variance: float = 0.95
    random_state: Optional[int] = 42

    def __post_init__(self) -> None:
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise ConfigurationError(f"n_components must be a positive integer (got {self.n_components!r}).")
        if not self.gmm_tol >= 0:
            raise ConfigurationError(f"gmm_tol must be >= 0 (got {self.gmm_tol!r}).")
        if not isinstance(self.gmm_max_iter, (int, np.integer)) or self.gmm_max_iter < 1:
            raise ConfigurationError(f"gmm_max_iter must be a positive integer (got {self.gmm_max_iter!r}).")
        if self.gmm_covariance_type not in GMM_COVARIANCE_TYPES:
            raise ConfigurationError(
                f"gmm_covariance_type must be one of {GMM_COVARIANCE_TYPES} (got {self.gmm_covariance_type!r})."
            )
        if self.gmm_init not in GMM_INIT_METHODS:
            raise ConfigurationError(f"gmm_init must be one of {GMM_INIT_METHODS} (got {self.gmm_init!r}).")
        if not self.gmm_reg_covar >= 0:
            raise ConfigurationError(f"gmm_reg_covar must be >= 0 (got {self.gmm_reg_covar!r}).")
        if self.gmm_max_pixels is not None and (
            not isinstance(self.gmm_max_pixels, (int, np.integer)) or self.gmm_max_pixels < self.n_components
        ):
            raise ConfigurationError(
                f"gmm_max_pixels must be None or an integer >= n_components (got {self.gmm_max_pixels!r})."
            )
        if not (0.0 < float(self.pca_variance) <= 1.0):
            raise ConfigurationError(f"pca_variance must be in (0, 1] (got {self.pca_variance!r}).")


@dataclass
class GMMInfo:
    """Diagnostics of one EM-GMM colour reduction."""

    converged: bool
    n_iter: int
    weights: np.ndarray
    means: np.ndarray
    n_distinct_colours_in: int
    skipped_reason: Optional[str] = None


@dataclass
class PCAInfo:
    """Diagnostics of one per-channel PCA reconstruction."""

    n_components: List[int] = field(default_factory=list)
    cpv: List[float] = field(default_factory=list)


@dataclass
class PreprocessingInfo:
    """Diagnostics of :func:`preprocess_image`."""

    gmm: Optional[GMMInfo] = None
    pca: Optional[PCAInfo] = None


def _pixels(img: np.ndarray) -> Tuple[np.ndarray, int]:
    channels = 1 if img.ndim == 2 else img.shape[2]
    return img.reshape(-1, channels).astype(np.float64), channels


# ----------------------------------------------------------------------
# Step 1: EM-GMM colour reduction (thesis Section 5.2.1)
# ----------------------------------------------------------------------
def em_gmm_colour_reduction(
    img: np.ndarray,
    n_components: int = 4,
    tol: float = 1e-3,
    max_iter: int = 100,
    covariance_type: str = "full",
    init_params: str = "kmeans",
    reg_covar: float = 1e-6,
    max_pixels: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, GMMInfo]:
    """Reduce the colours of ``img`` to ``n_components`` GMM mean colours.

    Returns the colour-reduced image as ``float64`` with the same shape as
    ``img`` and a :class:`GMMInfo`. Every pixel of the output equals the mean
    vector ``mu_j`` of the component ``j`` with maximal responsibility
    ``omega_j^(i)`` (thesis Section 5.2.1, step 5).

    If the image already contains ``n_components`` or fewer distinct colours
    the reduction is the identity (each colour is its own cluster) and no
    model is fitted.
    """
    X, channels = _pixels(img)
    if X.shape[0] == 0:
        raise ValueError("Cannot run EM-GMM on an empty image.")

    distinct = np.unique(X, axis=0)
    if distinct.shape[0] <= n_components:
        weights = np.array([np.mean(np.all(X == c, axis=1)) for c in distinct])
        info = GMMInfo(True, 0, weights, distinct, int(distinct.shape[0]),
                       skipped_reason="image has <= n_components distinct colours")
        return img.astype(np.float64), info

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        tol=tol,
        reg_covar=reg_covar,
        max_iter=max_iter,
        init_params=init_params,
        random_state=random_state,
    )
    fit_X = X
    if max_pixels is not None and X.shape[0] > max_pixels:
        rng = np.random.default_rng(random_state)
        fit_X = X[rng.choice(X.shape[0], size=max_pixels, replace=False)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        gmm.fit(fit_X)

    labels = gmm.predict(X)
    reduced = gmm.means_[labels].reshape(img.shape if channels > 1 else img.shape[:2])
    info = GMMInfo(
        converged=bool(gmm.converged_),
        n_iter=int(gmm.n_iter_),
        weights=gmm.weights_.copy(),
        means=gmm.means_.copy(),
        n_distinct_colours_in=int(distinct.shape[0]),
    )
    return reduced, info


# ----------------------------------------------------------------------
# Step 2: per-channel PCA with CPV selection (thesis Section 5.2.2)
# ----------------------------------------------------------------------
def select_n_components(explained_variance_ratio: np.ndarray, variance: float) -> int:
    """Smallest ``k`` whose Cumulative Percent Variance reaches ``variance``."""
    cpv = np.cumsum(explained_variance_ratio)
    k = int(np.searchsorted(cpv, variance - 1e-12, side="left")) + 1
    return min(k, len(explained_variance_ratio))


def pca_channel_reconstruction(channel: np.ndarray, variance: float = 0.95) -> Tuple[np.ndarray, int, float]:
    """PCA-reconstruct one 2-D channel matrix, rows = observations.

    Returns ``(reconstructed, k, cpv_k)``. A channel with fewer than two rows
    or zero total variance is returned unchanged (``k = 0``).
    """
    M = np.asarray(channel, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"channel must be 2-D (got shape {M.shape}).")
    if M.shape[0] < 2 or M.shape[1] < 1 or float(M.var(axis=0, ddof=1).sum()) <= 0.0:
        return M.copy(), 0, 1.0

    try:
        pca = PCA(svd_solver="full")
        Z = pca.fit_transform(M)
        ratio, components, mean = pca.explained_variance_ratio_, pca.components_, pca.mean_
    except np.linalg.LinAlgError:
        # LAPACK's default divide-and-conquer SVD (gesdd) occasionally fails to
        # converge; the slower QR-based driver (gesvd) gives the same decomposition.
        mean = M.mean(axis=0)
        U, S, components = scipy.linalg.svd(M - mean, full_matrices=False, lapack_driver="gesvd")
        Z = U * S
        var = S ** 2
        ratio = var / var.sum()
    k = select_n_components(ratio, variance)
    recon = Z[:, :k] @ components[:k] + mean
    return recon, k, float(np.sum(ratio[:k]))


def pca_reconstruction(img: np.ndarray, variance: float = 0.95) -> Tuple[np.ndarray, PCAInfo]:
    """Apply :func:`pca_channel_reconstruction` to every channel of ``img``."""
    data = np.asarray(img, dtype=np.float64)
    info = PCAInfo()
    if data.ndim == 2:
        out, k, cpv = pca_channel_reconstruction(data, variance)
        info.n_components.append(k)
        info.cpv.append(cpv)
        return out, info

    out = np.empty_like(data)
    for c in range(data.shape[2]):
        out[:, :, c], k, cpv = pca_channel_reconstruction(data[:, :, c], variance)
        info.n_components.append(k)
        info.cpv.append(cpv)
    return out, info


def _to_dtype(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        lim = np.iinfo(dtype)
        return np.clip(np.rint(x), lim.min, lim.max).astype(dtype)
    return x.astype(dtype)


# ----------------------------------------------------------------------
# Full GMM-MIGT preprocessing (Algorithm 5.1, lines 1-5)
# ----------------------------------------------------------------------
def preprocess_image(
    img: np.ndarray, config: Optional[PreprocessingConfig] = None
) -> Tuple[np.ndarray, PreprocessingInfo]:
    """Run EM-GMM colour reduction then PCA reconstruction on one image.

    The processed image is returned with the input's dtype (integer images
    are rounded and clipped to the dtype range), i.e. as the "processed
    image" output of Algorithm 5.1. With both steps disabled the input is
    returned unchanged.
    """
    cfg = config or PreprocessingConfig()
    info = PreprocessingInfo()
    if not cfg.use_gmm and not cfg.use_pca:
        return img, info

    x: np.ndarray = img.astype(np.float64)
    if cfg.use_gmm:
        x, info.gmm = em_gmm_colour_reduction(
            img,
            n_components=cfg.n_components,
            tol=cfg.gmm_tol,
            max_iter=cfg.gmm_max_iter,
            covariance_type=cfg.gmm_covariance_type,
            init_params=cfg.gmm_init,
            reg_covar=cfg.gmm_reg_covar,
            max_pixels=cfg.gmm_max_pixels,
            random_state=cfg.random_state,
        )
    if cfg.use_pca:
        x, info.pca = pca_reconstruction(x, cfg.pca_variance)
    return _to_dtype(x, img.dtype), info
