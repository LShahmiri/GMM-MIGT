"""MI computations (thesis Section 4.2.1, Eq. 4.1-4.7) and the image loader."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from gmm_migt import ImageLoadError, ImageShapeError, compute_color_mi, compute_gray_mi, compute_mi, load_image, save_image


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def manual_mi(a, b, bins=64):
    """Eq. 4.1 with Eq. 4.3-4.7: I = H(A) + H(B) - H(A,B) from the joint histogram."""
    h, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins)
    pab = h / h.sum()
    return entropy(pab.sum(axis=1)) + entropy(pab.sum(axis=0)) - entropy(pab.ravel())


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-40, 40, (32, 32)), 0, 255).astype(np.uint8)
    return a, b


def test_gray_mi_matches_entropy_definition(pair):
    a, b = pair
    assert compute_gray_mi(a, b) == pytest.approx(manual_mi(a, b), rel=1e-10)


def test_gray_mi_identical_equals_marginal_entropy(pair):
    a, _ = pair
    h, _, _ = np.histogram2d(a.ravel(), a.ravel(), bins=64)
    pa = (h / h.sum()).sum(axis=1)
    assert compute_gray_mi(a, a) == pytest.approx(entropy(pa), rel=1e-10)


def test_gray_mi_symmetric_and_higher_for_similar(pair):
    a, b = pair
    rng = np.random.default_rng(1)
    c = rng.integers(0, 256, a.shape, dtype=np.uint8)
    assert compute_gray_mi(a, b) == pytest.approx(compute_gray_mi(b, a))
    assert compute_gray_mi(a, b) > compute_gray_mi(a, c)


def test_constant_image_gives_zero():
    a = np.full((16, 16), 7, np.uint8)
    b = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert compute_gray_mi(a, b) == 0.0


def test_color_nmi_identical_is_two():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
    assert compute_color_mi(a, a) == pytest.approx(2.0)


def test_color_mutual_info_score_is_mean_of_channels():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-30, 30, a.shape), 0, 255).astype(np.uint8)
    expected = np.mean([manual_mi(a[:, :, c], b[:, :, c]) for c in range(3)])
    assert compute_color_mi(a, b, estimator="mutual_info_score") == pytest.approx(expected)


def test_auto_mode_dispatch(pair):
    a, b = pair
    assert compute_mi(a, b, "auto") == compute_gray_mi(a, b)
    ca = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    ca[0, 0, 0] = 1  # make it a real colour image
    cb = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    assert compute_mi(ca, cb, "auto") == compute_color_mi(ca, cb)
    assert compute_mi(ca, cb, "grayscale") == compute_gray_mi(ca, cb)


def test_shape_checks(pair):
    a, _ = pair
    with pytest.raises(ImageShapeError):
        compute_mi(a, a[:-1])
    assert compute_mi(a, a, expected_size=(32, 32)) > 0
    with pytest.raises(ImageShapeError):
        compute_mi(a, a, expected_size=(16, 16))


def test_invalid_mode_and_color_on_gray(pair):
    a, b = pair
    with pytest.raises(ValueError):
        compute_mi(a, b, mode="rgb")
    with pytest.raises(ImageShapeError):
        compute_mi(a, b, mode="color")
    with pytest.raises(ValueError):
        compute_color_mi(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4, 3), np.uint8), estimator="x")


# ---------------- loader ----------------
def test_load_fake_grayscale_becomes_2d(tmp_path):
    g = np.arange(64, dtype=np.uint8).reshape(8, 8)
    p = str(tmp_path / "g.png")
    cv2.imwrite(p, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    img = load_image(p)
    assert img.ndim == 2 and np.array_equal(img, g)


def test_load_drops_alpha_and_resizes(tmp_path):
    rgba = np.random.default_rng(0).integers(0, 256, (10, 12, 4), dtype=np.uint8)
    p = str(tmp_path / "a.png")
    cv2.imwrite(p, rgba)
    img = load_image(p)
    assert img.shape == (10, 12, 3)
    assert np.array_equal(img, rgba[:, :, :3])
    assert load_image(p, (6, 5)).shape == (5, 6, 3)


def test_load_unicode_path_and_errors(tmp_path):
    d = tmp_path / "گیاه"
    d.mkdir()
    p = str(d / "تصویر.png")
    img = np.random.default_rng(1).integers(0, 256, (5, 5, 3), dtype=np.uint8)
    save_image(p, img)
    assert np.array_equal(load_image(p), img)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ImageLoadError):
        load_image(str(bad))
    with pytest.raises(ImageLoadError):
        load_image(str(tmp_path / "missing.png"))
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ImageLoadError):
        load_image(str(empty))
    assert not os.path.exists(str(tmp_path / "missing.png"))
