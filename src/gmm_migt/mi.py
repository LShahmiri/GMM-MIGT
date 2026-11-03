"""Image loading and Mutual Information (MI) similarity.

The loader and the three MI modes (``auto``, ``grayscale``, ``color``) are
taken from MIGT 1.1.1 so that, with preprocessing disabled, GMM-MIGT produces
the same similarity scores as MIGT. MI is defined in thesis Section 4.2.1
(Eq. 4.1-4.7): ``I(A, B) = H(A) + H(B) - H(A, B)`` estimated from the joint
intensity histogram of two images of equal size.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import normalized_mutual_information as nmi
from sklearn.metrics import mutual_info_score

from .exceptions import ImageLoadError, ImageShapeError

MI_MODES = ("auto", "grayscale", "color")
COLOR_MI_ESTIMATORS = ("nmi", "mutual_info_score")

#: Joint-histogram bins used by the grayscale estimator (MIGT 1.1.1 value).
GRAY_MI_BINS = 64


# -----------------------------------
# Robust image loader (optional resize)
# -----------------------------------
def load_image(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load an image from disk as a NumPy array (OpenCV BGR channel order).

    * Three identical channels ("fake grayscale") are converted to a true
      single-channel grayscale image (as in MIGT).
    * An alpha channel, if present, is discarded.
    * If ``target_size=(width, height)`` is given the image is resized in
      memory with OpenCV's default bilinear interpolation (as in MIGT).

    Paths containing non-ASCII characters are supported.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ImageLoadError(f"Cannot load image: {path} ({exc})") from exc
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if img is None:
        raise ImageLoadError(f"Cannot load image: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    # Convert fake-grayscale (3 identical channels) to true grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        if np.allclose(img[:, :, 0], img[:, :, 1]) and np.allclose(img[:, :, 1], img[:, :, 2]):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if target_size is not None:
        img = cv2.resize(img, tuple(int(v) for v in target_size))

    return img


def save_image(path: str, img: np.ndarray) -> None:
    """Write ``img`` to ``path`` (format from the extension; unicode-safe)."""
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ImageLoadError(f"Cannot encode image for: {path}")
    buf.tofile(path)


def is_grayscale(img: np.ndarray) -> bool:
    """True for 2-D images and single-channel 3-D images."""
    return img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)


def _as_cv_supported(img: np.ndarray) -> np.ndarray:
    # cv2.cvtColor accepts 8U, 16U and 32F only.
    if img.dtype in (np.uint8, np.uint16, np.float32):
        return img
    return img.astype(np.float32)


# -----------------------------------
# Gray MI (histogram-based, MIGT-style)
# -----------------------------------
def compute_gray_mi(img1: np.ndarray, img2: np.ndarray, bins: int = GRAY_MI_BINS) -> float:
    """Shannon MI (nats) of two grayscale images from a ``bins x bins`` joint histogram.

    Colour inputs are converted to grayscale first. If either image is
    constant (standard deviation < 1e-6) the MI is 0.
    """
    if img1.ndim == 3:
        img1 = cv2.cvtColor(_as_cv_supported(img1), cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(_as_cv_supported(img2), cv2.COLOR_BGR2GRAY)

    # Reject constant images
    if img1.std() < 1e-6 or img2.std() < 1e-6:
        return 0.0

    hist, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    return float(mutual_info_score(None, None, contingency=hist))


# -----------------------------------
# Color MI (channel-wise)
# -----------------------------------
def compute_color_mi(img1: np.ndarray, img2: np.ndarray, estimator: str = "nmi") -> float:
    """Mean of the per-channel MI of two 3-channel images.

    ``estimator="nmi"`` (default, MIGT 1.1.1): scikit-image
    ``normalized_mutual_information`` per channel, ``(H(A)+H(B))/H(A,B)``,
    which lies in ``[1, 2]``.

    ``estimator="mutual_info_score"``: scikit-learn ``mutual_info_score`` on a
    64-bin joint histogram per channel - the estimator named in thesis
    Section 3.3, applied to the three RGB channels as in thesis Figure 4.1.

    A constant channel contributes 0.
    """
    if estimator not in COLOR_MI_ESTIMATORS:
        raise ValueError(f"estimator must be one of {COLOR_MI_ESTIMATORS} (got {estimator!r}).")
    if img1.ndim != 3 or img1.shape[2] != 3 or img2.ndim != 3 or img2.shape[2] != 3:
        raise ImageShapeError(
            f"Color MI requires two 3-channel images (got {img1.shape} and {img2.shape}). "
            "Use mode='grayscale' or mode='auto' for grayscale data."
        )
    img1 = cv2.cvtColor(_as_cv_supported(img1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(_as_cv_supported(img2), cv2.COLOR_BGR2RGB)

    mi_vals = []
    for c in range(3):
        a, b = img1[:, :, c], img2[:, :, c]
        if a.std() < 1e-6 or b.std() < 1e-6:
            mi_vals.append(0.0)
        elif estimator == "nmi":
            mi_vals.append(nmi(a, b))
        else:
            hist, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=GRAY_MI_BINS)
            mi_vals.append(mutual_info_score(None, None, contingency=hist))

    return float(np.mean(mi_vals))


# -----------------------------------
# Unified MI interface (STRICT by default)
# -----------------------------------
def compute_mi(
    img1: np.ndarray,
    img2: np.ndarray,
    mode: str = "auto",
    expected_size: Optional[Tuple[int, int]] = None,
    color_mi: str = "nmi",
) -> float:
    """Mutual information between two images.

    Parameters
    ----------
    img1, img2:
        Images as returned by :func:`load_image` (or by
        :func:`gmm_migt.preprocess_image`).
    mode:
        ``"auto"`` - grayscale MI if both images are grayscale, otherwise
        colour MI; ``"grayscale"`` - always grayscale MI; ``"color"`` - always
        channel-wise colour MI.
    expected_size:
        ``None`` (strict): both images must already have the same height and
        width. ``(width, height)``: both images must have exactly this size.
    color_mi:
        Colour estimator, see :func:`compute_color_mi`.
    """
    if mode not in MI_MODES:
        raise ValueError(f"mode must be one of {MI_MODES} (got {mode!r}).")

    # --- Size enforcement ---
    if expected_size is not None:
        if img1.shape[:2][::-1] != tuple(expected_size) or img2.shape[:2][::-1] != tuple(expected_size):
            raise ImageShapeError(
                f"Image size mismatch: expected {tuple(expected_size)}, "
                f"got {img1.shape} vs {img2.shape}"
            )
    elif img1.shape[:2] != img2.shape[:2]:
        raise ImageShapeError(f"Image shape mismatch: {img1.shape} vs {img2.shape}")

    if mode == "grayscale":
        return compute_gray_mi(img1, img2)
    if mode == "color":
        return compute_color_mi(img1, img2, estimator=color_mi)

    # auto
    if is_grayscale(img1) and is_grayscale(img2):
        return compute_gray_mi(img1, img2)
    return compute_color_mi(img1, img2, estimator=color_mi)
