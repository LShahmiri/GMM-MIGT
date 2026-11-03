"""Small numeric and filesystem helpers (derived from ``migt.utils``)."""

from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np

#: File extensions recognised as images (identical to MIGT 1.1.1).
IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

SPLIT_NAMES = ("train", "test", "val")


def split_counts(n: int, ratios: Sequence[float]) -> List[int]:
    """Return per-split counts for ``n`` items that sum to ``n`` exactly.

    Each count is ``floor(ratio * n)``; the remainder caused by flooring is
    added to the first split (train), exactly as in MIGT 1.1.1. With the
    thesis ratios (50/40/10) a bin of 10 items therefore gives 5/4/1, which is
    the guarantee stated in thesis Section 4.2.2.

    ``ratio * n`` is rounded to 9 decimals before flooring so that binary
    floating-point error (e.g. ``0.29 * 100 == 28.999999999999996``) cannot
    move an item from one split to another.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0 (got {n}).")
    r = np.asarray(ratios, dtype=float)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("ratios must be a non-empty 1-D sequence.")
    counts = np.floor(np.round(r * n, 9)).astype(int)
    counts[0] += n - int(counts.sum())
    return counts.tolist()


def list_images(directory: str) -> List[str]:
    """Sorted full paths of the image files directly inside ``directory``."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(directory, f))
    )


def list_classes(dataset_root: str) -> List[str]:
    """Sorted names of the class sub-directories of ``dataset_root``."""
    return sorted(
        d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))
    )
