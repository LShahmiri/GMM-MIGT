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


ROUNDING_MODES = ("class", "per_bin")


def class_counts(n: int, ratios: Sequence[float], min_one: bool = True) -> List[int]:
    """Per-split counts for a class of ``n`` images, rounded once for the whole class.

    Largest-remainder rounding of ``ratio * n`` (ties go to the split with the
    larger ratio, then to the earlier split). With ``min_one=True`` and
    ``n >= len(ratios)`` every split receives at least one image, taken from
    the largest split. Example: ``n=4`` with ``0.7/0.2/0.1`` gives ``2/1/1``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0 (got {n}).")
    r = np.asarray(ratios, dtype=float)
    exact = np.round(r * n, 9)
    counts = np.floor(exact).astype(int)
    rest = n - int(counts.sum())
    order = sorted(range(len(r)), key=lambda i: (-(exact[i] - counts[i]), -r[i], i))
    for i in order[:rest]:
        counts[i] += 1
    if min_one and n >= len(r):
        for i in range(len(r)):
            if counts[i] == 0:
                counts[int(np.argmax(counts))] -= 1
                counts[i] = 1
    return counts.tolist()


def _capped_largest_remainder(total: int, weights: Sequence[float], caps: Sequence[int]) -> List[int]:
    """Split ``total`` over items proportionally to ``weights``, never above ``caps``."""
    caps = [int(c) for c in caps]
    w = np.asarray(weights, dtype=float)
    if total <= 0 or w.sum() <= 0:
        return [0] * len(caps)
    exact = np.round(total * w / w.sum(), 9)
    out = [min(int(np.floor(e)), c) for e, c in zip(exact, caps)]
    while sum(out) < total:
        free = [i for i in range(len(out)) if out[i] < caps[i]]
        if not free:
            raise ValueError("total exceeds the sum of caps")
        i = max(free, key=lambda k: (exact[k] - out[k], caps[k] - out[k], -k))
        out[i] += 1
    return out


def stratified_counts(sizes: Sequence[int], ratios: Sequence[float], n_fixed_train: int = 0,
                      min_one: bool = True) -> List[List[int]]:
    """Per-stratum split counts with the split sizes fixed for the whole class.

    ``sizes`` are the stratum (bin) sizes and ``n_fixed_train`` images (the
    reference) are already in train. The class total
    ``N = sum(sizes) + n_fixed_train`` is rounded once with
    :func:`class_counts`; the test and validation counts are then spread over
    the strata in proportion to their sizes (largest remainder) and the rest
    of every stratum goes to train. Returns ``[[train, test(, val)], ...]``
    with row sums equal to ``sizes``.
    """
    sizes = [int(s) for s in sizes]
    n = sum(sizes) + int(n_fixed_train)
    q = class_counts(n, ratios, min_one)
    if q[0] < n_fixed_train:                      # keep room for the reference in train
        for i in range(len(q) - 1, 0, -1):
            while q[0] < n_fixed_train and q[i] > 0:
                q[i] -= 1
                q[0] += 1
    left = sizes[:]
    out = [[0] * len(q) for _ in sizes]
    for s in range(1, len(q)):
        part = _capped_largest_remainder(q[s], sizes, left)
        for b, k in enumerate(part):
            out[b][s] = k
            left[b] -= k
    for b in range(len(sizes)):
        out[b][0] = left[b]
    return out


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
