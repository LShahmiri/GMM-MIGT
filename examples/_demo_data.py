"""Create a small synthetic image dataset so that every example runs out of the box.

Each example accepts an optional dataset path as first argument; without it
a demo dataset is generated in a temporary directory.
"""

from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np


def make_demo_dataset(root: str, classes=(("leaf", 80), ("flower", 50), ("bark", 9)),
                      size=64, gray: bool = False, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    for c, (name, n) in enumerate(classes):
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        fg, bg = rng.integers(40, 215, 3), rng.integers(40, 215, 3)
        for i in range(n):
            f = rng.uniform(2, 6)
            mask = np.sin(2 * np.pi * f * (xx * np.cos(c + i) + yy * np.sin(c + i))) > 0
            img = np.where(mask[..., None], fg, bg) * rng.uniform(0.5, 1.3) + rng.normal(0, 20, (size, size, 3))
            img = np.clip(img, 0, 255).astype(np.uint8)
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(d, f"{name}_{i:03d}.png"), img)
    return root


def dataset_from_argv(gray: bool = False) -> str:
    """Dataset path from ``sys.argv[1]``, or a freshly generated demo dataset."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    root = os.path.join(tempfile.mkdtemp(prefix="gmm_migt_demo_"), "dataset")
    print(f"No dataset given - generating a demo dataset in {root}")
    return make_demo_dataset(root, gray=gray)


def output_dir(name: str) -> str:
    return os.path.join(tempfile.mkdtemp(prefix="gmm_migt_out_"), name)
