"""Shared fixtures: small synthetic image datasets written to a temp directory."""

from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np
import pytest


def synthetic_image(rng: np.random.Generator, base: np.ndarray, alpha: float, size=(24, 24), gray=False) -> np.ndarray:
    """Blend ``base`` with random noise; ``alpha`` controls similarity to ``base``."""
    h, w = size
    noise = rng.integers(0, 256, size=(h, w, 3)).astype(np.float64)
    img = np.clip(alpha * base + (1 - alpha) * noise, 0, 255).astype(np.uint8)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def write_class(root: str, name: str, n: int, seed: int, size=(24, 24), gray=False,
                ext: str = ".png") -> list:
    rng = np.random.default_rng(seed)
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    h, w = size
    # smooth structured base pattern
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([
        127 + 120 * np.sin(xx / (2 + seed % 3)),
        127 + 120 * np.cos(yy / (3 + seed % 2)),
        127 + 120 * np.sin((xx + yy) / 4.0),
    ], axis=-1)
    paths = []
    for i in range(n):
        alpha = 0.15 + 0.8 * rng.random()
        img = synthetic_image(rng, base, alpha, size, gray)
        p = os.path.join(d, f"img_{i:03d}{ext}")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


def make_dataset(root: str, spec: Dict[str, int], size=(24, 24), gray=False) -> str:
    for k, (name, n) in enumerate(sorted(spec.items())):
        write_class(root, name, n, seed=k + 1, size=size, gray=gray)
    return root


@pytest.fixture
def dataset(tmp_path) -> str:
    """Three classes: large (60), medium (30) and small (8)."""
    return make_dataset(str(tmp_path / "data"), {"alpha": 60, "beta": 30, "gamma": 8})


@pytest.fixture
def make_ds(tmp_path):
    counter = {"i": 0}

    def _make(spec: Dict[str, int], size=(24, 24), gray=False, name: Optional[str] = None) -> str:
        counter["i"] += 1
        root = str(tmp_path / (name or f"ds{counter['i']}"))
        return make_dataset(root, spec, size=size, gray=gray)

    return _make
