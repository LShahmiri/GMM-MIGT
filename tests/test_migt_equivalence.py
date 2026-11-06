"""GMM-MIGT with preprocessing disabled must reproduce MIGT 1.1.1 exactly.

GMM-MIGT = MIGT + (EM-GMM, PCA). With ``use_gmm=False, use_pca=False`` and
the MIGT 1.1.1 defaults, every MI score and every train/test assignment must
be identical to the original ``migt`` package. This test runs only if
``migt`` is installed (it is *not* a dependency of gmm-migt).
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

from gmm_migt import GMMMIGTSplitter, load_image
from gmm_migt import compute_mi as gmm_compute_mi

migt = pytest.importorskip("migt")


def _legacy_kwargs():
    """migt < 1.2 rounds per bin, uses mi_quantile for small classes and drops unscored images."""
    from importlib.metadata import version

    v = getattr(migt, "__version__", None) or version("migt")
    major, minor = (int(x) for x in v.split(".")[:2])
    if (major, minor) >= (1, 2):
        return {}
    return dict(rounding="per_bin", low_sample_strategy="mi_quantile", unscored="skip")


def tree(root):
    out = {}
    for split in ("train", "test", "val"):
        d = os.path.join(root, split)
        if os.path.isdir(d):
            for cls in sorted(os.listdir(d)):
                for f in os.listdir(os.path.join(d, cls)):
                    out[(cls, f)] = split
    return out


@pytest.mark.parametrize("min_bin", [10, 13])
@pytest.mark.parametrize("val", [None, 0.1])
@pytest.mark.parametrize("seed", [0, 42])
def test_identical_partition_without_preprocessing(make_ds, tmp_path, seed, val, min_bin):
    root = make_ds({"a": 70, "b": 45, "c": 33, "d": 4})
    train, test = (0.8, 0.2) if val is None else (0.6, 0.3)
    common = dict(mode="auto", bins=4, min_bin=min_bin, train=train, test=test, val=val, seed=seed, verbose=False)

    out_old = str(tmp_path / f"old{seed}{val}{min_bin}")
    with redirect_stdout(io.StringIO()):
        migt.MIGTSplitter(root, **common).run(out_old)
    out_new = str(tmp_path / f"new{seed}{val}{min_bin}")
    GMMMIGTSplitter(root, method="migt", histogram_convention="migt", **_legacy_kwargs(), **common).run(out_new)

    assert tree(out_old) == tree(out_new)


def test_identical_mi_scores(make_ds):
    from migt.mi import compute_mi as migt_compute_mi
    from migt.mi import load_image as migt_load_image

    root = make_ds({"a": 12})
    paths = sorted(os.path.join(root, "a", f) for f in os.listdir(os.path.join(root, "a")))
    ref_old, ref_new = migt_load_image(paths[0]), load_image(paths[0])
    assert np.array_equal(ref_old, ref_new)
    for p in paths[1:]:
        for mode in ("auto", "grayscale", "color"):
            assert gmm_compute_mi(ref_new, load_image(p), mode) == migt_compute_mi(ref_old, migt_load_image(p), mode)
