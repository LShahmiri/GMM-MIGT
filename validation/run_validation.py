"""Validation of the GMM-MIGT implementation against thesis Chapter 5.

Run from the project root (requires ``pip install -e .[dev]``)::

    python validation/run_validation.py

Writes figures and ``results.json`` to ``validation/results/``. Optional:
``--migt`` also checks equivalence with an installed ``migt`` package.

Checks
------
V1  EM-GMM + PCA on real photographs (scikit-learn sample images):
    number of colours after EM-GMM, PCA components / CPV per channel,
    and a figure in the layout of thesis Figure 5.2.
V2  Binning decisions of Algorithm 1 for the histograms printed in
    thesis Figures 4.6, 4.8, 4.10, 4.12 and 5.4.
V3  (--migt) GMM-MIGT with preprocessing disabled == MIGT 1.1.1.
V4  Controlled synthetic experiment: classes whose images come from four
    latent sub-types observed under strong illumination / colour-cast /
    noise variation. Compares how representative the train/test/val
    subsets are (sub-type proportions) for Random, MIGT and GMM-MIGT.
    This is an illustration of the mechanism, *not* a reproduction of the
    thesis CNN results (which need the original datasets and Xception
    training).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

from gmm_migt import (  # noqa: E402
    GMMMIGTSplitter,
    PreprocessingConfig,
    RandomSplitter,
    adaptive_histogram,
    forced_quantile_split,
    preprocess_image,
)


# ----------------------------------------------------------------------
# V1
# ----------------------------------------------------------------------
def v1_real_images() -> dict:
    from sklearn.datasets import load_sample_images

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_sample_images()
    rows = []
    fig, axes = plt.subplots(len(data.images), 3, figsize=(9, 3.2 * len(data.images)))
    for r, (rgb, name) in enumerate(zip(data.images, data.filenames)):
        bgr = rgb[:, :, ::-1].copy()
        t = time.perf_counter()
        gmm_only, ginfo = preprocess_image(bgr, PreprocessingConfig(use_pca=False))
        full, info = preprocess_image(bgr)
        dt = time.perf_counter() - t
        n_in = len(np.unique(bgr.reshape(-1, 3), axis=0))
        n_gmm = len(np.unique(gmm_only.reshape(-1, 3), axis=0))
        rows.append({
            "image": os.path.basename(name),
            "shape": list(bgr.shape),
            "distinct_colours_original": n_in,
            "distinct_colours_after_em_gmm": n_gmm,
            "gmm_converged": ginfo.gmm.converged,
            "gmm_iterations": ginfo.gmm.n_iter,
            "pca_components_per_channel_BGR": info.pca.n_components,
            "pca_cpv_per_channel": [round(c, 4) for c in info.pca.cpv],
            "rows_per_channel": bgr.shape[0],
            "seconds_gmm_plus_pca": round(dt, 2),
        })
        for c, (im, title) in enumerate([(bgr, "Original"), (gmm_only, "EM-GMM (K=4)"),
                                         (full, "EM-GMM / PCA (CPV 95%)")]):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v1_em_gmm_pca_figure_5_2_layout.png"), dpi=110)
    plt.close(fig)
    return {"images": rows, "pass": all(r["distinct_colours_after_em_gmm"] <= 4
                                         and all(c >= 0.95 for c in r["pca_cpv_per_channel"]) for r in rows)}


# ----------------------------------------------------------------------
# V2
# ----------------------------------------------------------------------
def _build(segments, lo, hi, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for (a, b), n in segments:
        vals += list(a + (b - a) * (0.02 + 0.96 * rng.random(n)))
    vals = sorted(vals)
    vals[0], vals[-1] = lo, hi
    return [(str(i), float(v)) for i, v in enumerate(vals)]


def _even(lo, hi, counts):
    e = np.linspace(lo, hi, len(counts) + 1)
    return list(zip(zip(e[:-1], e[1:]), counts))


def v2_binning() -> dict:
    cases = {
        "Fig 4.6 Little leaf privet (4 bins accepted)": (_build(_even(0.04, 1.0, [19, 17, 45, 18]), 0.04, 1.0), [19, 17, 45, 18]),
        "Fig 4.8 Snow (4 fails -> 3 bins)": (_build([((0.33, 0.4975), 30), ((0.553, 0.665), 5), ((0.665, 0.777), 215),
                                                     ((0.777, 0.8325), 100), ((0.8325, 1.0), 269)], 0.33, 1.0), [30, 220, 369]),
        "Fig 4.10 Holarrhena pubescens (4 -> 3 -> 2 bins)": (_build([((0.18, 0.385), 33), ((0.385, 0.4533), 22), ((0.4533, 0.59), 25),
                                                                     ((0.59, 0.7267), 13), ((0.7267, 0.795), 2), ((0.795, 1.0), 4)], 0.18, 1.0), [80, 19]),
        "Fig 4.12 Allium ramosum (2 bins fail -> 3 equal portions)": (_build([((0.12, 0.56), 91), ((0.56, 1.0), 8)], 0.12, 1.0), [33, 33, 33]),
        "Fig 5.4b Dew, GMM-MIGT (4 bins accepted)": (_build(_even(0.15, 1.0, [82, 506, 106, 10]), 0.15, 1.0), [82, 506, 106, 10]),
        "Fig 5.4a Dew, MIGT (4 bins accepted)": (_build(_even(0.27, 1.0, [111, 405, 167, 14]), 0.27, 1.0), [111, 405, 167, 14]),
    }
    out = {}
    for name, (items, expected) in cases.items():
        attempts = []
        bins = adaptive_histogram(items, 4, 10, "thesis", attempts)
        got = [b.size for b in bins] if bins else [b.size for b in forced_quantile_split(items, 3)]
        out[name] = {"attempts": [(a.n_bins, a.sizes) for a in attempts], "final": got,
                     "expected": expected, "pass": got == expected}
    # The MIGT 1.1.1 default min_bin=13 would reject the published GMM-MIGT Dew histogram
    dew = cases["Fig 5.4b Dew, GMM-MIGT (4 bins accepted)"][0]
    a13 = []
    adaptive_histogram(dew, 4, 13, "thesis", a13)
    out["Dew GMM-MIGT with min_bin=13"] = {"attempts": [(a.n_bins, a.sizes, a.accepted) for a in a13],
                                          "note": "4-bin histogram rejected -> min_bin=10 is what the thesis used"}
    return {"cases": out, "pass": all(v.get("pass", True) for v in out.values())}


# ----------------------------------------------------------------------
# V3
# ----------------------------------------------------------------------
def v3_migt_equivalence(root: str) -> dict:
    try:
        import migt
    except ImportError:
        return {"skipped": "migt not installed"}
    results = []
    for min_bin in (10, 13):  # GMM-MIGT default 10, MIGT 1.1.1 default 13
        for seed in (0, 1, 42):
            with tempfile.TemporaryDirectory() as tmp:
                with redirect_stdout(io.StringIO()):
                    migt.MIGTSplitter(root, seed=seed, min_bin=min_bin, verbose=False).run(os.path.join(tmp, "old"))
                GMMMIGTSplitter(root, seed=seed, min_bin=min_bin, use_gmm=False, use_pca=False,
                                verbose=False).run(os.path.join(tmp, "new"))
                same = _tree(os.path.join(tmp, "old")) == _tree(os.path.join(tmp, "new"))
            results.append({"min_bin": min_bin, "seed": seed, "identical": same})
    return {"runs": results, "pass": all(r["identical"] for r in results)}


def _tree(root):
    out = {}
    for split in ("train", "test", "val"):
        d = os.path.join(root, split)
        if os.path.isdir(d):
            for cls in os.listdir(d):
                for f in os.listdir(os.path.join(d, cls)):
                    out[(cls, f)] = split
    return out


# ----------------------------------------------------------------------
# V4
# ----------------------------------------------------------------------
def _subtype_pattern(kind: int, size: int, rng) -> np.ndarray:
    """Binary structure of one of four latent sub-types (with random phase/scale)."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    f = rng.uniform(3, 5)
    ph = rng.uniform(0, 2 * np.pi)
    if kind == 0:
        m = np.sin(2 * np.pi * f * xx + ph) > 0                         # vertical stripes
    elif kind == 1:
        m = np.sin(2 * np.pi * f * yy + ph) > 0                         # horizontal stripes
    elif kind == 2:
        cx, cy = rng.uniform(0.35, 0.65, 2)
        m = (xx - cx) ** 2 + (yy - cy) ** 2 < rng.uniform(0.06, 0.1)    # one large blob
    else:
        m = (np.sin(2 * np.pi * f * xx + ph) * np.sin(2 * np.pi * f * yy)) > 0  # checker
    return m


def _make_synthetic(root: str, n_classes=3, per_type=(60, 30, 20, 10), size=48, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    labels = {}
    for c in range(n_classes):
        d = os.path.join(root, f"class_{c}")
        os.makedirs(d)
        fg, bg = rng.integers(40, 215, 3), rng.integers(40, 215, 3)
        order = [k for k, n in enumerate(per_type) for _ in range(n)]
        rng.shuffle(order)
        for i, kind in enumerate(order):
            m = _subtype_pattern(kind, size, rng)
            img = np.where(m[..., None], fg, bg).astype(float)
            img = img * rng.uniform(0.45, 1.4) + rng.normal(0, 25, 3)      # illumination + colour cast
            img = img + rng.normal(0, 18, img.shape)                        # sensor noise
            img = np.clip(img, 0, 255).astype(np.uint8)
            name = f"img_{i:04d}.png"
            cv2.imwrite(os.path.join(d, name), img)
            labels[(f"class_{c}", name)] = kind
    return labels


def _representativeness(result, labels, n_types=4) -> dict:
    """Mean total-variation distance between sub-type proportions of each split and of its class."""
    tvs, missing = [], 0
    for cls, r in result.classes.items():
        kinds = np.array([labels[(cls, os.path.basename(it.path))] for it in r.items])
        overall = np.bincount(kinds, minlength=n_types) / len(kinds)
        for s in result.split_names:
            k = np.array([labels[(cls, os.path.basename(it.path))] for it in r.items if it.split == s])
            if len(k) == 0:
                continue
            p = np.bincount(k, minlength=n_types) / len(k)
            tvs.append(0.5 * np.abs(p - overall).sum())
            missing += int(np.sum((p == 0) & (overall > 0)))
    return {"mean_tv": float(np.mean(tvs)), "subtypes_missing_from_a_split": missing}


def _bin_purity(result, labels) -> float:
    """Normalised MI between the MI-bin of each image and its latent sub-type (0 = unrelated)."""
    from sklearn.metrics import normalized_mutual_info_score

    scores = []
    for cls, r in result.classes.items():
        y, b = [], []
        for it in r.items:
            if it.bin is not None:
                y.append(labels[(cls, os.path.basename(it.path))])
                b.append(it.bin)
        if len(set(b)) > 1:
            scores.append(normalized_mutual_info_score(y, b))
    return float(np.mean(scores)) if scores else 0.0


def v4_synthetic(n_datasets: int = 5) -> dict:
    rows = {"Random": [], "MIGT (no preprocessing)": [], "GMM-MIGT": []}
    purity = {"MIGT (no preprocessing)": [], "GMM-MIGT": []}
    for ds in range(n_datasets):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "data")
            labels = _make_synthetic(root, seed=100 + ds)
            common = dict(train=0.5, test=0.4, val=0.1, seed=ds)
            rnd = RandomSplitter(root, verbose=False, **common).plan()
            migt_like = GMMMIGTSplitter.from_thesis(root, use_gmm=False, use_pca=False, verbose=False, **common).plan()
            gmm = GMMMIGTSplitter.from_thesis(root, verbose=False, **common).plan()
            for name, res in (("Random", rnd), ("MIGT (no preprocessing)", migt_like), ("GMM-MIGT", gmm)):
                rows[name].append(_representativeness(res, labels))
            purity["MIGT (no preprocessing)"].append(_bin_purity(migt_like, labels))
            purity["GMM-MIGT"].append(_bin_purity(gmm, labels))
    summary = {}
    for name, vals in rows.items():
        tv = [v["mean_tv"] for v in vals]
        summary[name] = {
            "mean_total_variation": round(float(np.mean(tv)), 4),
            "std_total_variation": round(float(np.std(tv)), 4),
            "subtypes_missing_total": int(sum(v["subtypes_missing_from_a_split"] for v in vals)),
        }
        if name in purity:
            summary[name]["bin_vs_subtype_nmi"] = round(float(np.mean(purity[name])), 4)
    return {"n_datasets": n_datasets, "summary": summary,
            "note": "Lower total variation = split sub-type proportions closer to the class proportions."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--migt", action="store_true", help="also check equivalence with installed migt")
    ap.add_argument("--n-datasets", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    results = {}
    print("V1 EM-GMM + PCA on real images ...")
    results["V1_real_images"] = v1_real_images()
    print("V2 binning decisions of the thesis figures ...")
    results["V2_binning"] = v2_binning()
    if args.migt:
        print("V3 equivalence with MIGT 1.1.1 ...")
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "d")
            _make_synthetic(root, n_classes=3, seed=7)
            results["V3_migt_equivalence"] = v3_migt_equivalence(root)
    print("V4 synthetic representativeness experiment ...")
    results["V4_synthetic"] = v4_synthetic(args.n_datasets)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))
    return 0 if all(v.get("pass", True) for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
