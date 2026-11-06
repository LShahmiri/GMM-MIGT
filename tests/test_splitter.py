"""End-to-end tests of GMMMIGTSplitter on synthetic datasets."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from gmm_migt import ConfigurationError, GMMMIGTSplitter, SplitResult
from gmm_migt.report import REPORT_FILENAME

from .conftest import write_class


def files_in(root):
    out = {}
    for split in sorted(os.listdir(root)):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for cls in sorted(os.listdir(d)):
            out[(split, cls)] = sorted(os.listdir(os.path.join(d, cls)))
    return out


def assignment(result: SplitResult):
    return {(c, os.path.basename(it.path)): it.split for c, r in result.classes.items() for it in r.items}


# ---------------------------------------------------------------- partition
def test_run_creates_complete_partition(dataset, tmp_path):
    out = str(tmp_path / "out")
    res = GMMMIGTSplitter(dataset, verbose=False).run(out)
    tree = files_in(out)
    assert set(s for s, _ in tree) == {"train", "test"}  # val=None by default (as in MIGT)
    for cls in ("alpha", "beta", "gamma"):
        n = len(os.listdir(os.path.join(dataset, cls)))
        got = tree[("train", cls)] + tree[("test", cls)]
        assert sorted(got) == sorted(os.listdir(os.path.join(dataset, cls)))  # no loss
        assert len(got) == len(set(got)) == n  # no duplicates / leakage
    # MI-guided classes: reference image (first file) always in train
    for cls in ("alpha", "beta"):
        r = res.classes[cls]
        assert r.items[0].role == "reference" and r.items[0].split == "train"
        assert os.path.basename(r.reference) == "img_000.png"
    assert res.classes["gamma"].strategy == "low_sample_random"   # 8 <= min_bin: random (papers)
    assert os.path.isfile(os.path.join(out, REPORT_FILENAME))


def test_bin_labels_follow_convention(make_ds):
    root = make_ds({"big": 150})
    for conv, first, last in (("migt", "[", "]"), ("thesis", "[", "]")):
        res = GMMMIGTSplitter(root, histogram_convention=conv, min_bin=10, verbose=False).plan()
        r = res.classes["big"]
        labels = [b.label for b in r.bins]
        assert all(labels)
        if r.strategy == "histogram" and len(labels) > 1:
            assert labels[0][0] == first and labels[-1].endswith("1.00" + last)
            if conv == "migt":
                assert labels[0].endswith(")")
            else:
                assert labels[1].startswith("(")


def test_strategies_and_ratios(dataset):
    res = GMMMIGTSplitter(dataset, min_bin=13, low_sample_strategy="mi_quantile", rounding="per_bin",
                          verbose=False).plan()
    assert res.classes["alpha"].strategy in ("histogram", "forced_quantile")
    assert res.classes["gamma"].strategy == "low_sample_mi_quantile"  # 8 <= min_bin
    for b_idx, b in enumerate(res.classes["alpha"].bins):
        members = [it for it in res.classes["alpha"].items if it.bin == b_idx]
        n_train = sum(it.split == "train" for it in members)
        assert n_train == len(members) - int(np.floor(round(0.2 * len(members), 9)))
    if res.classes["alpha"].strategy == "histogram":
        assert all(b.size >= 13 for b in res.classes["alpha"].bins)


def test_three_way_split_thesis_preset(dataset, tmp_path):
    s = GMMMIGTSplitter.from_thesis(dataset, verbose=False)
    assert (s.min_bin, s.ratios, s.histogram_convention) == (10, [0.5, 0.4, 0.1], "thesis")
    assert (s.low_sample_strategy, s.rounding) == ("random", "per_bin")
    res = s.run(str(tmp_path / "out"))
    assert set(res.split_names) == {"train", "test", "val"}
    r = res.classes["alpha"]
    if r.strategy == "histogram":
        assert all(b.size >= 10 for b in r.bins)
        assert r.bins[0].label.startswith("[") and r.bins[-1].label.endswith("1.00]")
    assert os.path.isdir(str(tmp_path / "out" / "val" / "alpha"))
    # normalised MI (Eq. 4.9) has maximum 1
    norm = [it.normalized_mi for it in r.items if it.normalized_mi is not None]
    assert max(norm) == pytest.approx(1.0)


# ---------------------------------------------------------------- reproducibility
def test_reproducible_with_seed(dataset):
    a = GMMMIGTSplitter(dataset, verbose=False, seed=7).plan()
    b = GMMMIGTSplitter(dataset, verbose=False, seed=7).plan()
    assert assignment(a) == assignment(b)
    assert [it.mi for it in a.classes["alpha"].items] == [it.mi for it in b.classes["alpha"].items]


def test_seed_changes_split(dataset):
    a = GMMMIGTSplitter(dataset, verbose=False, seed=1).plan()
    b = GMMMIGTSplitter(dataset, verbose=False, seed=2).plan()
    assert assignment(a) != assignment(b)


def test_parallel_equals_serial(dataset):
    a = GMMMIGTSplitter(dataset, verbose=False, n_jobs=1).plan()
    b = GMMMIGTSplitter(dataset, verbose=False, n_jobs=2).plan()
    assert assignment(a) == assignment(b)
    assert [it.mi for it in a.classes["alpha"].items] == [it.mi for it in b.classes["alpha"].items]


def test_report_roundtrip(dataset, tmp_path):
    out = str(tmp_path / "out")
    res = GMMMIGTSplitter(dataset, verbose=False).run(out)
    loaded = SplitResult.from_json(os.path.join(out, REPORT_FILENAME))
    assert assignment(loaded) == assignment(res)
    assert loaded.config == res.config and loaded.version == res.version
    # a saved plan can be materialised again
    out2 = str(tmp_path / "out2")
    loaded.copy_to(out2)
    assert files_in(out) == files_in(out2)
    d = json.load(open(os.path.join(out, REPORT_FILENAME), encoding="utf-8"))
    assert d["method"] == "GMM-MIGT" and d["config"]["n_components"] == 4
    assert "TOTAL" in res.summary()


# ---------------------------------------------------------------- modes
@pytest.mark.parametrize("mode", ["auto", "grayscale", "color"])
def test_mi_modes(dataset, mode):
    res = GMMMIGTSplitter(dataset, mode=mode, verbose=False).plan()
    assert all(it.mi is not None for it in res.classes["alpha"].items[1:])


def test_grayscale_dataset_auto_and_color_error(make_ds):
    root = make_ds({"g1": 40}, gray=True)
    res = GMMMIGTSplitter(root, verbose=False).plan()
    assert res.classes["g1"].skipped == []
    with pytest.raises(ConfigurationError, match="grayscale"):
        GMMMIGTSplitter(root, mode="color", verbose=False).plan()


@pytest.mark.parametrize("use_gmm,use_pca", [(True, True), (True, False), (False, True), (False, False)])
def test_preprocessing_ablations(dataset, use_gmm, use_pca):
    res = GMMMIGTSplitter(dataset, use_gmm=use_gmm, use_pca=use_pca, verbose=False).plan()
    it = res.classes["alpha"].items[1]
    assert (it.gmm_converged is not None) == use_gmm
    assert (it.pca_components is not None) == use_pca


def test_preprocessing_changes_scores(dataset):
    a = GMMMIGTSplitter(dataset, verbose=False).plan()
    b = GMMMIGTSplitter(dataset, use_gmm=False, use_pca=False, verbose=False).plan()
    assert [i.mi for i in a.classes["alpha"].items[1:]] != [i.mi for i in b.classes["alpha"].items[1:]]


def test_custom_parameters_are_used(dataset):
    s = GMMMIGTSplitter(dataset, n_components=6, pca_variance=0.8, verbose=False)
    assert s.preprocessing.n_components == 6 and s.preprocessing.pca_variance == 0.8
    res = s.plan()
    assert res.config["n_components"] == 6 and res.config["pca_variance"] == 0.8


def test_color_mi_estimator(dataset):
    a = GMMMIGTSplitter(dataset, color_mi="mutual_info_score", verbose=False).plan()
    b = GMMMIGTSplitter(dataset, verbose=False).plan()
    mi_a = [it.mi for it in a.classes["alpha"].items[1:]]
    mi_b = [it.mi for it in b.classes["alpha"].items[1:]]
    assert all(1.0 <= v <= 2.0 for v in mi_b)  # skimage NMI range
    assert mi_a != mi_b and all(v >= 0 for v in mi_a)


def test_class_rounding_ratios(dataset):
    from gmm_migt import class_counts
    res = GMMMIGTSplitter(dataset, min_bin=13, train=0.7, test=0.2, val=0.1, verbose=False).plan()
    for name, r in res.classes.items():
        assert [r.counts[s] for s in ("train", "test", "val")] == class_counts(r.n_images, [0.7, 0.2, 0.1])


def test_low_sample_mi_quantile_uses_both_halves_for_test(make_ds):
    # 10 images = reference + 9; halves of 4 and 5 MI-sorted images. Versions <= 1.0.1 assigned
    # [ref] + low + high in order, so test/val came only from the high-MI half.
    root = make_ds({"small": 10})
    seen_low_in_test = False
    for seed in range(20):
        r = GMMMIGTSplitter(root, low_sample_strategy="mi_quantile", train=0.5, test=0.4, val=0.1,
                            method="migt", seed=seed, verbose=False).plan().classes["small"]
        assert r.strategy == "low_sample_mi_quantile"
        assert [r.counts[s] for s in ("train", "test", "val")] == [5, 4, 1]
        seen_low_in_test |= any(it.bin == 0 and it.split != "train" for it in r.items)
        legacy = GMMMIGTSplitter(root, low_sample_strategy="mi_quantile", train=0.5, test=0.4, val=0.1,
                                 method="migt", rounding="per_bin", seed=seed, verbose=False).plan()
        assert all(it.bin != 0 for it in legacy.classes["small"].items if it.split != "train")
    assert seen_low_in_test


def test_low_sample_random_is_random_and_complete(make_ds):
    root = make_ds({"small": 9})
    seen = set()
    for seed in range(15):
        r = GMMMIGTSplitter(root, train=0.7, test=0.2, val=0.1, seed=seed, verbose=False).plan().classes["small"]
        assert r.strategy == "low_sample_random" and r.reference is None
        assert len({it.path for it in r.items}) == 9
        assert [r.counts[s] for s in ("train", "test", "val")] == [6, 2, 1]     # val no longer 0
        seen.add(next(it.split for it in r.items if it.path.endswith("img_000.png")))
    assert seen != {"train"}                                    # the first file is not forced into train


def test_low_sample_random_never_duplicates_reference(make_ds, tmp_path):
    root = make_ds({"small": 9})
    for seed in range(15):
        out = str(tmp_path / f"o{seed}")
        res = GMMMIGTSplitter(root, min_bin=13, low_sample_strategy="random", rounding="per_bin",
                              seed=seed, verbose=False).run(out)
        tree = files_in(out)
        assert "img_000.png" in tree[("train", "small")]
        assert "img_000.png" not in tree[("test", "small")]
        assert len(tree[("train", "small")]) + len(tree[("test", "small")]) == 9
        assert res.classes["small"].strategy == "low_sample_random"


def test_class_size_boundary(make_ds):
    root = make_ds({"eq": 13, "above": 14})
    res = GMMMIGTSplitter(root, min_bin=13, verbose=False).plan()
    assert res.classes["eq"].strategy.startswith("low_sample")      # <= min_bin (MIGT rule)
    assert res.classes["above"].strategy == "forced_quantile"       # 13 MI values cannot fill 2 bins of 13


def test_default_min_bin_is_10_and_user_can_change_it(make_ds):
    root = make_ds({"eq": 10, "above": 11, "big": 150})
    s = GMMMIGTSplitter(root, verbose=False)
    assert (s.start_bins, s.min_bin) == (4, 10) and s.get_config()["min_bin"] == 10
    res = s.plan()
    assert res.classes["eq"].strategy.startswith("low_sample")      # 10 <= min_bin
    assert res.classes["above"].strategy == "forced_quantile"       # 10 MI values cannot fill 2 bins of 10
    big = res.classes["big"]
    assert [a["n_bins"] for a in big.histogram_attempts][0] == 4    # starts at 4 bins, then 3, 2
    if big.strategy == "histogram":
        assert all(b.size >= 10 for b in big.bins)
    assert GMMMIGTSplitter(root, min_bin=7, verbose=False).min_bin == 7


def test_tiny_and_empty_classes(make_ds, tmp_path):
    root = make_ds({"one": 1, "two": 2})
    os.makedirs(os.path.join(root, "empty"))
    out = str(tmp_path / "o")
    res = GMMMIGTSplitter(root, verbose=False).run(out)
    assert res.classes["empty"].strategy == "empty"
    assert os.path.isdir(os.path.join(out, "train", "empty"))  # no class skipping
    assert res.classes["one"].counts["train"] == 1
    assert res.classes["two"].strategy == "low_sample_random"
    assert res.classes["two"].counts == {"train": 1, "test": 1, "val": 0}
    legacy = GMMMIGTSplitter(root, low_sample_strategy="mi_quantile", verbose=False).plan()
    assert legacy.classes["two"].strategy == "low_sample_random_fallback"
    assert sum(legacy.classes["two"].counts.values()) == 2


def test_strict_shape_skips_and_resize_mode(make_ds):
    root = make_ds({"c": 30})
    cls_dir = os.path.join(root, "c")
    big = np.random.default_rng(0).integers(0, 256, (40, 30, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(cls_dir, "zz_big.png"), big)
    with open(os.path.join(cls_dir, "zz_broken.jpg"), "wb") as f:
        f.write(b"broken")
    with pytest.warns(UserWarning, match="2 image"):
        res = GMMMIGTSplitter(root, unscored="skip", verbose=False).plan()
    reasons = {os.path.basename(s["path"]): s["reason"] for s in res.classes["c"].skipped}
    assert set(reasons) == {"zz_big.png", "zz_broken.jpg"}
    assert "shape" in reasons["zz_big.png"].lower()
    with pytest.warns(UserWarning, match="1 image"):
        res2 = GMMMIGTSplitter(root, strict_shape=False, resize_to=(24, 24), unscored="skip",
                               verbose=False).plan()
    assert [os.path.basename(s["path"]) for s in res2.classes["c"].skipped] == ["zz_broken.jpg"]
    with pytest.warns(UserWarning, match="2 image"):
        res3 = GMMMIGTSplitter(root, verbose=False).plan()                    # default: include
    assert res3.classes["c"].skipped == [] and len(res3.classes["c"].items) == 32
    roles = {os.path.basename(it.path): it.role for it in res3.classes["c"].items}
    assert roles["zz_big.png"] == roles["zz_broken.jpg"] == "unscored"


def test_processed_images_saved(dataset, tmp_path):
    proc = str(tmp_path / "processed")
    GMMMIGTSplitter(dataset, processed_root=proc, verbose=False).plan()
    files = os.listdir(os.path.join(proc, "alpha"))
    assert len(files) == 60 and all(f.endswith(".png") for f in files)
    img = cv2.imread(os.path.join(proc, "alpha", "img_001.png"))
    assert img.shape == (24, 24, 3)


def test_verbose_output(dataset, capsys):
    GMMMIGTSplitter(dataset, verbose=True).plan()
    out = capsys.readouterr().out
    assert "Processing class: alpha" in out and "Using bins=" in out


def test_large_class_with_histogram(make_ds):
    root = make_ds({"big": 150})
    res = GMMMIGTSplitter.from_thesis(root, verbose=False).plan()
    r = res.classes["big"]
    assert r.histogram_attempts and r.histogram_attempts[0]["n_bins"] == 4
    assert sum(b.size for b in r.bins) == 149  # all non-reference images are binned
    c = r.counts
    assert c["train"] + c["test"] + c["val"] == 150


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("kw,msg", [
    ({"train": 0.5, "test": 0.4}, "val=None"),
    ({"train": 0.8, "test": 0.3}, "sum to 1"),
    ({"train": 0.0, "test": 1.0}, "> 0"),
    ({"mode": "rgb"}, "mode"),
    ({"bins": 1}, "bins"),
    ({"bins": 2.5}, "bins"),
    ({"min_bin": 0}, "min_bin"),
    ({"strict_shape": False}, "resize_to"),
    ({"strict_shape": False, "resize_to": (0, 5)}, "resize_to"),
    ({"low_sample_strategy": "x"}, "low_sample_strategy"),
    ({"color_mi": "x"}, "color_mi"),
    ({"histogram_convention": "x"}, "histogram_convention"),
    ({"fallback_quantiles": 1}, "fallback_quantiles"),
    ({"n_jobs": 0}, "n_jobs"),
    ({"seed": "a"}, "seed"),
    ({"n_components": 0}, "n_components"),
    ({"pca_variance": 2}, "pca_variance"),
])
def test_invalid_parameters(dataset, kw, msg):
    with pytest.raises(ConfigurationError, match=msg):
        GMMMIGTSplitter(dataset, **kw)


def test_invalid_paths(dataset, tmp_path):
    with pytest.raises(ConfigurationError):
        GMMMIGTSplitter(str(tmp_path / "missing"))
    s = GMMMIGTSplitter(dataset, verbose=False)
    with pytest.raises(ConfigurationError):
        s.run(dataset)
    with pytest.raises(ConfigurationError):
        s.run(os.path.join(dataset, "inside"))


def test_resize_ignored_warning(dataset):
    with pytest.warns(UserWarning, match="ignored"):
        GMMMIGTSplitter(dataset, resize_to=(10, 10))


def test_unreadable_reference_raises(make_ds):
    root = make_ds({"c": 20})
    with open(os.path.join(root, "c", "img_000.png"), "wb") as f:
        f.write(b"x")
    with pytest.raises(Exception, match="Reference image"):
        GMMMIGTSplitter(root, verbose=False).plan()


# ---------------------------------------------------------------- method option
def test_method_migt_equals_no_preprocessing(dataset):
    a = GMMMIGTSplitter(dataset, method="migt", verbose=False).plan()
    b = GMMMIGTSplitter(dataset, use_gmm=False, use_pca=False, verbose=False).plan()
    assert assignment(a) == assignment(b)
    assert [it.mi for it in a.classes["alpha"].items] == [it.mi for it in b.classes["alpha"].items]
    assert a.method == "MIGT" and a.config["method"] == "migt"
    assert a.config["use_gmm"] is False and a.config["use_pca"] is False
    g = GMMMIGTSplitter(dataset, verbose=False)
    assert g.method == "gmm_migt" and g.preprocessing.use_gmm and g.preprocessing.use_pca


@pytest.mark.parametrize("kw", [{"use_gmm": True}, {"use_pca": True}])
def test_method_migt_rejects_preprocessing(dataset, kw):
    with pytest.raises(ConfigurationError, match="method='migt'"):
        GMMMIGTSplitter(dataset, method="migt", **kw)


@pytest.mark.parametrize("kw,msg", [({"method": "x"}, "method"), ({"rounding": "x"}, "rounding"),
                                    ({"unscored": "x"}, "unscored")])
def test_invalid_new_options(dataset, kw, msg):
    with pytest.raises(ConfigurationError, match=msg):
        GMMMIGTSplitter(dataset, **kw)


def test_grayscale_photo_in_colour_class_is_scored(make_ds):
    root = make_ds({"c": 30})
    img = cv2.imread(os.path.join(root, "c", "img_005.png"))
    cv2.imwrite(os.path.join(root, "c", "img_005.png"), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    r = GMMMIGTSplitter(root, verbose=False).plan().classes["c"]
    it = next(i for i in r.items if i.path.endswith("img_005.png"))
    assert it.role == "sample" and it.mi is not None and 1.0 <= it.mi <= 2.0   # colour-NMI scale


def test_svd_failure_falls_back(monkeypatch):
    import gmm_migt.preprocessing as pp
    rng = np.random.default_rng(0)
    M = rng.random((24, 24)) * 255
    want = pp.pca_channel_reconstruction(M, 0.95)

    class Broken:
        def __init__(self, *a, **k):
            pass

        def fit_transform(self, X):
            raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(pp, "PCA", Broken)
    got = pp.pca_channel_reconstruction(M, 0.95)
    assert got[1] == want[1] and np.allclose(got[0], want[0]) and got[2] == pytest.approx(want[2])
