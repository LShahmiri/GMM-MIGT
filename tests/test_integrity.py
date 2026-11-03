"""No image may be lost or duplicated: every file is in exactly one split or reported as skipped."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from gmm_migt import ConfigurationError, GMMMIGTSplitter, IntegrityError, RandomSplitter
from gmm_migt.report import ItemRecord

SIZES = [1, 2, 3, 4, 5, 6, 7, 9, 13, 14, 20, 33, 47]


def output_files(root):
    """``{(class, file): [splits]}`` of every image copied under ``root``."""
    out = {}
    for split in ("train", "test", "val"):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for cls in os.listdir(d):
            for f in os.listdir(os.path.join(d, cls)):
                out.setdefault((cls, f), []).append(split)
    return out


def input_files(root):
    return {(c, f) for c in os.listdir(root) if os.path.isdir(os.path.join(root, c))
            for f in os.listdir(os.path.join(root, c))}


@pytest.fixture
def many_sizes(make_ds):
    root = make_ds({f"c{n:02d}": n for n in SIZES})
    os.makedirs(os.path.join(root, "empty"))
    return root


@pytest.mark.parametrize("convention", ["migt", "thesis"])
@pytest.mark.parametrize("strategy", ["mi_quantile", "random"])
@pytest.mark.parametrize("val", [None, 0.1])
@pytest.mark.parametrize("min_bin", [1, 2, 4, 10, 13])
def test_every_image_assigned_exactly_once(many_sizes, min_bin, val, strategy, convention):
    train, test = (0.8, 0.2) if val is None else (0.5, 0.4)
    res = GMMMIGTSplitter(many_sizes, min_bin=min_bin, train=train, test=test, val=val,
                          low_sample_strategy=strategy, histogram_convention=convention,
                          use_gmm=False, use_pca=False, verbose=False).plan()
    for n in SIZES:
        r = res.classes[f"c{n:02d}"]
        paths = [it.path for it in r.items]
        assert len(paths) == len(set(paths)) == n          # nothing lost, nothing twice
        assert r.items[0].role == "reference" and r.items[0].split == "train"
        assert sum(r.counts.values()) == n
    assert res.accounting() == {"found": sum(SIZES), "assigned": sum(SIZES), "skipped": 0}


@pytest.mark.parametrize("val", [None, 0.1])
def test_copied_output_matches_input(many_sizes, tmp_path, val):
    train, test = (0.8, 0.2) if val is None else (0.5, 0.4)
    out = str(tmp_path / "out")
    res = GMMMIGTSplitter(many_sizes, train=train, test=test, val=val, verbose=False).run(out)
    copied = output_files(out)
    assert set(copied) == input_files(many_sizes)
    assert all(len(splits) == 1 for splits in copied.values())  # no image in two splits
    assert sum(res.totals().values()) == len(copied) == sum(SIZES)
    assert "[OK]" in res.summary()


def test_random_splitter_keeps_every_image(many_sizes, tmp_path):
    out = str(tmp_path / "out")
    RandomSplitter(many_sizes, train=0.5, test=0.4, val=0.1, verbose=False).run(out)
    copied = output_files(out)
    assert set(copied) == input_files(many_sizes) and all(len(s) == 1 for s in copied.values())


def test_skipped_images_are_reported_and_warned(make_ds, tmp_path):
    root = make_ds({"c": 30})
    d = os.path.join(root, "c")
    cv2.imwrite(os.path.join(d, "zz_big.png"), np.random.default_rng(0).integers(0, 256, (40, 30, 3), dtype=np.uint8))
    with open(os.path.join(d, "zz_broken.jpg"), "wb") as f:
        f.write(b"broken")
    with pytest.warns(UserWarning, match="2 image\\(s\\) could not be MI-scored.*resize_to"):
        res = GMMMIGTSplitter(root, verbose=False).run(str(tmp_path / "out"))
    assert res.accounting() == {"found": 32, "assigned": 30, "skipped": 2}
    assert len(output_files(str(tmp_path / "out"))) == 30
    assert "32 image(s) found = 30 assigned + 2 skipped [OK]" in res.summary()


def test_no_skip_warning_without_skipped_images(dataset, recwarn):
    GMMMIGTSplitter(dataset, verbose=False).plan()
    assert not [w for w in recwarn if "MI-scored" in str(w.message)]


def test_warning_when_a_class_gets_no_test_image(make_ds):
    root = make_ds({"five": 5})
    # 4 MI values -> forced quantile parts of 1, 1, 2 images -> floor(0.2 * k) = 0 test images
    with pytest.warns(UserWarning, match=r"No image assigned to: five \(test\)"):
        res = GMMMIGTSplitter(root, min_bin=4, use_gmm=False, use_pca=False, verbose=False).plan()
    assert res.classes["five"].counts["test"] == 0 and sum(res.classes["five"].counts.values()) == 5


def test_reusing_output_directory_is_refused(dataset, tmp_path):
    out = str(tmp_path / "out")
    GMMMIGTSplitter(dataset, seed=1, verbose=False).run(out)
    with pytest.raises(ConfigurationError, match="already contains images"):
        GMMMIGTSplitter(dataset, seed=2, verbose=False).run(out)


def test_verify_detects_lost_duplicated_and_new_images(dataset):
    res = GMMMIGTSplitter(dataset, use_gmm=False, use_pca=False, verbose=False).plan()
    res.verify(dataset)

    lost = res.classes["alpha"].items.pop()
    with pytest.raises(IntegrityError, match="59 assigned"):
        res.verify()
    res.classes["alpha"].items.append(lost)

    res.classes["beta"].items.append(ItemRecord(path=res.classes["beta"].items[1].path, split="test"))
    with pytest.raises(IntegrityError, match="more than once"):
        res.verify()
    res.classes["beta"].items.pop()

    res.classes["gamma"].items[1].split = "val"  # val=None: no such split
    with pytest.raises(IntegrityError, match="unknown split"):
        res.verify()
    res.classes["gamma"].items[1].split = "test"

    cv2.imwrite(os.path.join(dataset, "alpha", "new.png"), np.zeros((24, 24, 3), np.uint8))
    with pytest.raises(IntegrityError, match="on disk"):
        res.verify(dataset)
    os.makedirs(os.path.join(dataset, "delta"))
    os.remove(os.path.join(dataset, "alpha", "new.png"))
    with pytest.raises(IntegrityError, match="delta"):
        res.verify(dataset)


def test_copy_to_refuses_a_tampered_plan(dataset, tmp_path):
    res = GMMMIGTSplitter(dataset, use_gmm=False, use_pca=False, verbose=False).plan()
    res.classes["alpha"].items.pop()
    with pytest.raises(IntegrityError):
        res.copy_to(str(tmp_path / "out"))
