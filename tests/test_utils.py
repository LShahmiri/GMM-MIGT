from __future__ import annotations

import os

import pytest

from gmm_migt import list_images, split_counts
from gmm_migt.utils import list_classes


@pytest.mark.parametrize("n", list(range(0, 60)))
@pytest.mark.parametrize("ratios", [[0.8, 0.2], [0.5, 0.4, 0.1], [0.6, 0.3, 0.1], [0.7, 0.29, 0.01]])
def test_split_counts_sum_and_floor(n, ratios):
    c = split_counts(n, ratios)
    assert sum(c) == n and all(x >= 0 for x in c)
    for i in range(1, len(ratios)):
        assert c[i] == int(round(ratios[i] * n, 9) // 1)


def test_split_counts_thesis_guarantee():
    # Section 4.2.2: a bin of 10 samples gives 5 train, 4 test, 1 validation
    assert split_counts(10, [0.5, 0.4, 0.1]) == [5, 4, 1]


def test_split_counts_float_robust():
    # 0.29 * 100 == 28.999999999999996 in binary floating point
    assert split_counts(100, [0.71, 0.29]) == [71, 29]


def test_split_counts_errors():
    with pytest.raises(ValueError):
        split_counts(-1, [1.0])
    with pytest.raises(ValueError):
        split_counts(3, [])


def test_listing(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "file.txt").write_text("x")
    for name in ["z.PNG", "a.jpg", "notes.txt", "m.webp"]:
        (tmp_path / "a" / name).write_bytes(b"")
    (tmp_path / "a" / "dir.jpg").mkdir()
    assert list_classes(str(tmp_path)) == ["a", "b"]
    assert [os.path.basename(p) for p in list_images(str(tmp_path / "a"))] == ["a.jpg", "m.webp", "z.PNG"]
