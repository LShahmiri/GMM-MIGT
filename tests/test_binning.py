"""MIGT binning, validated against the histograms reported in the thesis.

The MI values below are constructed so that the equal-width histograms have
exactly the bin counts printed in thesis Figures 4.6, 4.8, 4.10, 4.12 and
5.4. The tests check that the adaptive procedure of Algorithm 1 takes the
same decisions as reported there.
"""

from __future__ import annotations

import numpy as np
import pytest

from gmm_migt import adaptive_histogram, forced_quantile_split, histogram_bins, interval_labels, normalise_mi
from gmm_migt.binning import HistogramAttempt


def build(segments, lo, hi, seed=0):
    """MI items with ``count`` values strictly inside each ``(a, b)`` segment, plus exact lo/hi."""
    rng = np.random.default_rng(seed)
    vals = []
    for (a, b), count in segments:
        vals += list(a + (b - a) * (0.02 + 0.96 * rng.random(count)))
    vals = sorted(vals)
    vals[0], vals[-1] = lo, hi  # min / max define the histogram range
    return [(f"img{i}", float(v)) for i, v in enumerate(vals)]


def sizes(bins):
    return [b.size for b in bins]


def even(lo, hi, n):
    e = np.linspace(lo, hi, n + 1)
    return list(zip(e[:-1], e[1:]))


# Figure 4.10 - Holarrhena pubescens (VNPlant-200): 4 bins [33,47,15,4] -> 3 bins [55,38,6] -> 2 bins [80,19]
HOLARRHENA = [((0.18, 0.385), 33), ((0.385, 0.4533), 22), ((0.4533, 0.59), 25),
              ((0.59, 0.7267), 13), ((0.7267, 0.795), 2), ((0.795, 1.0), 4)]


@pytest.mark.parametrize("convention", ["migt", "thesis"])
def test_figure_4_10_adaptive_reduction(convention):
    items = build(HOLARRHENA, 0.18, 1.0)
    assert sizes(histogram_bins(items, 4, convention)) == [33, 47, 15, 4]
    assert sizes(histogram_bins(items, 3, convention)) == [55, 38, 6]
    attempts = []
    bins = adaptive_histogram(items, 4, 10, convention, attempts)
    assert sizes(bins) == [80, 19]
    assert [(a.n_bins, a.sizes, a.accepted) for a in attempts] == [
        (4, [33, 47, 15, 4], False), (3, [55, 38, 6], False), (2, [80, 19], True)]


def test_figure_4_6_four_bins_accepted():
    # Little leaf privet (BJFU100): [19, 17, 45, 18]
    items = build(list(zip(even(0.04, 1.0, 4), [19, 17, 45, 18])), 0.04, 1.0)
    for conv in ("migt", "thesis"):
        assert sizes(adaptive_histogram(items, 4, 10, conv)) == [19, 17, 45, 18]


def test_figure_4_8_snow_three_bins():
    # Snow (WEAPD): 4-bin histogram has a bin < 10, 3-bin histogram [30, 220, 369] is accepted
    segs = [((0.33, 0.4975), 30), ((0.553, 0.665), 5), ((0.665, 0.777), 215),
            ((0.777, 0.8325), 100), ((0.8325, 1.0), 269)]
    items = build(segs, 0.33, 1.0)
    attempts = []
    bins = adaptive_histogram(items, 4, 10, "thesis", attempts)
    assert attempts[0].sizes == [30, 5, 315, 269] and not attempts[0].accepted
    assert sizes(bins) == [30, 220, 369]


def test_figure_4_12_falls_back_to_three_equal_portions():
    # Allium ramosum (VNPlant-200): 2-bin histogram [91, 8] -> forced 3 equal MI portions
    items = build([((0.12, 0.56), 91), ((0.56, 1.0), 8)], 0.12, 1.0)
    assert sizes(histogram_bins(items, 2)) == [91, 8]
    assert adaptive_histogram(items, 4, 10, "thesis") is None
    parts = forced_quantile_split(items, 3)
    assert sizes(parts) == [33, 33, 33]
    assert max(mi for _, mi in parts[0].items) <= min(mi for _, mi in parts[1].items)


def test_figure_5_4_gmm_migt_dew_requires_min_bin_10():
    # Dew (WEAPD) with GMM-MIGT: 4 bins [82, 506, 106, 10] are used in the thesis.
    items = build(list(zip(even(0.15, 1.0, 4), [82, 506, 106, 10])), 0.15, 1.0)
    assert sizes(adaptive_histogram(items, 4, 10, "thesis")) == [82, 506, 106, 10]
    # MIGT 1.1.1's default min_bin=13 would reject this histogram
    attempts = []
    adaptive_histogram(items, 4, 13, "thesis", attempts)
    assert attempts[0].n_bins == 4 and not attempts[0].accepted
    # and the MIGT (no preprocessing) histogram of the same class [111, 405, 167, 14]
    items = build(list(zip(even(0.27, 1.0, 4), [111, 405, 167, 14])), 0.27, 1.0)
    assert sizes(adaptive_histogram(items, 4, 10, "thesis")) == [111, 405, 167, 14]


def test_empty_bin_conventions():
    rng = np.random.default_rng(1)
    vals = list(rng.uniform(0.0, 0.2, 20)) + list(rng.uniform(0.27, 0.32, 20)) + list(rng.uniform(0.8, 0.99, 19))
    vals = [0.0] + vals[1:] + [1.0]
    items = [(str(i), v) for i, v in enumerate(vals)]
    # migt: the empty 4th-quarter bin [0.5, 0.75) is dropped -> 3 non-empty bins accepted
    migt = adaptive_histogram(items, 4, 13, "migt")
    assert len(migt) == 3
    # thesis: an empty bin is a bin with < MinBin samples -> reduce to 3 (empty middle) -> 2
    thesis = adaptive_histogram(items, 4, 13, "thesis")
    assert sizes(thesis) == [40, 20]
    assert len(histogram_bins(items, 4, "thesis")) == 4


def test_interval_closure_conventions():
    items = [("a", 0.0), ("b", 0.5), ("c", 1.0)]
    assert sizes(histogram_bins(items, 2, "migt")) == [1, 2]    # [0, .5) [.5, 1]
    assert sizes(histogram_bins(items, 2, "thesis")) == [2, 1]  # [0, .5] (.5, 1]


def test_migt_histogram_matches_migt_1_1_1_rule():
    """Value ``mi`` is in bin i iff ``a <= mi < b`` (or last bin and ``mi == b``) - MIGT 1.1.1."""
    rng = np.random.default_rng(3)
    items = [(str(i), float(v)) for i, v in enumerate(rng.random(200))]
    for nb in (2, 3, 4, 5):
        values = np.array([v for _, v in items])
        edges = np.linspace(values.min(), values.max(), nb + 1)
        expected = []
        for i in range(nb):
            a, b = edges[i], edges[i + 1]
            got = [(p, mi) for p, mi in items if (a <= mi < b) or (i == nb - 1 and mi == b)]
            if got:
                expected.append(got)
        assert [b.items for b in histogram_bins(items, nb, "migt")] == expected


def test_degenerate_inputs():
    assert histogram_bins([], 4) == []
    same = [(str(i), 0.5) for i in range(30)]
    assert len(histogram_bins(same, 4)) == 1
    assert adaptive_histogram(same, 4, 10) is None
    assert sizes(forced_quantile_split(same[:2], 3)) == [2]
    assert forced_quantile_split([], 3) == []
    hundred = [(str(i), i / 100) for i in range(100)]
    assert sizes(forced_quantile_split(hundred, 3)) == [33, 33, 34]
    assert sizes(forced_quantile_split(hundred[:11], 2)) == [5, 6]
    with pytest.raises(ValueError):
        histogram_bins(same, 4, "excel")
    with pytest.raises(ValueError):
        forced_quantile_split(same, 1)


def test_normalisation_eq_4_9():
    assert np.allclose(normalise_mi([1.0, 2.0, 4.0]), [0.25, 0.5, 1.0])
    assert np.allclose(normalise_mi([0.0, 0.0]), [0.0, 0.0])
    assert normalise_mi([]).size == 0


def test_bins_invariant_to_max_normalisation():
    rng = np.random.default_rng(4)
    raw = rng.uniform(0.02, 0.4, 150)
    norm = normalise_mi(raw)
    a = [(str(i), float(v)) for i, v in enumerate(raw)]
    b = [(str(i), float(v)) for i, v in enumerate(norm)]
    for nb in (2, 3, 4):
        assert [[p for p, _ in x.items] for x in histogram_bins(a, nb, "thesis")] == \
               [[p for p, _ in x.items] for x in histogram_bins(b, nb, "thesis")]


def test_interval_labels_match_thesis_figures():
    assert interval_labels([0.04, 0.28, 0.52, 0.76, 1.0]) == [
        "[0.04, 0.28]", "(0.28, 0.52]", "(0.52, 0.76]", "(0.76, 1.00]"]
    assert interval_labels([0.0, 0.5, 1.0], "migt") == ["[0.00, 0.50)", "[0.50, 1.00]"]


def test_attempt_serialisation():
    a = HistogramAttempt(3, [1, 2, 3], [0.0, 0.3, 0.6, 1.0], True)
    assert a.to_dict() == {"n_bins": 3, "sizes": [1, 2, 3], "edges": [0.0, 0.3, 0.6, 1.0], "accepted": True}
