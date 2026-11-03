"""MI histogram binning (MIGT, thesis Section 4.2.2 and Algorithm 1).

GMM-MIGT keeps the MIGT binning and sample selection unchanged (thesis
Section 5.2: "the binning strategy and sample selection process remain
consistent with MIGT"). The functions here are the binning methods of
``migt.MIGTSplitter`` exposed as standalone functions.

Two histogram conventions are supported:

``"migt"`` (default, MIGT 1.1.1 code)
    Equal-width bins over ``[min(MI), max(MI)]`` with left-closed intervals
    ``[a, b)`` (last bin ``[a, b]``). Empty bins are discarded before the
    ``min_bin`` test, so e.g. a 4-bin histogram with one empty bin may be
    accepted as 3 bins.

``"thesis"`` (thesis Algorithm 1 / Figures 4.6-4.12, 5.4)
    Equal-width bins with right-closed intervals ``(a, b]`` (first bin
    ``[a, b]``), as labelled in the thesis figures. Every bin, including an
    empty one, must hold at least ``min_bin`` samples ("any BL < MinBin").

Equal-width binning is invariant to the max-normalisation of Eq. 4.9, so the
bins are computed on the raw MI values and the normalised values are
reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HISTOGRAM_CONVENTIONS = ("migt", "thesis")

MIItem = Tuple[str, float]


@dataclass
class BinInfo:
    """One histogram bin (or quantile portion) of MI-scored items."""

    mi_min: float
    mi_max: float
    items: List[MIItem]
    lower: Optional[float] = None
    upper: Optional[float] = None

    @property
    def size(self) -> int:
        return len(self.items)


@dataclass
class HistogramAttempt:
    """Bin sizes of one tried histogram (for reporting, cf. Figure 4.10)."""

    n_bins: int
    sizes: List[int]
    edges: List[float] = field(default_factory=list)
    accepted: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {"n_bins": self.n_bins, "sizes": self.sizes, "edges": self.edges, "accepted": self.accepted}


def _check_convention(convention: str) -> None:
    if convention not in HISTOGRAM_CONVENTIONS:
        raise ValueError(f"histogram_convention must be one of {HISTOGRAM_CONVENTIONS} (got {convention!r}).")


def normalise_mi(values: Sequence[float]) -> np.ndarray:
    """Max-normalisation of Eq. 4.9: ``MI_i / max(MI)``; zeros if ``max <= 0``."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    m = float(v.max())
    return v / m if m > 0 else np.zeros_like(v)


def histogram_bins(
    mi_list: Sequence[MIItem], num_bins: int, convention: str = "migt"
) -> List[BinInfo]:
    """Equal-width ("value-based") histogram of the MI values.

    With ``convention="migt"`` only non-empty bins are returned; with
    ``"thesis"`` all ``num_bins`` bins are returned, including empty ones.
    If all MI values are equal a single bin is returned.
    """
    _check_convention(convention)
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1 (got {num_bins}).")
    if len(mi_list) == 0:
        return []
    values = np.array([mi for _, mi in mi_list], dtype=float)
    lo, hi = float(values.min()), float(values.max())

    if lo == hi:
        # all MI equal => single bin
        return [BinInfo(lo, hi, list(mi_list), lo, hi)]

    edges = np.linspace(lo, hi, num_bins + 1)
    # side="right" -> [a, b) bins; side="left" -> (a, b] bins. min/max always
    # fall in the first/last bin because only interior edges are searched.
    side = "right" if convention == "migt" else "left"
    idx = np.searchsorted(edges[1:-1], values, side=side)

    out: List[BinInfo] = []
    for i in range(num_bins):
        items = [item for item, k in zip(mi_list, idx) if k == i]
        if items:
            mi_vals = [mi for _, mi in items]
            out.append(BinInfo(min(mi_vals), max(mi_vals), items, float(edges[i]), float(edges[i + 1])))
        elif convention == "thesis":
            out.append(BinInfo(float("nan"), float("nan"), [], float(edges[i]), float(edges[i + 1])))
    return out


def adaptive_histogram(
    mi_list: Sequence[MIItem],
    start_bins: int = 4,
    min_bin: int = 10,
    convention: str = "migt",
    attempts: Optional[List[HistogramAttempt]] = None,
) -> Optional[List[BinInfo]]:
    """Adaptive bin reduction ``start_bins -> ... -> 2`` (Algorithm 1, lines 3-12).

    Returns the first histogram in which every bin holds at least
    ``min_bin`` items, or ``None`` if even the 2-bin histogram fails (the
    caller then applies the forced quantile split). If ``attempts`` is a list,
    one :class:`HistogramAttempt` per tried bin count is appended to it.
    """
    _check_convention(convention)
    for b in range(start_bins, 1, -1):
        bins = histogram_bins(mi_list, b, convention)
        attempt = HistogramAttempt(b, [x.size for x in bins], _edges(mi_list, b))
        if attempts is not None:
            attempts.append(attempt)

        # bins count can drop due to empty bins (migt) => must still respect min_bin
        if len(bins) <= 1:
            continue

        if all(x.size >= min_bin for x in bins):
            attempt.accepted = True
            return bins

    return None


def _edges(mi_list: Sequence[MIItem], num_bins: int) -> List[float]:
    if len(mi_list) == 0:
        return []
    values = [mi for _, mi in mi_list]
    lo, hi = float(min(values)), float(max(values))
    if lo == hi:
        return [lo, hi]
    return [float(e) for e in np.linspace(lo, hi, num_bins + 1)]


def forced_quantile_split(mi_list: Sequence[MIItem], n_parts: int = 3) -> List[BinInfo]:
    """Sort by MI ascending and cut into ``n_parts`` equal portions.

    This is the MIGT fallback used when the 2-bin histogram still has a bin
    below ``min_bin`` (thesis Section 4.2.2: three equal portions were found
    empirically to work best). Portion ``i`` holds the sorted items
    ``[i*n//n_parts, (i+1)*n//n_parts)``. With fewer than ``n_parts`` items a
    single portion is returned.
    """
    if n_parts < 2:
        raise ValueError(f"n_parts must be >= 2 (got {n_parts}).")
    mi_sorted = sorted(mi_list, key=lambda x: x[1])
    n = len(mi_sorted)
    if n == 0:
        return []
    if n < n_parts:
        vals = [mi for _, mi in mi_sorted]
        return [BinInfo(min(vals), max(vals), mi_sorted)]

    cuts = [i * n // n_parts for i in range(n_parts + 1)]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        chunk = mi_sorted[a:b]
        vals = [mi for _, mi in chunk]
        out.append(BinInfo(min(vals), max(vals), chunk))
    return out


def interval_labels(edges: Sequence[float], convention: str = "thesis", decimals: int = 2) -> List[str]:
    """Human-readable interval labels, e.g. ``['[0.04, 0.28]', '(0.28, 0.52]']``.

    Pass normalised edges to obtain labels in the style of the thesis figures.
    """
    _check_convention(convention)
    n = len(edges) - 1
    labels = []
    for i in range(n):
        a, b = f"{edges[i]:.{decimals}f}", f"{edges[i + 1]:.{decimals}f}"
        if convention == "thesis":
            labels.append(f"{'[' if i == 0 else '('}{a}, {b}]")
        else:
            labels.append(f"[{a}, {b}{']' if i == n - 1 else ')'}")
    return labels
