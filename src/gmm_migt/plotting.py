"""Optional plotting of MI histograms (requires ``pip install gmm-migt[plot]``).

Produces bar charts in the style of thesis Figures 4.6-4.12 and 5.4: one bar
per accepted bin, labelled with its normalised MI interval and sample count.
"""

from __future__ import annotations

from typing import Any, Optional

from .report import ClassReport


def plot_class_histogram(report: ClassReport, ax: Optional[Any] = None, title: Optional[str] = None) -> Any:
    """Bar chart of the bin sizes of one class; returns the matplotlib Axes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("Plotting requires matplotlib: pip install gmm-migt[plot]") from exc

    if not report.bins:
        raise ValueError(f"Class {report.name!r} has no bins (strategy={report.strategy!r}).")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sizes = [b.size for b in report.bins]
    labels = [b.label or f"part {b.index + 1}" for b in report.bins]
    bars = ax.bar(range(len(sizes)), sizes)
    for rect, n in zip(bars, sizes):
        ax.annotate(str(n), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("normalised MI")
    ax.set_ylabel("number of images")
    ax.set_title(title or f"{report.name} ({report.strategy})")
    return ax
