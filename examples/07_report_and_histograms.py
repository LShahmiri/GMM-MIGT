"""Inspect, save, reload and plot a partition report (needs gmm-migt[plot] for plotting).

    python examples/07_report_and_histograms.py [path/to/dataset]
"""

import os

from _demo_data import dataset_from_argv, output_dir

from gmm_migt import GMMMIGTSplitter, SplitResult

dataset = dataset_from_argv()
out = output_dir("gmm_migt_output")

result = GMMMIGTSplitter.from_thesis(dataset, processed_root=out + "_processed", verbose=False).run(out)

# JSON report with every MI score, bin and split
report_path = os.path.join(out, "gmm_migt_report.json")
reloaded = SplitResult.from_json(report_path)
print(reloaded.summary())

r = reloaded.classes["leaf"] if "leaf" in reloaded.classes else next(iter(reloaded.classes.values()))
for it in r.items[:5]:
    print(f"{os.path.basename(it.path)}: split={it.split} MI={it.mi} normalised={it.normalized_mi} bin={it.bin}")

# lists of files per split, e.g. to build your own data loader
splits = reloaded.splits()
print({s: sum(len(v) for v in d.values()) for s, d in splits.items()})

# Histogram in the style of the thesis figures
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gmm_migt.plotting import plot_class_histogram

    ax = plot_class_histogram(r)
    png = os.path.join(out, f"histogram_{r.name}.png")
    plt.savefig(png, bbox_inches="tight")
    print("histogram saved to", png)
except ImportError:
    print("Install matplotlib (pip install gmm-migt[plot]) to plot histograms.")
