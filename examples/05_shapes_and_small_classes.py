"""Image size handling (strict vs resize) and small-class strategies.

    python examples/05_shapes_and_small_classes.py [path/to/dataset]
"""

from _demo_data import dataset_from_argv, output_dir

from gmm_migt import GMMMIGTSplitter

dataset = dataset_from_argv()

# 1) Strict shape mode (research-safe): images whose size differs from the
#    reference image are skipped and reported.
res = GMMMIGTSplitter(dataset, strict_shape=True, verbose=False).plan()
print("skipped (strict):", res.skipped())

# 2) Practical mode: resize in memory to (width, height) for preprocessing + MI;
#    the original files are copied unchanged.
res = GMMMIGTSplitter(dataset, strict_shape=False, resize_to=(48, 48), verbose=False).run(output_dir("resized"))
print(res.summary())

# 3) Classes with <= min_bin images: choose 'mi_quantile' (default) or 'random'
for strategy in ("mi_quantile", "random"):
    res = GMMMIGTSplitter(dataset, low_sample_strategy=strategy, verbose=False).plan()
    small = [r for r in res.classes.values() if r.strategy.startswith("low_sample")]
    for r in small:
        print(f"{strategy:<12} -> class {r.name}: {r.strategy}, counts {r.counts}")
