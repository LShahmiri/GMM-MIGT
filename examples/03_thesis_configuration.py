"""Reproduce the configuration reported in the PhD thesis (Chapters 4-5).

min_bin=10, train/test/validation = 50/40/10, thesis histogram convention,
K=4 GMM clusters (tol 0.001, 100 iterations) and 95% PCA CPV.

    python examples/03_thesis_configuration.py [path/to/dataset]
"""

from _demo_data import dataset_from_argv, output_dir

from gmm_migt import GMMMIGTSplitter

dataset = dataset_from_argv()

splitter = GMMMIGTSplitter.from_thesis(dataset, seed=42)
# equivalent to:
# GMMMIGTSplitter(dataset, bins=4, min_bin=10, train=0.5, test=0.4, val=0.1,
#                 histogram_convention="thesis", n_components=4, gmm_tol=1e-3,
#                 gmm_max_iter=100, pca_variance=0.95)
result = splitter.run(output_dir("gmm_migt_thesis"))
print(result.summary())

for name, r in result.classes.items():
    print(f"\n{name}: strategy={r.strategy}")
    for a in r.histogram_attempts:
        print(f"  tried {a['n_bins']} bins -> sizes {a['sizes']} {'(accepted)' if a['accepted'] else ''}")
    for b in r.bins:
        print(f"  bin {b.index}: {b.label or ''} n={b.size}")
