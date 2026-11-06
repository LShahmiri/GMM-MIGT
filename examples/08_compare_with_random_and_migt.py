"""Partition the same dataset with GMM-MIGT, MIGT (no preprocessing) and Random.

The thesis compares these partitioning methods (Chapter 5). This script
produces the three partitions side by side, each in its own folder.

    python examples/08_compare_with_random_and_migt.py [path/to/dataset]
"""

from _demo_data import dataset_from_argv, output_dir

from gmm_migt import GMMMIGTSplitter, RandomSplitter

dataset = dataset_from_argv()
ratios = dict(train=0.5, test=0.4, val=0.1, seed=42)

gmm_migt = GMMMIGTSplitter.from_thesis(dataset, verbose=False, **ratios).run(output_dir("gmm_migt"))
migt_like = GMMMIGTSplitter.from_thesis(dataset, method="migt", verbose=False,
                                        **ratios).run(output_dir("migt"))
random_split = RandomSplitter(dataset, verbose=False, **ratios).run(output_dir("random"))

for name, res in (("GMM-MIGT", gmm_migt), ("MIGT", migt_like), ("Random", random_split)):
    print(f"\n=== {name} ===")
    print(res.summary())
