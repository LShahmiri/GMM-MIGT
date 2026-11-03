"""Quick start: partition a dataset with GMM-MIGT using the default parameters.

    python examples/01_quickstart.py [path/to/dataset]
"""

from _demo_data import dataset_from_argv, output_dir

from gmm_migt import GMMMIGTSplitter

dataset = dataset_from_argv()

splitter = GMMMIGTSplitter(dataset_root=dataset)   # same defaults as MIGT + EM-GMM (K=4) + PCA (95%)
result = splitter.run(output_root=output_dir("gmm_migt_output"))

print(result.summary())
