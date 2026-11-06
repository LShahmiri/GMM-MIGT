"""GMM-MIGT: Gaussian Mixture Model - Mutual Information Guided Training.

GMM-MIGT partitions an image-classification dataset into train / test /
(optional) validation sets. It is the MIGT method (Mutual Information
Guided Training) with two preprocessing steps applied to every image before
the MI similarity scores are computed: EM-GMM colour reduction and
per-channel PCA reconstruction.

Reference: L. Shahmiri, P. Wong and L. S. Dooley, "A Gaussian Mixture Model
and Mutual Information Guided Dataset Partitioning Algorithm for
Convolutional Neural Network Training," 2025 25th International Conference
on Digital Signal Processing (DSP), pp. 1-5,
doi: 10.1109/DSP65409.2025.11074907; and L. Shahmiri, PhD thesis, The Open
University, 2025, Chapter 5.

Quick start::

    from gmm_migt import GMMMIGTSplitter

    splitter = GMMMIGTSplitter(dataset_root="path/to/dataset")   # method="migt" for MIGT
    result = splitter.run(output_root="gmm_migt_output")
    print(result.summary())
"""

from ._version import __version__
from .baselines import RandomSplitter
from .binning import (
    BinInfo,
    adaptive_histogram,
    forced_quantile_split,
    histogram_bins,
    interval_labels,
    normalise_mi,
)
from .exceptions import ConfigurationError, GMMMIGTError, ImageLoadError, ImageShapeError, IntegrityError
from .mi import compute_color_mi, compute_gray_mi, compute_mi, is_grayscale, load_image, save_image
from .preprocessing import (
    PreprocessingConfig,
    em_gmm_colour_reduction,
    pca_channel_reconstruction,
    pca_reconstruction,
    preprocess_image,
    select_n_components,
)
from .report import ClassReport, ItemRecord, SplitResult
from .splitter import GMMMIGTSplitter
from .utils import IMAGE_EXTENSIONS, class_counts, list_images, split_counts, stratified_counts

__all__ = [
    "__version__",
    # main API
    "GMMMIGTSplitter",
    "RandomSplitter",
    "SplitResult",
    "ClassReport",
    "ItemRecord",
    # preprocessing
    "PreprocessingConfig",
    "preprocess_image",
    "em_gmm_colour_reduction",
    "pca_reconstruction",
    "pca_channel_reconstruction",
    "select_n_components",
    # MI
    "compute_mi",
    "compute_gray_mi",
    "compute_color_mi",
    "load_image",
    "save_image",
    "is_grayscale",
    # binning
    "BinInfo",
    "histogram_bins",
    "adaptive_histogram",
    "forced_quantile_split",
    "normalise_mi",
    "interval_labels",
    # utils
    "split_counts",
    "class_counts",
    "stratified_counts",
    "list_images",
    "IMAGE_EXTENSIONS",
    # errors
    "GMMMIGTError",
    "ConfigurationError",
    "ImageLoadError",
    "ImageShapeError",
    "IntegrityError",
]
