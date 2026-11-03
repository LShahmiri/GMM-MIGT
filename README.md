<p align="center">
  <img src="https://raw.githubusercontent.com/LShahmiri/GMM-MIGT/main/docs/images/gmm-migt-logo.png" alt="GMM-MIGT logo" width="300">
</p>

<h1 align="center">GMM-MIGT</h1>
<p align="center"><b>Gaussian Mixture Model – Mutual Information Guided Training</b><br>
Representative train / test / validation partitioning of image datasets</p>

---

## Contents

1. [Introduction](#1-introduction)
2. [Scientific background](#2-scientific-background)
3. [Differences between MIGT and GMM-MIGT](#3-differences-between-migt-and-gmm-migt)
4. [Installation](#4-installation)
5. [Quick start](#5-quick-start)
6. [Supported modes](#6-supported-modes)
7. [API documentation](#7-api-documentation)
8. [Parameters](#8-parameters)
9. [Practical examples](#9-practical-examples)
10. [Reproducibility](#10-reproducibility)
11. [Citation](#11-citation)
12. [References](#12-references)

---

## 1. Introduction

**GMM-MIGT** is a model-agnostic dataset-partitioning method for image
classification. It splits each class of an image dataset into **train**,
**test** and (optionally) **validation** subsets. It guides the split with
**Mutual Information (MI)** similarity scores rather than random sampling,
so that easy, typical and unusual samples of every class are represented in
every subset.

GMM-MIGT extends **MIGT** (Mutual Information Guided Training). It adds two
preprocessing steps that remove less informative content from each image
before the MI scores are computed:

1. **EM-GMM colour reduction**: a Gaussian Mixture Model fitted with the
   Expectation-Maximisation algorithm reduces every image to `K = 4`
   homogeneous colours. This reduces the effect of lighting, pigmentation
   and reflections.
2. **PCA reconstruction**: Principal Component Analysis is applied to each
   colour channel. It keeps only the components that explain 95 % of the
   variance and removes low-variance noise and redundancy.

The method was published in L. Shahmiri, P. Wong and L. S. Dooley,
*A Gaussian Mixture Model and Mutual Information Guided Dataset Partitioning
Algorithm for Convolutional Neural Network Training*, IEEE DSP 2025
([doi:10.1109/DSP65409.2025.11074907](https://doi.org/10.1109/DSP65409.2025.11074907)).
It is described in detail in Chapter 5 of the PhD thesis *A Robust Framework
for Automatic Classification of Medicinal Plants in the Natural Environment*
(L. Shahmiri, The Open University, 2025). In the thesis, GMM-MIGT gave fewer
misclassifications than MIGT, HDBSCAN-based partitioning and Random
partitioning on the VNPlant-200, BJFU100 and WEAPD datasets with an
Xception CNN.

`gmm-migt` is an **independent package**. It does not depend on, replace or
modify the `migt` package, and both can be installed in the same
environment (`import migt` / `import gmm_migt`).

```text
 Dataset ──► EM-GMM (K=4) ──► PCA (CPV 95%) ──► MI vs. reference ──► adaptive histogram ──► train / test / val
              colour reduction   per channel       per class            (4 → 3 → 2 bins)
```

## 2. Scientific background

**Mutual Information as similarity.** For two images *A* and *B* of equal
size, MI measures how much information one image carries about the other.
It is estimated from their joint intensity histogram (thesis Eq. 4.1–4.7):

$$I(A,B) = H(A) + H(B) - H(A,B) = \sum_{a,b} P_{AB}(a,b)\log\frac{P_{AB}(a,b)}{P_A(a)P_B(b)}.$$

In each class, the first image `x1` is the **reference**. The similarity
score of every other image is `S_i = MI(x1, x_i)` (Eq. 4.8). The scores are
max-normalised to `[0, 1]` with `MI_i / max(MI)` (Eq. 4.9).

**MIGT histogram partitioning** (thesis Section 4.2.2, Algorithm 1):

* The normalised scores are placed in an equal-width histogram with
  `bins = 4`.
* If any bin holds fewer than `min_bin` samples, the number of bins is
  reduced to 3 and then to 2.
* Each accepted bin is split by the train/test/val ratios, so every subset
  receives samples from every similarity range.
* If even 2 bins fail, the scores are sorted and cut into **three equal
  portions**, and each portion is split by the ratios. A single random bin
  is never used.

**EM-GMM colour reduction** (Section 5.2.1, Eq. 5.1–5.4):

* The pixels `x(i)` (colour vectors) are modelled by a mixture of `K`
  Gaussians with weights `Φ_j`, means `μ_j` and covariances `Σ_j`.
* The **E-step** computes the responsibilities
  `ω_j(i) = g_j(x(i)) Φ_j / Σ_l g_l(x(i)) Φ_l`.
* The **M-step** updates `Φ_j = (1/m) Σ_i ω_j(i)`,
  `μ_j = Σ_i ω_j(i) x(i) / Σ_i ω_j(i)` and
  `Σ_j = Σ_i ω_j(i)(x(i) − μ_j)(x(i) − μ_j)ᵀ / Σ_i ω_j(i)`.
* The steps repeat until the log-likelihood changes by less than `0.001`,
  or for at most 100 iterations.
* Each pixel is then replaced by the mean colour of its most probable
  component.

**PCA** (Section 5.2.2, Eq. 5.5–5.8):

* PCA is applied **separately to each colour channel**, with the rows of
  the channel matrix as observations.
* After mean subtraction, the covariance matrix and its eigen-decomposition
  are computed.
* The smallest number of components whose **Cumulative Percent Variance
  (CPV)** reaches 95 % is kept.
* The data are projected onto those components and the channel is
  reconstructed.

The MI scores are computed between these **processed** images. The
partition itself copies the **original**, unmodified image files.

## 3. Differences between MIGT and GMM-MIGT

| Aspect | MIGT | GMM-MIGT |
|---|---|---|
| Input to MI | original image | EM-GMM colour-reduced + PCA-reconstructed image |
| Colour handling | raw pixel intensities | `K` mean colours (default 4) |
| Noise / redundancy | none removed | PCA keeps 95 % CPV per channel |
| Binning, fallbacks, ratios, reference image | Algorithm 1 | **identical** (thesis Section 5.2) |
| Output | copies of original images | copies of original images (+ optional processed images) |
| Python package | `migt` / `MIGTSplitter` | `gmm-migt` / `GMMMIGTSplitter` |

`GMMMIGTSplitter` accepts **every `MIGTSplitter` argument**, with the same
name and meaning. All defaults are the same except `min_bin`, which is 10 in
GMM-MIGT (the thesis value; 13 in the MIGT 1.1.1 code). With `use_gmm=False, use_pca=False` and the
same `min_bin` it reproduces the partitions of MIGT 1.1.1 exactly. This is
checked by the automated test suite for `min_bin=10` and `min_bin=13`.

**Thesis vs. MIGT 1.1.1 code.** A few values in the published MIGT code
differ from those reported in the thesis. Apart from `min_bin`, GMM-MIGT
keeps the MIGT 1.1.1 behaviour by default and offers the thesis values
through `GMMMIGTSplitter.from_thesis(...)`:

| Setting | MIGT 1.1.1 code | Thesis (`from_thesis`) |
|---|---|---|
| `min_bin` | 13 (GMM-MIGT default: **10**, as in the thesis) | 10 (Section 4.2.2; also consistent with the 10-sample bin in Figure 5.4) |
| `train / test / val` | 0.8 / 0.2 / none | 0.5 / 0.4 / 0.1 |
| bin intervals | `[a, b)`, last `[a, b]` | `[a, b]`, then `(a, b]` (figure labels) |
| empty histogram bin | dropped, remaining bins tested | counts as a bin with < `min_bin` samples |

One fix is applied in all configurations. With
`low_sample_strategy="random"`, MIGT 1.1.1 could copy the reference image
into both *train* and *test*, which leaks data. GMM-MIGT always places it in
*train* only.

**No image is lost silently.** Every `plan()` and `run()` checks that each
image file of each class is in exactly one split, or is listed in
`result.skipped()` with the reason. An image is skipped only when its MI
cannot be computed: the file cannot be decoded, or, with
`strict_shape=True`, its size differs from the reference image. As in MIGT,
skipped images are not copied, but GMM-MIGT also raises a `UserWarning`;
use `strict_shape=False, resize_to=(w, h)` to include images of other sizes.
After copying, `run()` checks that every output folder holds exactly the
planned files with the sizes of the originals. It refuses an output folder
that already contains images, so a second run cannot put one image into two
splits. `summary()` ends with a line such as
`Check: 98 image(s) found = 98 assigned + 0 skipped [OK]`. Any violation
raises `IntegrityError`.

**Small bins and empty test sets.** As in MIGT, every bin is split
separately and the rounding remainder goes to train. With 80/20, a bin of
fewer than 5 images therefore adds no test image. This can happen in the
forced-quantile fallback: a 12-image class whose histogram fails is split
into MI parts of 3, 4 and 4 images and ends with 12 train and 0 test images. Nothing is
lost, but GMM-MIGT warns when a class gets no image in a split. A larger
`min_bin` gives larger bins.

## 4. Installation

```bash
pip install gmm-migt
```

Optional extras:

```bash
pip install "gmm-migt[plot]"   # histogram plots (matplotlib)
pip install "gmm-migt[dev]"    # tests, build and twine
```

Requirements: Python ≥ 3.9, NumPy, scikit-learn, scikit-image,
opencv-python, joblib and tqdm.

To install from source:

```bash
git clone <repository-url> gmm-migt
cd gmm-migt
pip install -e ".[dev]"
pytest
```

## 5. Quick start

```text
dataset/
 ├── class1/  img1.jpg  img2.jpg ...
 └── class2/  img1.jpg  img2.jpg ...
```

```python
from gmm_migt import GMMMIGTSplitter

splitter = GMMMIGTSplitter(dataset_root="path/to/dataset")
result = splitter.run(output_root="gmm_migt_output")
print(result.summary())
```

```text
gmm_migt_output/
 ├── train/<class>/...
 ├── test/<class>/...
 ├── val/<class>/...          # only if val is set
 └── gmm_migt_report.json     # every MI score, bin and split
```

To use the configuration from the thesis (`min_bin=10`, 50/40/10 split):

```python
splitter = GMMMIGTSplitter.from_thesis("path/to/dataset")
```

From the command line:

```bash
gmm-migt path/to/dataset gmm_migt_output
gmm-migt path/to/dataset gmm_migt_output --thesis --n-components 4 --pca-variance 0.95
gmm-migt path/to/dataset --dry-run          # print the partition, copy nothing
```

## 6. Supported modes

All modes and options of MIGT 1.1.1 are available, plus the GMM-MIGT
preprocessing options.

| Category | Option | Values |
|---|---|---|
| **MI mode** | `mode` | `"auto"` (grayscale MI for grayscale images, otherwise colour MI) · `"grayscale"` (histogram MI, 64 bins) · `"color"` (channel-wise MI, mean over R, G, B) |
| Colour MI estimator | `color_mi` | `"nmi"` (scikit-image normalised MI, as in MIGT 1.1.1) · `"mutual_info_score"` (scikit-learn MI per channel, thesis Section 3.3) |
| **Image size** | `strict_shape`, `resize_to` | strict (images must match the reference image's size, others are skipped) · resize in memory to `(w, h)` |
| **Splits** | `train`, `test`, `val` | two-way (`val=None`) or three-way |
| **Small classes** (≤ `min_bin` images) | `low_sample_strategy` | `"mi_quantile"` (two MI portions) · `"random"` |
| Histogram fallback | `fallback_quantiles` | equal MI portions when 2 bins fail (default 3) |
| Histogram convention | `histogram_convention` | `"migt"` (MIGT 1.1.1 code) · `"thesis"` |
| **EM-GMM** | `use_gmm`, `n_components`, `gmm_*` | on/off, `K`, tolerance, iterations, covariance type, initialisation, pixel subsampling |
| **PCA** | `use_pca`, `pca_variance` | on/off, CPV threshold |
| Ablations | `use_gmm` / `use_pca` | GMM-MIGT · EM-GMM only · PCA only · none (= MIGT) |
| Execution | `n_jobs`, `processed_root`, `plan()`, `run()` | parallel processes · save processed images · dry run · copy |
| Baseline | `RandomSplitter` | random per-class partition with the same outputs |

The decision logic applied to each class is identical to MIGT:

| Situation | Strategy (`ClassReport.strategy`) |
|---|---|
| Class has no images | `empty` (folders are still created) |
| Class size ≤ `min_bin` | `low_sample_mi_quantile` or `low_sample_random` (user choice); `low_sample_random_fallback` if fewer than 2 MI values |
| Histogram with `bins` → … → 2 bins succeeds | `histogram` |
| Histogram fails at 2 bins, or fewer than 3 MI values | `forced_quantile` (never random) |

## 7. API documentation

### `GMMMIGTSplitter(dataset_root, **parameters)`

The main class. All parameters are listed in [Section 8](#8-parameters).

| Method | Description |
|---|---|
| `plan() -> SplitResult` | Compute the partition without copying files. |
| `run(output_root, write_report=True) -> SplitResult` | Compute the partition and copy the original images into `output_root/<split>/<class>/`. Optionally write `gmm_migt_report.json`. |
| `from_thesis(dataset_root, **overrides)` (classmethod) | Splitter with the thesis values (`min_bin=10`, 0.5/0.4/0.1, `histogram_convention="thesis"`). |
| `get_config() -> dict` | All parameters (stored in every report). |

### `SplitResult`

| Member | Description |
|---|---|
| `classes: dict[str, ClassReport]` | Per-class records. |
| `splits()` | `{split: {class: [paths]}}`. |
| `counts()`, `totals()` | Image counts per class / split. |
| `skipped()` | Images whose MI could not be computed, with the reason. |
| `accounting()` | `{"found", "assigned", "skipped"}` image counts (found = assigned + skipped). |
| `verify(dataset_root=None)` | Raise `IntegrityError` if an image is missing, duplicated or in an unknown split; with `dataset_root`, also compare with the files on disk. |
| `summary()` | Text table, ending with the image-count check. |
| `to_json(path)`, `from_json(path)`, `to_dict()` | Serialisation. |
| `copy_to(output_root)` | Materialise a (re-loaded) partition and check the copied files. |

`ClassReport` has the fields `name`, `n_images`, `strategy`, `reference`,
`mi_max`, `histogram_attempts` (every tried bin count with its bin sizes),
`bins` (`BinSummary`: size, MI range, normalised interval, label) and
`items`. Each item is an `ItemRecord` with `path`, `split`, `role`, `mi`,
`normalized_mi`, `bin`, `gmm_converged`, `gmm_n_iter` and
`pca_components`. It also has `skipped`, `counts` and `paths(split)`.

### Preprocessing functions

| Function | Description |
|---|---|
| `preprocess_image(img, config=None) -> (processed, info)` | EM-GMM, then PCA. Returns an image with the input dtype. |
| `em_gmm_colour_reduction(img, n_components=4, tol=1e-3, max_iter=100, covariance_type="full", init_params="kmeans", reg_covar=1e-6, max_pixels=None, random_state=None)` | Step 1. Returns `(float image of cluster means, GMMInfo)`. |
| `pca_reconstruction(img, variance=0.95)` | Step 2, per channel. Returns `(image, PCAInfo)`. |
| `pca_channel_reconstruction(channel, variance=0.95)` | PCA on one 2-D matrix (rows = observations). Returns `(reconstruction, k, cpv)`. |
| `select_n_components(explained_variance_ratio, variance)` | CPV rule: smallest `k` with cumulative ratio ≥ `variance`. |
| `PreprocessingConfig(...)` | Frozen dataclass with all preprocessing parameters (validated). |

### MI and binning functions

| Function | Description |
|---|---|
| `load_image(path, target_size=None)` | OpenCV loader (BGR). Converts fake grayscale to grayscale, drops alpha, supports unicode paths. |
| `compute_mi(img1, img2, mode="auto", expected_size=None, color_mi="nmi")` | MI between two images. |
| `compute_gray_mi(img1, img2, bins=64)`, `compute_color_mi(img1, img2, estimator="nmi")` | The individual estimators. |
| `histogram_bins(mi_list, num_bins, convention)` | Equal-width bins of `(path, mi)` items. |
| `adaptive_histogram(mi_list, start_bins, min_bin, convention, attempts=None)` | The 4 → 3 → 2 bin procedure. Returns `None` if every bin count fails. |
| `forced_quantile_split(mi_list, n_parts=3)` | Equal portions of the sorted MI values. |
| `normalise_mi(values)` | Eq. 4.9. |
| `interval_labels(edges, convention)` | Labels such as `"[0.04, 0.28]"`, `"(0.28, 0.52]"`. |
| `split_counts(n, ratios)` | Per-split counts that sum to `n`. The remainder goes to train. |

Other members: `RandomSplitter(dataset_root, train, test, val, seed)` with
`plan()` and `run()`; `gmm_migt.plotting.plot_class_histogram(class_report)`;
and the exceptions `GMMMIGTError`, `ConfigurationError`, `ImageLoadError`,
`ImageShapeError` and `IntegrityError`.

## 8. Parameters

**MIGT parameters** (same names as `migt.MIGTSplitter`; same defaults except `min_bin`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dataset_root` | str | – | Dataset directory with one sub-folder per class. |
| `mode` | str | `"auto"` | MI mode: `auto` / `grayscale` / `color`. |
| `bins` | int | `4` | Initial number of histogram bins (≥ 2), reduced down to 2. |
| `min_bin` | int | `10` | Minimum number of samples per bin (any integer ≥ 1). The histogram starts with `bins` bins (4) and reduces them 4 → 3 → 2 until every bin holds at least `min_bin` images. Classes with ≤ `min_bin` images use `low_sample_strategy`. Thesis: 10. MIGT 1.1.1 code: 13. |
| `train` | float | `0.8` | Training ratio. Thesis: 0.5. |
| `test` | float | `0.2` | Test ratio. Thesis: 0.4. |
| `val` | float or None | `None` | Validation ratio (`None` = no validation set). Thesis: 0.1. |
| `strict_shape` | bool | `True` | Require identical image sizes. |
| `resize_to` | (w, h) or None | `None` | Resize in memory (required if `strict_shape=False`). |
| `seed` | int | `42` | Seed for shuffling and the EM-GMM initialisation. |
| `verbose` | bool | `True` | Print progress. |
| `low_sample_strategy` | str | `"mi_quantile"` | `mi_quantile` or `random` for small classes. |

**GMM-MIGT preprocessing** (defaults are the thesis values, Algorithm 5.1):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_gmm` | bool | `True` | Apply EM-GMM colour reduction. |
| `n_components` | int | `4` | Number of GMM clusters `K`. The thesis tested K = 2…6 and chose 4. |
| `gmm_tol` | float | `0.001` | EM convergence threshold (change of the mean log-likelihood). |
| `gmm_max_iter` | int | `100` | Maximum number of EM iterations. |
| `gmm_covariance_type` | str | `"full"` | `full` (Eq. 5.4), `tied`, `diag` or `spherical`. |
| `gmm_init` | str | `"kmeans"` | EM initialisation: `kmeans` (scikit-learn default, seeded), `k-means++`, `random` or `random_from_data`. |
| `gmm_reg_covar` | float | `1e-6` | Regularisation added to the covariance diagonals. |
| `gmm_max_pixels` | int or None | `None` | Fit the GMM on a seeded random subset of at most this many pixels, then apply it to all pixels. Useful for very large images. `None` = all pixels (thesis). |
| `use_pca` | bool | `True` | Apply per-channel PCA reconstruction. |
| `pca_variance` | float | `0.95` | CPV threshold in (0, 1]. |

**Additional options:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `color_mi` | str | `"nmi"` | Colour MI estimator (`nmi` or `mutual_info_score`). |
| `histogram_convention` | str | `"migt"` | `migt` or `thesis` (see Section 3). |
| `fallback_quantiles` | int | `3` | Number of equal portions used when the histogram fails. |
| `n_jobs` | int | `1` | Worker processes for preprocessing + MI (`-1` = all cores). |
| `processed_root` | str or None | `None` | Save the processed images (PNG) to `processed_root/<class>/`. |

## 9. Practical examples

The `examples/` folder of the source distribution contains runnable scripts
for every mode. Each script accepts a dataset path, or generates a demo
dataset if none is given:

| Script | Shows |
|---|---|
| `01_quickstart.py` | default run |
| `02_mi_modes.py` | `auto` / `grayscale` / `color`, colour estimators, grayscale data |
| `03_thesis_configuration.py` | `from_thesis`, histogram attempts and bin labels |
| `04_custom_preprocessing.py` | `n_components`, `pca_variance`, GMM options, ablations |
| `05_shapes_and_small_classes.py` | strict vs. resize, `low_sample_strategy` |
| `06_single_image_preprocessing.py` | the preprocessing and MI functions on single images |
| `07_report_and_histograms.py` | JSON report, reloading, histogram plots |
| `08_compare_with_random_and_migt.py` | GMM-MIGT vs. MIGT vs. Random |

**Custom preprocessing.** Every parameter has a default and can be changed:

```python
splitter = GMMMIGTSplitter(
    "dataset",
    n_components=6,          # 6 GMM clusters instead of 4
    pca_variance=0.90,       # keep 90 % cumulative variance
    gmm_max_iter=200,
    train=0.6, test=0.3, val=0.1,
    n_jobs=-1,               # use all CPU cores
)
result = splitter.run("output")
```

**Mixed image sizes.**

```python
GMMMIGTSplitter("dataset", strict_shape=False, resize_to=(224, 224)).run("output")
```

**Ablation study.**

```python
for gmm, pca in [(True, True), (True, False), (False, True), (False, False)]:
    r = GMMMIGTSplitter("dataset", use_gmm=gmm, use_pca=pca, verbose=False).plan()
    print(gmm, pca, r.totals())
```

**Preprocess a single image.**

```python
from gmm_migt import load_image, preprocess_image, PreprocessingConfig, save_image

img = load_image("leaf.jpg")
processed, info = preprocess_image(img, PreprocessingConfig(n_components=4, pca_variance=0.95))
print(info.gmm.converged, info.pca.n_components)
save_image("leaf_processed.png", processed)
```

**Thesis evaluation protocol.** In the thesis, the test set produced by each
partitioning method was replaced by one fixed test set from a Random
partition, so that every method was evaluated on the same images
(Sections 4.3 and 5.3). `RandomSplitter` produces such a partition. Make
sure that the fixed test images do not also appear in the training or
validation subsets you use.

## 10. Reproducibility

* **Deterministic.** With the same `seed`, dataset, parameters and library
  versions, `plan()` and `run()` produce identical partitions. The result
  does not depend on `n_jobs`. The EM-GMM initialisation uses `seed` as
  `random_state`, and the shuffling inside each bin uses a private
  `random.Random(seed)`, so global random state is neither used nor
  changed.
* **Ordering.** Classes and images are processed in sorted order. The
  first image (sorted by file name) of each class is the reference image
  and is always placed in *train*.
* **Audit trail.** `gmm_migt_report.json` stores the package version, all
  parameters, every histogram attempt, the bins, and every image's MI,
  normalised MI, bin and split. `SplitResult.from_json(...).copy_to(...)`
  recreates a partition without recomputing it.
* **Library versions.** EM-GMM and PCA use scikit-learn, and the MI
  estimators use scikit-learn / scikit-image. Record the versions
  (`pip freeze`) together with the report. Different BLAS/LAPACK builds can
  change floating-point results in the last digits.
* **Runtime.** EM-GMM with full covariances on every pixel takes a few
  seconds for a 512 × 512 image. Use `n_jobs=-1` to parallelise. For
  multi-megapixel images use `resize_to` or `gmm_max_pixels`, and report
  this choice, because it is not part of the thesis configuration.

## 11. Citation

If you use GMM-MIGT, please cite:

> L. Shahmiri, P. Wong and L. S. Dooley, "A Gaussian Mixture Model and Mutual
> Information Guided Dataset Partitioning Algorithm for Convolutional Neural
> Network Training," *2025 25th International Conference on Digital Signal
> Processing (DSP)*, Pylos (Messinia, Southwest Peloponnese), Greece, 2025,
> pp. 1-5, doi: 10.1109/DSP65409.2025.11074907.

```bibtex
@inproceedings{shahmiri2025gmmmigt,
  author    = {Shahmiri, Lida and Wong, Patrick and Dooley, Laurence S.},
  title     = {A Gaussian Mixture Model and Mutual Information Guided Dataset Partitioning Algorithm for Convolutional Neural Network Training},
  booktitle = {2025 25th International Conference on Digital Signal Processing (DSP)},
  address   = {Pylos (Messinia, Southwest Peloponnese), Greece},
  year      = {2025},
  pages     = {1--5},
  doi       = {10.1109/DSP65409.2025.11074907},
  url       = {https://ieeexplore.ieee.org/abstract/document/11074907}
}
```

The full methodology is described in the PhD thesis:

```bibtex
@phdthesis{shahmiri2025robust,
  author = {Shahmiri, Lida},
  title  = {A Robust Framework for Automatic Classification of Medicinal Plants in the Natural Environment},
  school = {The Open University},
  year   = {2025},
  doi    = {10.21954/ou.ro.00106127},
  url    = {https://oro.open.ac.uk/106127/}
}
```

The MIGT method on which GMM-MIGT is built:

```bibtex
@inproceedings{shahmiri2022migt,
  author    = {Shahmiri, Lida and Wong, Patrick and Dooley, Laurence S.},
  title     = {Accurate Medicinal Plant Identification in Natural Environments by Embedding Mutual Information in a Convolution Neural Network Model},
  booktitle = {IEEE 5th International Conference on Image Processing Applications and Systems (IPAS)},
  year      = {2022},
  url       = {https://ieeexplore.ieee.org/abstract/document/10053008}
}
```

## 12. References

* L. Shahmiri, P. Wong, L. S. Dooley, *A Gaussian Mixture Model and Mutual
  Information Guided Dataset Partitioning Algorithm for Convolutional Neural
  Network Training*, 2025 25th International Conference on Digital Signal
  Processing (DSP), Pylos, Greece, pp. 1–5.
  https://doi.org/10.1109/DSP65409.2025.11074907
* L. Shahmiri, *A Robust Framework for Automatic Classification of Medicinal
  Plants in the Natural Environment*, PhD thesis, The Open University, 2025.
  Chapter 5: *Gaussian Mixture Model – Mutual Information Guided Training
  (GMM-MIGT)*. https://doi.org/10.21954/ou.ro.00106127
* L. Shahmiri, P. Wong, L. S. Dooley, *Accurate Medicinal Plant
  Identification in Natural Environments by Embedding Mutual Information in
  a Convolution Neural Network Model*, IEEE IPAS 2022.
  https://ieeexplore.ieee.org/abstract/document/10053008
* MIGT Python package: https://pypi.org/project/migt/

## License

MIT License. Copyright (c) 2026 Lida Shahmiri.
