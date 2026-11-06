# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-11-06

### Added
- `method="migt"` runs MIGT (MI on the original images, no preprocessing)
  with the same splitter; `method="gmm_migt"` (default) runs GMM-MIGT.
  Also available as `--method` on the command line.
- `rounding` option. `"class"` (new default) rounds the train/test/val
  sizes once per class (largest remainder, at least one image per split
  when the class has at least as many images as splits) and spreads them
  over the bins in proportion to their sizes. `"per_bin"` keeps the
  previous behaviour.
- `unscored` option. `"include"` (new default) puts images whose MI cannot
  be computed into an extra stratum of their class that is split at random,
  so no image is lost; `result.unscored()` lists them with the reason.
  `"skip"` keeps the previous behaviour. Both raise a `UserWarning`.
- `class_counts()` and `stratified_counts()`; `ItemRecord.reason`;
  `RandomSplitter(rounding=...)`.

### Changed
- `low_sample_strategy` now defaults to `"random"`: classes with
  `<= min_bin` images are split at random, as in the papers. The reference
  image has no special role in a random split.
- With `low_sample_strategy="mi_quantile"` the two MI halves are used as
  strata, so test/validation images are drawn from both halves (before,
  `[reference] + low half + high half` was cut in this order and test/val
  came only from the high-MI half).
- `from_thesis()` sets `low_sample_strategy="random"` and
  `rounding="per_bin"`.
- `use_gmm` and `use_pca` default to `None` (on for `gmm_migt`, off for
  `migt`); `method="migt"` together with `use_gmm=True` or `use_pca=True`
  is rejected.
- SciPy is listed as a dependency (it was already required by scikit-learn).

### Fixed
- PCA no longer fails with "SVD did not converge": if LAPACK's default SVD
  driver does not converge, the QR-based driver is used.
- In `mode="auto"` a grayscale photo in a colour class (or a colour photo in
  a grayscale class) is now scored on the scale of the reference image
  instead of failing.
- A class could end with no test or validation image because every bin was
  rounded separately (see `rounding`).

The partitions of 1.0.1 are reproduced with `rounding="per_bin"`,
`low_sample_strategy="mi_quantile"` and `unscored="skip"`.

## [1.0.1] - 2026-11-03

### Added
- GMM-MIGT logo at the top of the README (also shown on PyPI).
- Homepage, Repository and Issues links to
  https://github.com/LShahmiri/GMM-MIGT.

### Changed
- README: the "install from source" instructions use the GitHub repository.

## [1.0.0] - 2026-11-03

First release of **GMM-MIGT**, an independent package (`pip install gmm-migt`,
`import gmm_migt`) implementing the method of Shahmiri, Wong and Dooley, IEEE DSP
2025 (doi:10.1109/DSP65409.2025.11074907), as detailed in Chapter 5 of
L. Shahmiri's PhD thesis (The Open University, 2025). It does not depend on or
modify the `migt` package.

### Added
- `GMMMIGTSplitter`: the MIGT 1.1.1 partitioning logic (adaptive MI histogram
  4 → 3 → 2, forced MI-quantile fallback, low-sample strategies, reference image
  in train) with GMM-MIGT preprocessing before every MI computation. It accepts
  all `MIGTSplitter` arguments with the same defaults, except `min_bin`,
  which defaults to 10, the thesis value (MIGT 1.1.1 code: 13; any value
  can be passed). The histogram starts with `bins=4` bins and reduces them
  4 -> 3 -> 2 until every bin holds at least `min_bin` images.
- Image-count checks: `plan()`, `run()` and `copy_to()` verify that every
  image is in exactly one split or listed as skipped, and that the copied
  folders hold exactly the planned files. `run()` refuses an output folder
  that already contains images. Skipped images raise a `UserWarning`, as
  does a class that receives no image in some split.
  `SplitResult.verify()`, `SplitResult.accounting()` and `IntegrityError`
  were added.
- EM-GMM colour reduction (`em_gmm_colour_reduction`). Defaults: K = 4,
  tolerance 0.001, 100 iterations, full covariances. All are configurable.
- Per-channel PCA reconstruction with Cumulative Percent Variance selection
  (`pca_reconstruction`). Default 95 %, configurable.
- `GMMMIGTSplitter.from_thesis()` preset (min_bin = 10, 50/40/10 split, thesis
  histogram convention).
- Options `histogram_convention`, `color_mi`, `fallback_quantiles`,
  `use_gmm` / `use_pca` (ablations), `gmm_max_pixels`, `n_jobs` and
  `processed_root`.
- `plan()` (dry run) and the `SplitResult` report: JSON round-trip,
  `summary()`, `splits()`, `copy_to()`, MI and normalised MI for every image,
  histogram attempts, skipped images with reasons.
- `RandomSplitter` baseline, `gmm-migt` command-line tool and
  `python -m gmm_migt`.
- Optional histogram plotting (`gmm-migt[plot]`).
- Unit, integration and MIGT-equivalence tests. Validation script and report.

### Fixed (relative to the MIGT 1.1.1 behaviour)
- With `low_sample_strategy="random"`, the reference image could be copied
  into both train and test. It is now placed in train only.
- `split_counts` is robust to binary floating-point error
  (e.g. `0.29 * 100 = 28.999…`).
- Invalid `mode` values now raise an error instead of silently behaving like
  `"auto"`. `mode="color"` on a grayscale reference raises a clear error.
- Images with an alpha channel and paths with non-ASCII characters are
  supported.
