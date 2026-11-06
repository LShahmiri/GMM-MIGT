# GMM-MIGT – design analysis

This document records the analysis behind the package: Stages 1–5 of the
development plan. Section numbers such as §4.2.2 refer to the PhD thesis
(L. Shahmiri, The Open University, 2025), which details the method published
in Shahmiri, Wong and Dooley, IEEE DSP 2025
(doi:10.1109/DSP65409.2025.11074907).

---

## Stage 1 – The original MIGT 1.1.1 package

Source: `migt-1.1.1/` (`migt/__init__.py`, `splitter.py`, `mi.py`, `utils.py`).

| Component | Behaviour |
|---|---|
| `MIGTSplitter.__init__` | Arguments: `dataset_root`, `mode="auto"`, `bins=4`, `min_bin=13`, `train=0.8`, `test=0.2`, `val=None`, `strict_shape=True`, `resize_to=None`, `seed=42`, `verbose=True`, `low_sample_strategy="mi_quantile"`. Validates the ratios (> 0, sum to 1), the strategy, and `resize_to` when `strict_shape=False`. Seeds the global `random` and `np.random`. |
| `run(output_root)` | For each class directory (`os.listdir` order), creates the split folders, lists the images (sorted, 7 extensions) and chooses Case A, B or C. Returns nothing. |
| Case A, `len(images) <= min_bin` | `_low_sample_split`. `random`: shuffles all images, splits them, and copies the reference to train again (it may already be in test). `mi_quantile`: computes MI of images[1:] against images[0], sorts them, splits them into two halves, shuffles each half, and splits the sequence `[ref]+low+high`. Falls back to random if fewer than 2 MI values are available. |
| Case B, histogram | MI of images[1:] against images[0]. Images whose MI fails (load error, shape mismatch, NaN) are skipped and **not copied**. Fewer than 3 values → forced quantile-3. Otherwise `_adaptive_histogram` tries bins = `bins`…2 with equal-width bins over [min, max], `[a,b)` intervals and a closed last bin, **drops empty bins**, skips any attempt with ≤ 1 non-empty bin, and accepts the first attempt in which every bin has ≥ `min_bin` items. |
| Case C, histogram fails | `_force_quantile_3`: sorted thirds `[:n//3]`, `[n//3:2n//3]`, `[2n//3:]`. |
| Split of a bin | Shuffle, then `split_counts`: `floor(r·n)` per split, with the remainder added to train. Train = first `n_train` items, test = next `n_test`, the rest → val (or test if `val=None`). The reference image is copied to train. |
| `mi.load_image` | `cv2.imread(IMREAD_UNCHANGED)`. Converts 3 identical channels to grayscale. Optional `cv2.resize(target_size)`. |
| `mi.compute_mi` | Strict shape check (or `expected_size`). `grayscale` → `compute_gray_mi`: 64×64 joint histogram + `sklearn.metrics.mutual_info_score`, 0 for constant images. `color` → `compute_color_mi`: `skimage.metrics.normalized_mutual_information` per RGB channel, averaged, 0 for constant channels. `auto`: grayscale MI if both images are grayscale, otherwise colour MI. Any other `mode` string is treated as `auto`. |

**Modes and capabilities identified:** 3 MI modes; adaptive histogram
binning; forced quantile-3 fallback; 2 low-sample strategies; two-way or
three-way split; strict-shape and resize modes; seed; verbose output;
reference image kept in train; no class skipping (folders are created even
for empty classes).

## Stage 2 – Thesis Chapter 5 (GMM-MIGT)

* **Purpose (§5.1).** MI computed on raw pixels is distorted by lighting,
  pigmentation, dust and blur in natural-environment images. GMM-MIGT
  removes less informative content *before* computing MI.
* **Step 1, EM-GMM (§5.2.1, Eq. 5.1–5.4).**
  * Fit a GMM to the pixel colour vectors with EM: E-step responsibilities
    (5.1); M-step weights (5.2), means (5.3) and covariances (5.4).
  * Convergence: the change in log-likelihood falls below a threshold, or
    the maximum number of iterations is reached.
  * Each pixel is assigned to the cluster with the maximum responsibility
    and replaced by that cluster's mean.
  * `K = 4` was chosen empirically from K = 2…6.
  * Implementation: scikit-learn, tolerance 0.001, max 100 iterations.
* **Step 2, PCA (§5.2.2, Eq. 5.5–5.8).**
  * Six steps: data, mean subtraction, covariance matrix,
    eigen-decomposition, choice of components, projection.
  * Applied **separately to the R, G and B channels, with the rows of each
    channel matrix as feature vectors**.
  * The number of components follows the CPV rule with a 95 % threshold.
  * The image is reconstructed after the projection (Algorithm 5.1,
    line 5).
* **Algorithm 5.1.** Lines 1–3 are EM-GMM and lines 4–5 are PCA plus
  reconstruction. Output: processed images.
* **Partitioning (§5.2, end).** "The binning strategy and sample selection
  process remain consistent with MIGT, as discussed in Section 4.2, while
  leveraging the new similarity scores derived from the pre-processed
  images."
* **Experimental setup (§5.3).** Identical to MIGT (§4.3). An Xception
  network pre-trained on ImageNet is used. The GMM-MIGT test set is
  discarded, and every method is evaluated on one fixed test set from a
  Random partition.
* **MIGT details used by GMM-MIGT (§4.2):**
  * The reference image is the first image of the class (Eq. 4.8).
  * Max-normalisation (Eq. 4.9).
  * Adaptive 4 → 3 → 2 bins with MinBin = 10.
  * Forced split into 3 equal portions.
  * Ratios 50/40/10, so a bin of 10 gives 5/4/1.
  * Thresholds "should be treated as configurable parameters".
  * §3.3: MI computed with scikit-learn `mutual_info_score`.
* **The CNN** uses the original images (Chapter 6 shows LIME explanations
  on original images). The processed images are used only to compute the
  similarity scores.

## Stage 3 – MIGT vs. GMM-MIGT

| Step | MIGT | GMM-MIGT | Source |
|---|---|---|---|
| Load image | loader | same loader | code |
| Preprocessing | – | EM-GMM (K=4, tol 1e-3, 100 iterations, full covariances) → per-channel PCA (CPV 95 %) → reconstruct | §5.2, Alg. 5.1 |
| Reference | first image | first image (also preprocessed) | §4.2.1 |
| Similarity | MI(original ref, original image) | MI(processed ref, processed image) | §5.2 |
| Normalisation, binning, fallbacks, ratios | Algorithm 1 | unchanged | §5.2 |
| Files copied | originals | originals | §5.3, Ch. 6 |

The **only algorithmic difference** is the similarity input. GMM-MIGT
therefore keeps MIGT's code paths and adds `preprocessing.py`.

## Stage 4 – Feature compatibility matrix

| MIGT 1.1.1 feature | GMM-MIGT | Notes |
|---|---|---|
| `dataset_root` | ✅ same | |
| `mode="auto"` | ✅ same | Applied to the processed images. Preprocessing keeps the channel count, so `auto` makes the same choice as MIGT. |
| `mode="grayscale"` | ✅ same | Colour images are preprocessed in colour (EM-GMM on RGB vectors, as in the thesis) and converted to grayscale for MI. |
| `mode="color"` | ✅ same | Raises a clear error if the reference image is grayscale. MIGT would silently skip every image. |
| Grayscale datasets | ✅ extended | The thesis describes RGB pixels. For single-channel images EM-GMM is fitted on 1-D intensities and PCA on the single channel. This is the direct 1-channel specialisation of Eq. 5.1–5.8. |
| `bins` | ✅ same (≥ 2) | `bins=1` silently meant "always quantile-3" in MIGT. It is now rejected with an explanatory error. |
| `min_bin` | ✅ same meaning; default **10** (thesis value, author's choice; MIGT 1.1.1 code: 13) | Any value can be passed. |
| `train`, `test`, `val` | ✅ same (default 0.8/0.2/None) | Thesis 0.5/0.4/0.1 via `from_thesis`. A clearer error explains `val=None`. |
| `strict_shape`, `resize_to` | ✅ same | Resizing happens before preprocessing. A warning is shown if `resize_to` is ignored. |
| `seed` | ✅ same | Also seeds EM-GMM. A private RNG is used, so global state is untouched. The RNG sequence is identical to MIGT's (tested). |
| `verbose` | ✅ same | ASCII output (MIGT's emoji can fail on Windows consoles with cp1252). |
| `low_sample_strategy="mi_quantile"` | ✅ same; stratified since 1.1.0 | The two halves are used as strata; the MIGT 1.1.1 ordering is kept with `rounding="per_bin"` (see D7). |
| `low_sample_strategy="random"` | ✅ fixed; default since 1.1.0 | The reference image is no longer duplicated into test/val. Random is the default, as in the papers. |
| Adaptive histogram 4 → 3 → 2 | ✅ same | `histogram_convention="migt"` reproduces MIGT bin for bin (tested). |
| Forced quantile-3 | ✅ same | Configurable through `fallback_quantiles` (the thesis studied 3, 4 and 5 and chose 3). |
| Reference image always in train | ✅ same | |
| No class skipping | ✅ same | |
| Skipping of failed images | ✅ fixed | Since 1.1.0 such images are split at random as an extra stratum and reported (`SplitResult.unscored()`); `unscored="skip"` keeps the MIGT behaviour, with a warning. |
| Per-bin rounding | ✅ fixed | Since 1.1.0 split sizes are rounded per class (`rounding="class"`), so no split is left empty by rounding; `rounding="per_bin"` reproduces MIGT. |
| `run(output_root)` | ✅ same, returns a `SplitResult` | |
| — | ➕ `plan()`, JSON report, `from_thesis`, `RandomSplitter`, CLI, `n_jobs`, `processed_root`, ablation switches, plotting | New. |

No MIGT mode had to be dropped. Every mode applies to GMM-MIGT because the
preprocessing only changes the similarity input.

### Discrepancies between the thesis and the MIGT 1.1.1 code

| # | Topic | Thesis | MIGT 1.1.1 code | Resolution in GMM-MIGT |
|---|---|---|---|---|
| D1 | MinBin | 10 (§4.2.2, Alg. 1). Fig. 5.4 accepts a 10-sample bin. | 13 | Default 10 (thesis, author's choice), user-configurable; 13 reproduces the MIGT 1.1.1 code. |
| D2 | Ratios | 50/40/10 | 80/20/– | Default MIGT; `from_thesis` → thesis. |
| D3 | Interval closure | Figures labelled `[a,b]`, `(a,b]` | `[a,b)`, last `[a,b]` | `histogram_convention` (`migt` default, `thesis`). Only values exactly on a bin edge are affected. |
| D4 | Empty bins | "any BL < MinBin" ⇒ an empty bin fails | Empty bins dropped before the test | `histogram_convention`. |
| D5 | Colour MI estimator | §3.3: scikit-learn `mutual_info_score` | scikit-image NMI per channel | `color_mi` (`nmi` default, `mutual_info_score`). |
| D6 | Low-sample classes | Not covered (all classes ≥ 100 images) | `random` / `mi_quantile` | Kept. `random`: duplicate-reference bug fixed. |
| D7 | `mi_quantile` ordering | n/a | `[ref]+low+high` is split sequentially, so train holds mostly the low-MI half | Since 1.1.0 each half is a stratum; the MIGT ordering is kept with `rounding="per_bin"`. |
| D8 | EM initialisation | "Randomly initialise the model parameters" | n/a (no GMM) | `gmm_init="kmeans"` default. This is scikit-learn's default and the thesis states it used scikit-learn with tol 0.001 / 100 iterations, which are also scikit-learn defaults. k-means initialisation is itself random and seeded. `gmm_init="random"` follows the literal wording. |
| D9 | Dew class counts | Fig. 5.4: 704 (GMM-MIGT) vs 697 (MIGT) binned images; Table 3.3: 700 Dew images | n/a | Cannot be explained from the text. Possibly augmentation or images skipped by MIGT. Noted only. |

### Details not specified by the thesis (choices made)

| Detail | Choice | Rationale |
|---|---|---|
| Image fed to PCA | Float GMM output (cluster means) without intermediate quantisation | Avoids extra rounding. Algorithm 5.1 does not mention saving the intermediate image. |
| PCA standardisation | Mean-centring only | The six PCA steps list mean subtraction only. Eq. 5.6 (standard deviation) is described as a "main parameter", not as a scaling step. |
| CPV tie rule | Smallest `k` with CPV ≥ 95 % | §5.2.2 wording ("capture a specific percentage"). scikit-learn's `n_components=0.95` uses "> 95 %". The two differ only at exact ties. |
| Processed image dtype | Rounded and clipped to the input dtype (uint8 for 8-bit images) | Algorithm 5.1 output = "processed images". |
| Convergence criterion | scikit-learn's change in mean log-likelihood lower bound per pixel | This is what scikit-learn's `tol` means. |
| Grayscale images | 1-D EM-GMM / single-channel PCA | Direct specialisation of the equations. |
| Alpha channels | Discarded on load | MIGT would fail on 4-channel images in colour MI. |

## Stage 5 – Architecture

```text
src/gmm_migt/
  __init__.py        public API (__all__)
  _version.py        single version source
  exceptions.py      GMMMIGTError, ConfigurationError, ImageLoadError, ImageShapeError
  utils.py           split_counts, list_images, list_classes   (from migt.utils)
  mi.py              load_image, compute_mi, gray / colour estimators (from migt.mi)
  preprocessing.py   NEW: PreprocessingConfig, em_gmm_colour_reduction, pca_reconstruction, preprocess_image
  binning.py         histogram_bins, adaptive_histogram, forced_quantile_split (from MIGTSplitter methods)
  splitter.py        GMMMIGTSplitter (MIGTSplitter logic + preprocessing before MI)
  report.py          SplitResult, ClassReport, ItemRecord, BinSummary (JSON, copy_to)
  baselines.py       RandomSplitter (Random comparator of the thesis)
  plotting.py        optional histogram plots (matplotlib)
  cli.py, __main__.py  `gmm-migt` command
```

Design principles:

* MIGT logic is kept line for line where possible (comments retained).
* The new science lives in a single module (`preprocessing.py`), so it can
  be tested and reused on its own.
* `plan()` separates computation from file I/O. Files are only copied after
  every class has been processed successfully.
* Every methodological choice is a parameter with a documented default.
* There is no dependency on `migt`, so the two packages coexist.

## Traceability

| Code | Thesis |
|---|---|
| `preprocessing.em_gmm_colour_reduction` | §5.2.1, Eq. 5.1–5.4, Alg. 5.1 lines 1–3 |
| `PreprocessingConfig.n_components=4`, `gmm_tol=1e-3`, `gmm_max_iter=100` | §5.2.1 (p. 102–103), Alg. 5.1 "Initialise" |
| `preprocessing.pca_channel_reconstruction`, `select_n_components` | §5.2.2, Eq. 5.5–5.8, CPV |
| `pca_variance=0.95` | §5.2.2 (p. 107), Alg. 5.1 line 4 |
| `GMMMIGTSplitter._plan_class` (MI of processed images) | §5.2 (end), Fig. 5.3 |
| `mi.compute_gray_mi` / `compute_color_mi` | §4.2.1 Eq. 4.1–4.7, §3.3 |
| `binning.normalise_mi`, `ItemRecord.normalized_mi` | Eq. 4.9 |
| `binning.adaptive_histogram` | §4.2.2, Algorithm 1 lines 3–12 |
| `binning.forced_quantile_split` | §4.2.2, Algorithm 1 lines 13–23 |
| `utils.split_counts` | §4.2.2 (5/4/1 guarantee) |
| `GMMMIGTSplitter.from_thesis` | §4.2.2, §5.2 |
| `RandomSplitter` | §4.3, §5.3 (Random comparator / fixed test set) |
