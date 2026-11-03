# Validation report

All checks can be re-run:

```bash
pip install -e ".[dev]"
pytest                                   # automated test suite
pip install migt                         # optional: enables the MIGT equivalence checks
python validation/run_validation.py --migt
```

Results below were obtained with Python 3.12, NumPy 2.5, scikit-learn 1.9,
scikit-image 0.26 and OpenCV 5.0 on Windows 11. They are stored in
`validation/results/`.

## 1. Correctness of the algorithm components

| Check | Method | Result |
|---|---|---|
| EM equations (Eq. 5.1–5.4) | A literal NumPy implementation of the E-step and M-step is run from the same initial parameters as scikit-learn's `GaussianMixture`, for 7 iterations (`test_sklearn_em_equals_thesis_equations`). | Weights, means and covariances identical (rtol 1e-6) |
| Colour reduction (§5.2.1, step 5) | Output colours ⊆ fitted means; every pixel is assigned by argmax responsibility | pass |
| PCA (Eq. 5.5–5.8 + CPV) | Mean, covariance with n−1, `eigh`, smallest k with CPV ≥ threshold, projection and reconstruction, compared with the package for thresholds 0.5, 0.85, 0.95 and 0.99 | identical k and reconstruction (atol 1e-8) |
| PCA per channel, rows = observations | Each channel compared with an independent single-channel reconstruction | pass |
| MI (Eq. 4.1–4.7) | `H(A)+H(B)−H(A,B)` from the joint histogram vs `compute_gray_mi` | identical (rtol 1e-10) |
| `split_counts` | sum = n for 0–59 items × 4 ratio sets; the thesis guarantee 10 → 5/4/1 | pass |

## 2. Binning decisions vs. the thesis figures

MI values were constructed to reproduce the bin counts printed in the
thesis. Algorithm 1 (MinBin = 10, thesis convention) then makes the same
decision as reported:

| Figure | Attempts (bins: sizes) | Final | Thesis |
|---|---|---|---|
| 4.6 Little leaf privet | 4: [19,17,45,18] ✓ | [19,17,45,18] | 4 bins ✓ |
| 4.8 Snow | 4: [30,5,315,269] ✗ → 3: [30,220,369] ✓ | [30,220,369] | 3 bins ✓ |
| 4.10 Holarrhena pubescens | 4: [33,47,15,4] ✗ → 3: [55,38,6] ✗ → 2: [80,19] ✓ | [80,19] | 4 → 3 → 2 ✓ |
| 4.12 Allium ramosum | … → 2: [91,8] ✗ → 3 equal portions | [33,33,33] | equal portions ✓ |
| 5.4(a) Dew, MIGT | 4: [111,405,167,14] ✓ | same | ✓ |
| 5.4(b) Dew, GMM-MIGT | 4: [82,506,106,10] ✓ | same | ✓ |

The three histograms of Figure 4.10 are mutually consistent, so a single
MI dataset reproduces all of them. With MIGT 1.1.1's `min_bin=13`, the
published GMM-MIGT Dew histogram (Fig. 5.4b) would have been rejected (the
procedure moves on to 3 bins). This confirms that the thesis experiments
used MinBin = 10 (discrepancy D1).

## 3. Equivalence with MIGT 1.1.1

`GMMMIGTSplitter(use_gmm=False, use_pca=False)` with MIGT's defaults (and the same `min_bin`) was
compared file by file with the original `migt` package (installed from a
copy of the provided source) on synthetic datasets with 3 classes:

* `min_bin` 10 (GMM-MIGT default) and 13 (MIGT default), seeds 0, 1 and 42,
  with `val=None` and with `val=0.1`: **identical
  train/test/val assignments**;
* MI values for all three modes: **identical**.

This shows that the GMM-MIGT package changes MIGT only through the added
preprocessing.

## 4. EM-GMM + PCA on real photographs

On scikit-learn's two sample photographs (427 × 640 RGB), see
`validation/results/v1_em_gmm_pca_figure_5_2_layout.png` (same layout as
thesis Figure 5.2):

| Image | Colours before → after EM-GMM | EM iterations | PCA components (B, G, R) | CPV |
|---|---|---|---|---|
| china.jpg | 96 615 → 4 | 24 (converged) | 23, 45, 64 of 427 | 0.950 |
| flower.jpg | 62 941 → 4 | 12 (converged) | 85, 58, 36 of 427 | 0.951 |

Qualitatively, the result matches Figures 5.1 and 5.2: EM-GMM gives four
homogeneous colour regions, and PCA reconstruction slightly smooths them
while keeping the dominant structures. Preprocessing takes about 3–13 s per
image on one core, depending on the image.

## 5. Controlled synthetic experiment

Five synthetic datasets were generated (3 classes × 120 images). Each
image belongs to one of four latent sub-types (stripes ×2, blob, checker;
proportions 60/30/20/10) and has strong random illumination gain, colour
cast and noise. The partitions were produced with the thesis configuration
(50/40/10, MinBin 10).

| Method | Mean total-variation distance of split sub-type proportions (lower = more representative) | Sub-types missing from a split | NMI(bin, sub-type) |
|---|---|---|---|
| Random | **0.099 ± 0.018** | 8 | – |
| MIGT (no preprocessing) | 0.115 ± 0.016 | 13 | 0.088 |
| GMM-MIGT | 0.109 ± 0.016 | 10 | **0.159** |

Interpretation:

* In this controlled setting, the EM-GMM + PCA preprocessing makes the MI
  bins about twice as informative about the latent sub-type as raw-pixel
  MI. This agrees with the mechanism claimed in §5.5 (more accurate
  similarity scores). It also gives slightly more representative splits
  than MIGT.
* Neither MI-based method beat Random on this proxy. With only 2–4 MI bins
  per class, and a single reference image, the 1-D similarity axis does not
  separate four sub-types well. The result is reported as obtained, with
  no tuning of the experiment.
* **This experiment is not evidence for or against the thesis results.**
  Those concern CNN classification accuracy on real datasets.

## 6. Reproduction of the thesis experiments: what is missing

The classification results of Tables 5.1 and 5.2 could not be reproduced
here, because the following are required and are not in the project
folder:

1. **The datasets**: VNPlant-200 (20 000 images), BJFU100 (10 000 images)
   and WEAPD (6 862 images).
2. **The CNN training pipeline**: Xception (ImageNet weights) fine-tuned
   with a 200-neuron dense layer, Adam (lr 1e-4), early stopping, up to
   100 epochs, augmentation, 224 × 224 input, on a GPU (the thesis used a
   Colab A100).
3. **The fixed Random test set** used for all methods, and a description of
   how overlap between that test set and the GMM-MIGT train/val subsets
   was avoided (not described in §4.3 / §5.3).
4. **Preprocessing details**:
   * whether images (especially BJFU100 at 3120 × 4208 and WEAPD with mixed
     resolutions) were resized before EM-GMM / MI, and to what size;
   * the EM initialisation (D8);
   * the colour MI estimator (D5);
   * the per-class seed handling.
5. **The augmentation status of the binned data.** Fig. 5.4 shows 704
   binned Dew images, while Table 3.3 lists 700.

With these, `GMMMIGTSplitter.from_thesis(...)` plus the documented
protocol (Section 9 of the README) is the intended route to reproduce the
partitions. The validation in Sections 1–3 provides evidence that the
partitioning logic and the preprocessing follow the thesis description
exactly.
