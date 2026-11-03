"""GMM-MIGT dataset splitter.

:class:`GMMMIGTSplitter` is built on the MIGT 1.1.1 ``MIGTSplitter``: the
same constructor arguments, the same per-class decision logic (adaptive MI
histogram 4 -> 3 -> 2, forced MI quantile fallback, low-sample strategies,
reference image always in train) and the same output folder structure.

The GMM-MIGT difference (thesis Chapter 5, Algorithm 5.1) is that every
image - including the reference - is first passed through two
preprocessing steps, EM-GMM colour reduction and per-channel PCA
reconstruction (see :mod:`gmm_migt.preprocessing`), and the MI similarity
scores are computed between the *processed* images. The partition itself
copies the *original* image files.
"""

from __future__ import annotations

import math
import os
import random
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from ._version import __version__
from .binning import (
    HISTOGRAM_CONVENTIONS,
    BinInfo,
    HistogramAttempt,
    adaptive_histogram,
    forced_quantile_split,
    interval_labels,
)
from .exceptions import ConfigurationError, ImageLoadError, ImageShapeError
from .mi import COLOR_MI_ESTIMATORS, MI_MODES, compute_mi, is_grayscale, load_image, save_image
from .preprocessing import PreprocessingConfig, preprocess_image
from .report import REPORT_FILENAME, BinSummary, ClassReport, ItemRecord, SplitResult
from .utils import list_classes, list_images, split_counts

LOW_SAMPLE_STRATEGIES = ("mi_quantile", "random")


# --------------------------------------------------
# Per-image scoring (module level so it can run in worker processes)
# --------------------------------------------------
def _score_one(
    path: str,
    ref_proc: np.ndarray,
    target_size: Optional[Tuple[int, int]],
    config: PreprocessingConfig,
    mode: str,
    color_mi: str,
    processed_path: Optional[str],
) -> Dict[str, Any]:
    """Load, preprocess and MI-score one image against the processed reference."""
    rec: Dict[str, Any] = {"path": path, "mi": None, "reason": None}
    try:
        img = load_image(path, target_size)
        if img.shape[:2] != ref_proc.shape[:2]:
            raise ImageShapeError(f"Image shape mismatch: {ref_proc.shape} vs {img.shape}")
        proc, info = preprocess_image(img, config)
        if processed_path is not None:
            save_image(processed_path, proc)
        mi = compute_mi(ref_proc, proc, mode, color_mi=color_mi)
        if info.gmm is not None:
            rec["gmm_converged"] = info.gmm.converged
            rec["gmm_n_iter"] = info.gmm.n_iter
        if info.pca is not None:
            rec["pca_components"] = list(info.pca.n_components)
        if np.isnan(mi):
            rec["reason"] = "MI is NaN"
        else:
            rec["mi"] = float(mi)
    except Exception as exc:  # noqa: BLE001 - MIGT skips any image whose MI fails
        rec["reason"] = f"{type(exc).__name__}: {exc}"
    return rec


class GMMMIGTSplitter:
    """Partition an image-classification dataset with GMM-MIGT.

    The dataset must have one sub-directory per class::

        dataset_root/
            class_a/ img1.jpg img2.jpg ...
            class_b/ ...

    Parameters
    ----------
    dataset_root:
        Path of the dataset directory.
    mode:
        MI mode: ``"auto"`` (grayscale MI for grayscale images, otherwise
        colour MI), ``"grayscale"`` or ``"color"``.
    bins:
        Initial number of histogram bins; reduced one by one down to 2.
    min_bin:
        Minimum number of samples required in every histogram bin (default
        10, the thesis value; the MIGT 1.1.1 code uses 13). Classes with
        ``<= min_bin`` images use ``low_sample_strategy``.
    train, test, val:
        Split ratios (must be > 0 and sum to 1). ``val=None`` creates only
        train and test.
    strict_shape:
        ``True``: images must have the reference image's height and width
        (others are skipped). ``False``: images are resized to ``resize_to``
        in memory before preprocessing and MI.
    resize_to:
        ``(width, height)``; required when ``strict_shape=False``.
    seed:
        Seed of the within-bin shuffling and of the EM-GMM initialisation.
    verbose:
        Print progress.
    low_sample_strategy:
        ``"mi_quantile"`` (two MI portions) or ``"random"`` for small classes.
    use_gmm, n_components, gmm_tol, gmm_max_iter, gmm_covariance_type,
    gmm_init, gmm_reg_covar, gmm_max_pixels:
        EM-GMM colour reduction (thesis Section 5.2.1). Defaults are the
        thesis values: K = 4, tolerance 0.001, 100 iterations, full
        covariances.
    use_pca, pca_variance:
        Per-channel PCA reconstruction keeping a Cumulative Percent Variance
        of ``pca_variance`` (thesis Section 5.2.2; default 0.95).
    color_mi:
        Colour MI estimator: ``"nmi"`` (MIGT 1.1.1) or
        ``"mutual_info_score"`` (thesis Section 3.3).
    histogram_convention:
        ``"migt"`` (MIGT 1.1.1 code) or ``"thesis"`` (thesis Algorithm 1),
        see :mod:`gmm_migt.binning`.
    fallback_quantiles:
        Number of equal MI portions used when the histogram fails (MIGT and
        thesis: 3).
    n_jobs:
        Number of worker processes for preprocessing + MI (``-1`` = all
        cores). Results do not depend on ``n_jobs``.
    processed_root:
        If given, the processed (EM-GMM + PCA) images are saved as PNG to
        ``processed_root/<class>/`` (the "Processed images" output of
        Algorithm 5.1).

    Examples
    --------
    >>> from gmm_migt import GMMMIGTSplitter
    >>> splitter = GMMMIGTSplitter(dataset_root="dataset")      # doctest: +SKIP
    >>> result = splitter.run(output_root="gmm_migt_output")    # doctest: +SKIP
    >>> print(result.summary())                                 # doctest: +SKIP
    """

    def __init__(
        self,
        dataset_root: str,
        mode: str = "auto",
        bins: int = 4,
        min_bin: int = 10,
        train: float = 0.8,
        test: float = 0.2,
        val: Optional[float] = None,
        strict_shape: bool = True,
        resize_to: Optional[Tuple[int, int]] = None,
        seed: int = 42,
        verbose: bool = True,
        low_sample_strategy: str = "mi_quantile",  # "mi_quantile" | "random"
        # ---- GMM-MIGT preprocessing (thesis Section 5.2 / Algorithm 5.1) ----
        use_gmm: bool = True,
        n_components: int = 4,
        gmm_tol: float = 1e-3,
        gmm_max_iter: int = 100,
        gmm_covariance_type: str = "full",
        gmm_init: str = "kmeans",
        gmm_reg_covar: float = 1e-6,
        gmm_max_pixels: Optional[int] = None,
        use_pca: bool = True,
        pca_variance: float = 0.95,
        # ---- MI / binning options ----
        color_mi: str = "nmi",
        histogram_convention: str = "migt",
        fallback_quantiles: int = 3,
        # ---- execution ----
        n_jobs: int = 1,
        processed_root: Optional[str] = None,
    ):
        self.dataset_root = os.fspath(dataset_root)
        self.mode = mode
        self.start_bins = bins
        self.min_bin = min_bin
        self.strict_shape = strict_shape
        self.resize_to = tuple(resize_to) if resize_to is not None else None
        self.seed = seed
        self.verbose = verbose
        self.low_sample_strategy = low_sample_strategy
        self.color_mi = color_mi
        self.histogram_convention = histogram_convention
        self.fallback_quantiles = fallback_quantiles
        self.n_jobs = n_jobs
        self.processed_root = os.fspath(processed_root) if processed_root is not None else None

        self._validate_basic()

        # ratios
        if val is None:
            self.val: Optional[float] = None
            self.ratios = [float(train), float(test)]
        else:
            self.val = float(val)
            self.ratios = [float(train), float(test), self.val]
        self.train, self.test = float(train), float(test)

        if any(not math.isfinite(r) or r <= 0 for r in self.ratios):
            raise ConfigurationError("train/test/val must be > 0.")
        s = sum(self.ratios)
        if not np.isclose(s, 1.0):
            hint = (" For a train/test split only, pass val=None." if val is not None
                    else f" With val=None, train + test must be 1 (or pass val={1 - s:g} for a three-way split).")
            raise ConfigurationError(f"train/test/val ratios must sum to 1.0 (got {s:g}).{hint}")

        self.preprocessing = PreprocessingConfig(
            use_gmm=use_gmm,
            n_components=n_components,
            gmm_tol=gmm_tol,
            gmm_max_iter=gmm_max_iter,
            gmm_covariance_type=gmm_covariance_type,
            gmm_init=gmm_init,
            gmm_reg_covar=gmm_reg_covar,
            gmm_max_pixels=gmm_max_pixels,
            use_pca=use_pca,
            pca_variance=pca_variance,
            random_state=seed,
        )

    # --------------------------------------------------
    # Construction helpers
    # --------------------------------------------------
    @classmethod
    def from_thesis(cls, dataset_root: str, **overrides: Any) -> "GMMMIGTSplitter":
        """Splitter configured with the parameter values reported in the thesis.

        * ``min_bin=10`` and ``train/test/val = 0.5/0.4/0.1`` (Section 4.2.2),
        * ``histogram_convention="thesis"`` (Algorithm 1, Figures 4.6-4.12),
        * ``bins=4``, ``n_components=4``, ``gmm_tol=0.001``,
          ``gmm_max_iter=100``, ``pca_variance=0.95`` (Algorithm 5.1).

        Any argument can be overridden by keyword.
        """
        params: Dict[str, Any] = dict(
            bins=4,
            min_bin=10,
            train=0.5,
            test=0.4,
            val=0.1,
            histogram_convention="thesis",
            n_components=4,
            gmm_tol=1e-3,
            gmm_max_iter=100,
            pca_variance=0.95,
        )
        params.update(overrides)
        return cls(dataset_root, **params)

    def _validate_basic(self) -> None:
        if not os.path.isdir(self.dataset_root):
            raise ConfigurationError(f"dataset_root is not a directory: {self.dataset_root!r}")
        if self.mode not in MI_MODES:
            raise ConfigurationError(f"mode must be one of {MI_MODES} (got {self.mode!r}).")
        if not isinstance(self.start_bins, (int, np.integer)) or isinstance(self.start_bins, bool) or self.start_bins < 2:
            raise ConfigurationError(f"bins must be an integer >= 2 (got {self.start_bins!r}).")
        if not isinstance(self.min_bin, (int, np.integer)) or isinstance(self.min_bin, bool) or self.min_bin < 1:
            raise ConfigurationError(f"min_bin must be an integer >= 1 (got {self.min_bin!r}).")
        self.start_bins, self.min_bin = int(self.start_bins), int(self.min_bin)
        if not self.strict_shape and self.resize_to is None:
            raise ConfigurationError(
                "When strict_shape=False, resize_to must be provided, e.g. resize_to=(224,224)."
            )
        if self.resize_to is not None:
            if len(self.resize_to) != 2 or not all(
                isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v > 0 for v in self.resize_to
            ):
                raise ConfigurationError(f"resize_to must be (width, height) positive integers (got {self.resize_to!r}).")
            if self.strict_shape:
                warnings.warn("resize_to is ignored because strict_shape=True.", UserWarning, stacklevel=3)
        if self.low_sample_strategy not in LOW_SAMPLE_STRATEGIES:
            raise ConfigurationError("low_sample_strategy must be 'mi_quantile' or 'random'.")
        if self.color_mi not in COLOR_MI_ESTIMATORS:
            raise ConfigurationError(f"color_mi must be one of {COLOR_MI_ESTIMATORS} (got {self.color_mi!r}).")
        if self.histogram_convention not in HISTOGRAM_CONVENTIONS:
            raise ConfigurationError(
                f"histogram_convention must be one of {HISTOGRAM_CONVENTIONS} (got {self.histogram_convention!r})."
            )
        if not isinstance(self.fallback_quantiles, (int, np.integer)) or self.fallback_quantiles < 2:
            raise ConfigurationError(f"fallback_quantiles must be an integer >= 2 (got {self.fallback_quantiles!r}).")
        if not isinstance(self.n_jobs, (int, np.integer)) or self.n_jobs == 0:
            raise ConfigurationError(f"n_jobs must be a non-zero integer (got {self.n_jobs!r}).")
        if self.seed is not None and (not isinstance(self.seed, (int, np.integer)) or isinstance(self.seed, bool)):
            raise ConfigurationError(f"seed must be an integer or None (got {self.seed!r}).")

    def get_config(self) -> Dict[str, Any]:
        """All parameters of this splitter (stored in every report)."""
        p = self.preprocessing
        return {
            "dataset_root": os.path.abspath(self.dataset_root),
            "mode": self.mode,
            "bins": self.start_bins,
            "min_bin": self.min_bin,
            "train": self.train,
            "test": self.test,
            "val": self.val,
            "strict_shape": self.strict_shape,
            "resize_to": list(self.resize_to) if self.resize_to else None,
            "seed": self.seed,
            "low_sample_strategy": self.low_sample_strategy,
            "use_gmm": p.use_gmm,
            "n_components": p.n_components,
            "gmm_tol": p.gmm_tol,
            "gmm_max_iter": p.gmm_max_iter,
            "gmm_covariance_type": p.gmm_covariance_type,
            "gmm_init": p.gmm_init,
            "gmm_reg_covar": p.gmm_reg_covar,
            "gmm_max_pixels": p.gmm_max_pixels,
            "use_pca": p.use_pca,
            "pca_variance": p.pca_variance,
            "color_mi": self.color_mi,
            "histogram_convention": self.histogram_convention,
            "fallback_quantiles": self.fallback_quantiles,
        }

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_config().items())
        return f"GMMMIGTSplitter({args})"

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @property
    def _target_size(self) -> Optional[Tuple[int, int]]:
        return None if self.strict_shape else self.resize_to  # type: ignore[return-value]

    def _assign(self, paths: Sequence[str]) -> Dict[str, str]:
        """Split ``paths`` in order by the ratios, with no loss (MIGT ``_copy_split``).

        - if val is None: leftovers go to test
        - if val exists: leftovers go to val
        """
        counts = split_counts(len(paths), self.ratios)
        n_train, n_test = counts[0], counts[1]
        out = {}
        for i, p in enumerate(paths):
            if i < n_train:
                out[p] = "train"
            elif i < n_train + n_test:
                out[p] = "test"
            else:
                out[p] = "val" if self.val is not None else "test"
        return out

    def _processed_paths(self, cls: str, images: Sequence[str]) -> Dict[str, Optional[str]]:
        if self.processed_root is None:
            return {p: None for p in images}
        out_dir = os.path.join(self.processed_root, cls)
        os.makedirs(out_dir, exist_ok=True)
        stems = Counter(os.path.splitext(os.path.basename(p))[0] for p in images)
        out: Dict[str, Optional[str]] = {}
        for p in images:
            name = os.path.basename(p)
            stem = os.path.splitext(name)[0]
            out[p] = os.path.join(out_dir, (stem if stems[stem] == 1 else name) + ".png")
        return out

    # --------------------------------------------------
    # Similarity scores (GMM-MIGT: preprocess, then MI)
    # --------------------------------------------------
    def _prepare_reference(self, ref: str, cls: str, processed_path: Optional[str]) -> np.ndarray:
        try:
            ref_img = load_image(ref, self._target_size)
        except ImageLoadError as exc:
            raise ImageLoadError(f"Reference image of class {cls!r} cannot be loaded: {exc}") from exc
        if self.mode == "color" and is_grayscale(ref_img):
            raise ConfigurationError(
                f"mode='color' requires colour images, but the reference image of class {cls!r} "
                f"is grayscale: {ref}. Use mode='auto' or mode='grayscale'."
            )
        ref_proc, _ = preprocess_image(ref_img, self.preprocessing)
        if processed_path is not None:
            save_image(processed_path, ref_proc)
        return ref_proc

    def _score_images(
        self, ref_proc: np.ndarray, paths: Sequence[str], processed: Dict[str, Optional[str]]
    ) -> List[Dict[str, Any]]:
        args = (ref_proc, self._target_size, self.preprocessing, self.mode, self.color_mi)
        it = tqdm(paths, desc="  Preprocessing + MI", disable=not self.verbose)
        if self.n_jobs == 1:
            return [_score_one(p, *args, processed[p]) for p in it]
        from joblib import Parallel, delayed

        return list(Parallel(n_jobs=self.n_jobs)(delayed(_score_one)(p, *args, processed[p]) for p in it))

    # --------------------------------------------------
    # Report helpers
    # --------------------------------------------------
    @staticmethod
    def _item(path: str, split: str, score: Optional[Dict[str, Any]], mi_max: Optional[float],
              bin_index: Optional[int], role: str = "sample") -> ItemRecord:
        mi = score.get("mi") if score else None
        norm = None
        if mi is not None and mi_max is not None:
            norm = mi / mi_max if mi_max > 0 else 0.0
        return ItemRecord(
            path=path,
            split=split,
            role=role,
            mi=mi,
            normalized_mi=norm,
            bin=bin_index,
            gmm_converged=score.get("gmm_converged") if score else None,
            gmm_n_iter=score.get("gmm_n_iter") if score else None,
            pca_components=score.get("pca_components") if score else None,
        )

    def _bin_summaries(self, bins: Sequence[BinInfo], mi_max: float, histogram: bool) -> List[BinSummary]:
        def nz(x: Optional[float]) -> Optional[float]:
            if x is None or not math.isfinite(x):
                return None
            return x / mi_max if mi_max > 0 else 0.0

        out = []
        n = len(bins)
        for i, b in enumerate(bins):
            lower = b.lower if histogram else b.mi_min
            upper = b.upper if histogram else b.mi_max
            s = BinSummary(
                index=i,
                size=b.size,
                mi_min=float(b.mi_min),
                mi_max=float(b.mi_max),
                lower=lower,
                upper=upper,
                normalized_lower=nz(lower),
                normalized_upper=nz(upper),
            )
            if s.normalized_lower is not None and s.normalized_upper is not None:
                if histogram:
                    # label each bin from its own edges (migt may drop empty bins)
                    lab = interval_labels([s.normalized_lower, s.normalized_upper], self.histogram_convention)[0]
                    if self.histogram_convention == "thesis" and i > 0:
                        lab = "(" + lab[1:]
                    elif self.histogram_convention == "migt" and i < n - 1:
                        lab = lab[:-1] + ")"
                    s.label = lab
                else:
                    s.label = f"part {i + 1}: [{s.normalized_lower:.2f}, {s.normalized_upper:.2f}]"
            out.append(s)
        return out

    # --------------------------------------------------
    # Low-sample strategy (ONLY when class is small from start)
    # --------------------------------------------------
    def _low_sample_split(self, cls: str, images: List[str], report: ClassReport, rng: random.Random) -> None:
        """Called ONLY when ``len(images) <= min_bin`` (as in MIGT).

        - ``random``: random order; the reference image is placed in train.
        - ``mi_quantile``: MI-sorted halves, each shuffled; ``[ref] + low +
          high`` is split by the ratios (MIGT 1.1.1 behaviour).
        """
        ref = images[0]
        others = images[1:]

        def random_split(paths: List[str], strategy: str) -> None:
            paths = paths[:]
            rng.shuffle(paths)
            # The reference occupies the first train slot, so it is never
            # duplicated into test/val (MIGT 1.1.1 copied it a second time).
            assigned = self._assign([ref] + paths)
            report.strategy = strategy
            report.items = [self._item(ref, assigned[ref], None, None, None, "reference")]
            report.items += [self._item(p, assigned[p], None, None, None) for p in paths]

        # ---- RANDOM ----
        if self.low_sample_strategy == "random":
            random_split(others, "low_sample_random")
            return

        # ---- MI QUANTILE-2 ----
        processed = self._processed_paths(cls, images)
        ref_proc = self._prepare_reference(ref, cls, processed[ref])
        scores = self._score_images(ref_proc, others, processed)
        by_path = {s["path"]: s for s in scores}
        mi_list = [(s["path"], s["mi"]) for s in scores if s["mi"] is not None]

        # if MI fails, fallback to random (still low-sample case allowed)
        if len(mi_list) < 2:
            self._log("  Too few MI values -> random split")
            random_split(others, "low_sample_random_fallback")
            return

        report.skipped = [{"path": s["path"], "reason": s["reason"]} for s in scores if s["mi"] is None]
        halves = forced_quantile_split(mi_list, 2)
        mi_max = max(mi for _, mi in mi_list)
        report.mi_max = mi_max
        report.bins = self._bin_summaries(halves, mi_max, histogram=False)
        report.strategy = "low_sample_mi_quantile"

        ordered: List[str] = []
        bin_of: Dict[str, int] = {}
        for i, h in enumerate(halves):
            paths = [p for p, _ in h.items]
            rng.shuffle(paths)
            ordered += paths
            bin_of.update({p: i for p in paths})

        assigned = self._assign([ref] + ordered)
        report.items = [self._item(ref, assigned[ref], None, mi_max, None, "reference")]
        report.items += [self._item(p, assigned[p], by_path[p], mi_max, bin_of[p]) for p in ordered]

    # --------------------------------------------------
    # One class
    # --------------------------------------------------
    def _plan_class(self, cls: str, rng: random.Random) -> ClassReport:
        class_dir = os.path.join(self.dataset_root, cls)
        images = list_images(class_dir)
        report = ClassReport(name=cls, n_images=len(images), strategy="empty")
        self._log(f"\n> Processing class: {cls}")
        self._log(f"  Found {len(images)} images")

        if len(images) == 0:
            self._log("  Skipped: no images found.")
            return report
        report.reference = images[0]

        # -----------------------------
        # CASE A: class is small from the start
        # -----------------------------
        if len(images) <= self.min_bin:
            self._log(f"  Class too small (<= min_bin={self.min_bin}) -> low_sample_strategy='{self.low_sample_strategy}'")
            self._low_sample_split(cls, images, report, rng)
            return report

        # -----------------------------
        # CASE B: class is big -> GMM-MIGT preprocessing + MI + histogram
        # -----------------------------
        ref = images[0]
        processed = self._processed_paths(cls, images)
        ref_proc = self._prepare_reference(ref, cls, processed[ref])
        scores = self._score_images(ref_proc, images[1:], processed)
        by_path = {s["path"]: s for s in scores}
        mi_list = [(s["path"], s["mi"]) for s in scores if s["mi"] is not None]
        report.skipped = [{"path": s["path"], "reason": s["reason"]} for s in scores if s["mi"] is None]

        self._log(f"  Valid MI values: {len(mi_list)} / {len(images) - 1}")
        if report.skipped:
            self._log(f"  Skipped {len(report.skipped)} image(s) whose MI could not be computed "
                      f"(e.g. {report.skipped[0]['reason']})")

        attempts: List[HistogramAttempt] = []
        histogram = True
        # If MI list is too small, we can't histogram reliably -> FORCED quantile
        if len(mi_list) < 3:
            self._log(f"  Too few MI values for histogram -> FORCED Quantile-{self.fallback_quantiles}")
            bins = forced_quantile_split(mi_list, self.fallback_quantiles)
            histogram = False
        else:
            found = adaptive_histogram(mi_list, self.start_bins, self.min_bin, self.histogram_convention, attempts)
            # -----------------------------
            # CASE C: histogram failed bins -> 2
            # must be FORCED quantile split (no random)
            # -----------------------------
            if found is None:
                self._log(f"  Histogram failed ({self.start_bins}->2) -> FORCED Quantile-{self.fallback_quantiles} (MI-based)")
                bins = forced_quantile_split(mi_list, self.fallback_quantiles)
                histogram = False
            else:
                bins = found

        report.strategy = "histogram" if histogram else "forced_quantile"
        report.histogram_attempts = [a.to_dict() for a in attempts]
        mi_max = max((mi for _, mi in mi_list), default=None)
        report.mi_max = mi_max
        if bins:
            report.bins = self._bin_summaries(bins, mi_max if mi_max is not None else 0.0, histogram)
        self._log(f"  Using bins={len(bins)} | sizes={[b.size for b in bins]}")

        # split each bin (ONLY MI images); ref is not in bins, it goes to train.
        report.items = [self._item(ref, "train", None, mi_max, None, "reference")]
        for i, b in enumerate(bins):
            paths = [p for p, _ in b.items]
            rng.shuffle(paths)
            assigned = self._assign(paths)
            report.items += [self._item(p, assigned[p], by_path[p], mi_max, i) for p in paths]
        return report

    # --------------------------------------------------
    # Main
    # --------------------------------------------------
    def plan(self) -> SplitResult:
        """Compute the partition without copying any files.

        The result is verified with :meth:`SplitResult.verify`: every image
        file of every class is either in exactly one split or listed in
        ``result.skipped()``. Skipped images also trigger a ``UserWarning``.
        """
        rng = random.Random(self.seed)
        result = SplitResult(config=self.get_config(), version=__version__)
        for cls in list_classes(self.dataset_root):
            result.classes[cls] = self._plan_class(cls, rng)
        result.verify(self.dataset_root)
        skipped = result.skipped()
        if skipped:
            hint = (" Images of a different size can be included with strict_shape=False and resize_to=(w, h)."
                    if self.strict_shape else "")
            warnings.warn(
                f"{len(skipped)} image(s) could not be MI-scored and are not in any split, e.g. "
                f"{skipped[0]['path']} ({skipped[0]['reason']}). See result.skipped().{hint}",
                UserWarning,
                stacklevel=2,
            )
        names = result.split_names
        empty = [f"{c} ({s})" for c, r in result.classes.items()
                 if r.n_images >= len(names) for s in names if r.counts[s] == 0]
        if empty:
            warnings.warn(
                f"No image assigned to: {', '.join(empty[:5])}{' ...' if len(empty) > 5 else ''}. "
                "Each bin is split separately and rounding gives the remainder to train, so bins "
                "smaller than about 1/ratio images add nothing to test/val. A larger min_bin gives "
                "larger bins.",
                UserWarning,
                stacklevel=2,
            )
        return result

    def run(self, output_root: str, write_report: bool = True) -> SplitResult:
        """Partition the dataset and copy the original images to ``output_root``.

        Creates ``output_root/train/<class>/``, ``output_root/test/<class>/``
        and, if ``val`` is set, ``output_root/val/<class>/``. With
        ``write_report=True`` the full :class:`SplitResult` is also written to
        ``output_root/gmm_migt_report.json``.
        """
        out = os.path.abspath(os.fspath(output_root))
        root = os.path.abspath(self.dataset_root)
        if out == root or out.startswith(root + os.sep):
            raise ConfigurationError(
                f"output_root must not be dataset_root or lie inside it (got {output_root!r})."
            )
        result = self.plan()
        result.copy_to(out, write_report=write_report)
        if self.verbose:
            print("\nGMM-MIGT splitting finished.")
            if write_report:
                print(f"Report written to {os.path.join(out, REPORT_FILENAME)}")
        return result
