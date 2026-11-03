"""Command-line interface: ``gmm-migt DATASET OUTPUT [options]``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ._version import __version__
from .binning import HISTOGRAM_CONVENTIONS
from .exceptions import GMMMIGTError
from .mi import COLOR_MI_ESTIMATORS, MI_MODES
from .preprocessing import GMM_COVARIANCE_TYPES, GMM_INIT_METHODS
from .report import REPORT_FILENAME
from .splitter import LOW_SAMPLE_STRATEGIES, GMMMIGTSplitter


def _size(text: str):
    try:
        w, h = (int(v) for v in text.lower().replace(",", "x").split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, e.g. 224x224 (got {text!r})") from exc
    return (w, h)


def _val(text: str):
    return None if text.lower() in ("none", "0", "") else float(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gmm-migt",
        description="Partition an image dataset into train/test/(val) with GMM-MIGT "
                    "(EM-GMM + PCA preprocessing followed by MI-guided histogram partitioning).",
    )
    p.add_argument("dataset_root", help="dataset directory with one sub-directory per class")
    p.add_argument("output_root", nargs="?", help="output directory (omit with --dry-run)")
    p.add_argument("--version", action="version", version=f"gmm-migt {__version__}")
    p.add_argument("--thesis", action="store_true",
                   help="start from the thesis parameter values (min_bin=10, 0.5/0.4/0.1, thesis histogram convention)")
    p.add_argument("--dry-run", action="store_true", help="compute and print the partition without copying files")
    p.add_argument("--report", metavar="PATH", help="also write the JSON report to PATH")

    g = p.add_argument_group("MIGT options")
    g.add_argument("--mode", choices=MI_MODES)
    g.add_argument("--bins", type=int)
    g.add_argument("--min-bin", type=int, help="minimum images per histogram bin (default 10)")
    g.add_argument("--train", type=float)
    g.add_argument("--test", type=float)
    g.add_argument("--val", type=_val, default=argparse.SUPPRESS, help="validation ratio, or 'none'")
    g.add_argument("--resize-to", type=_size, metavar="WxH",
                   help="resize images in memory (sets strict_shape=False)")
    g.add_argument("--seed", type=int)
    g.add_argument("--low-sample-strategy", choices=LOW_SAMPLE_STRATEGIES)
    g.add_argument("--quiet", action="store_true")

    g = p.add_argument_group("GMM-MIGT preprocessing")
    g.add_argument("--no-gmm", action="store_true", help="disable EM-GMM colour reduction")
    g.add_argument("--n-components", type=int, help="number of GMM clusters K (default 4)")
    g.add_argument("--gmm-tol", type=float)
    g.add_argument("--gmm-max-iter", type=int)
    g.add_argument("--gmm-covariance-type", choices=GMM_COVARIANCE_TYPES)
    g.add_argument("--gmm-init", choices=GMM_INIT_METHODS)
    g.add_argument("--gmm-max-pixels", type=int)
    g.add_argument("--no-pca", action="store_true", help="disable PCA reconstruction")
    g.add_argument("--pca-variance", type=float, help="PCA cumulative percent variance (default 0.95)")

    g = p.add_argument_group("advanced")
    g.add_argument("--color-mi", choices=COLOR_MI_ESTIMATORS)
    g.add_argument("--histogram-convention", choices=HISTOGRAM_CONVENTIONS)
    g.add_argument("--fallback-quantiles", type=int)
    g.add_argument("--n-jobs", type=int)
    g.add_argument("--processed-root", help="save processed (EM-GMM + PCA) images here")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_root is None and not args.dry_run:
        print("error: output_root is required unless --dry-run is given", file=sys.stderr)
        return 2

    kw = {}
    for name in ("mode", "bins", "min_bin", "train", "test", "seed", "low_sample_strategy",
                 "n_components", "gmm_tol", "gmm_max_iter", "gmm_covariance_type", "gmm_init",
                 "gmm_max_pixels", "pca_variance", "color_mi", "histogram_convention",
                 "fallback_quantiles", "n_jobs", "processed_root"):
        v = getattr(args, name)
        if v is not None:
            kw[name] = v
    if hasattr(args, "val"):
        kw["val"] = args.val
    if args.resize_to is not None:
        kw["strict_shape"] = False
        kw["resize_to"] = args.resize_to
    if args.no_gmm:
        kw["use_gmm"] = False
    if args.no_pca:
        kw["use_pca"] = False
    kw["verbose"] = not args.quiet

    try:
        factory = GMMMIGTSplitter.from_thesis if args.thesis else GMMMIGTSplitter
        splitter = factory(args.dataset_root, **kw)
        if args.dry_run:
            result = splitter.plan()
        else:
            result = splitter.run(args.output_root)
    except GMMMIGTError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.report:
        result.to_json(args.report)
    print()
    print(result.summary())
    if not args.dry_run:
        print(f"\nOutput: {args.output_root} (report: {REPORT_FILENAME})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
