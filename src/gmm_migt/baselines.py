"""Random partitioning baseline.

The thesis compares GMM-MIGT with Random dataset partitioning (Chapters 4-5)
and evaluates every method on a fixed test set obtained from a Random
partition. :class:`RandomSplitter` provides that baseline with the same
ratios, seed handling, output structure and report format as
:class:`gmm_migt.GMMMIGTSplitter`.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np

from ._version import __version__
from .exceptions import ConfigurationError
from .report import ClassReport, ItemRecord, SplitResult
from .utils import list_classes, list_images, split_counts


class RandomSplitter:
    """Per-class random train/test/(val) partition.

    Parameters
    ----------
    dataset_root:
        Dataset directory with one sub-directory per class.
    train, test, val:
        Split ratios (> 0, sum to 1). ``val=None`` for train/test only.
    seed:
        Seed of the shuffling.
    verbose:
        Print progress.
    """

    def __init__(
        self,
        dataset_root: str,
        train: float = 0.8,
        test: float = 0.2,
        val: Optional[float] = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.dataset_root = os.fspath(dataset_root)
        if not os.path.isdir(self.dataset_root):
            raise ConfigurationError(f"dataset_root is not a directory: {self.dataset_root!r}")
        self.train, self.test = float(train), float(test)
        self.val = None if val is None else float(val)
        self.ratios = [self.train, self.test] + ([self.val] if self.val is not None else [])
        if any(r <= 0 for r in self.ratios):
            raise ConfigurationError("train/test/val must be > 0.")
        if not np.isclose(sum(self.ratios), 1.0):
            raise ConfigurationError(f"train/test/val ratios must sum to 1.0 (got {sum(self.ratios):g}).")
        self.seed = seed
        self.verbose = verbose

    def get_config(self) -> Dict[str, Any]:
        return {
            "dataset_root": os.path.abspath(self.dataset_root),
            "train": self.train,
            "test": self.test,
            "val": self.val,
            "seed": self.seed,
        }

    def plan(self) -> SplitResult:
        """Compute the random partition without copying files."""
        rng = random.Random(self.seed)
        result = SplitResult(config=self.get_config(), version=__version__, method="Random")
        names = ["train", "test", "val" if self.val is not None else "test"]
        for cls in list_classes(self.dataset_root):
            images = list_images(os.path.join(self.dataset_root, cls))
            report = ClassReport(name=cls, n_images=len(images), strategy="random" if images else "empty")
            paths = images[:]
            rng.shuffle(paths)
            counts = split_counts(len(paths), self.ratios)
            bounds = np.cumsum(counts)
            for i, p in enumerate(paths):
                split = names[int(np.searchsorted(bounds, i, side="right"))]
                report.items.append(ItemRecord(path=p, split=split))
            result.classes[cls] = report
            if self.verbose:
                print(f"> {cls}: {report.counts}")
        result.verify(self.dataset_root)
        return result

    def run(self, output_root: str, write_report: bool = True) -> SplitResult:
        """Partition and copy the images into ``output_root/<split>/<class>/``."""
        result = self.plan()
        result.copy_to(os.fspath(output_root), write_report=write_report)
        return result
