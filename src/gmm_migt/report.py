"""Result objects returned by :class:`gmm_migt.GMMMIGTSplitter`.

A :class:`SplitResult` records, for every class, which strategy was used,
every histogram that was tried, the accepted bins, the MI score of every
image and the split it was assigned to. It can be saved to / loaded from
JSON and materialised (copied) into an output directory, which makes every
partition fully auditable and reproducible.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .exceptions import ConfigurationError, IntegrityError
from .utils import IMAGE_EXTENSIONS, SPLIT_NAMES, list_classes, list_images

#: Strategy identifiers stored in :attr:`ClassReport.strategy`.
STRATEGIES = (
    "histogram",                  # adaptive MI histogram succeeded
    "forced_quantile",            # histogram failed at 2 bins -> equal MI portions
    "low_sample_random",          # small class, low_sample_strategy="random"
    "low_sample_mi_quantile",     # small class, low_sample_strategy="mi_quantile"
    "low_sample_random_fallback", # small class, MI quantile impossible -> random
    "random",                     # RandomSplitter baseline
    "empty",                      # class directory without images
)

REPORT_FILENAME = "gmm_migt_report.json"


def _key(path: str) -> str:
    """Comparable form of a path (absolute; case-insensitive on Windows)."""
    return os.path.normcase(os.path.abspath(path))


def _image_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory)
            if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(directory, f))]


@dataclass
class ItemRecord:
    """One image and the split it was assigned to."""

    path: str
    split: str
    role: str = "sample"  # "reference" | "sample"
    mi: Optional[float] = None
    normalized_mi: Optional[float] = None
    bin: Optional[int] = None
    gmm_converged: Optional[bool] = None
    gmm_n_iter: Optional[int] = None
    pca_components: Optional[List[int]] = None


@dataclass
class BinSummary:
    """One accepted bin / portion of a class."""

    index: int
    size: int
    mi_min: float
    mi_max: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    normalized_lower: Optional[float] = None
    normalized_upper: Optional[float] = None
    label: Optional[str] = None


@dataclass
class ClassReport:
    """Partition record of one class."""

    name: str
    n_images: int
    strategy: str
    reference: Optional[str] = None
    mi_max: Optional[float] = None
    histogram_attempts: List[Dict[str, Any]] = field(default_factory=list)
    bins: List[BinSummary] = field(default_factory=list)
    items: List[ItemRecord] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        c = {s: 0 for s in SPLIT_NAMES}
        for it in self.items:
            c[it.split] += 1
        return c

    def paths(self, split: str) -> List[str]:
        return [it.path for it in self.items if it.split == split]

    def check(self, images: Optional[Sequence[str]] = None) -> None:
        """Raise :class:`IntegrityError` unless every image is accounted for exactly once.

        Every image of the class must be either assigned to one split or
        listed in :attr:`skipped`, and no image may appear twice. If
        ``images`` (the files on disk) is given, the recorded paths must be
        exactly those files.
        """
        listed = [it.path for it in self.items] + [s["path"] for s in self.skipped]
        dup = sorted(p for p, k in Counter(_key(p) for p in listed).items() if k > 1)
        if dup:
            raise IntegrityError(f"Class {self.name!r}: {len(dup)} image(s) recorded more than once, e.g. {dup[0]}")
        if len(listed) != self.n_images:
            raise IntegrityError(
                f"Class {self.name!r}: {self.n_images} image(s) found but {len(self.items)} assigned "
                f"+ {len(self.skipped)} skipped = {len(listed)}."
            )
        if images is not None:
            on_disk = {_key(p) for p in images}
            missing = sorted(on_disk - {_key(p) for p in listed})
            if missing or len(on_disk) != len(listed):
                raise IntegrityError(
                    f"Class {self.name!r}: the partition does not match the {len(on_disk)} image file(s) "
                    f"on disk" + (f", e.g. {missing[0]} is missing" if missing else "") + "."
                )


@dataclass
class SplitResult:
    """Complete record of one partition run."""

    config: Dict[str, Any]
    classes: Dict[str, ClassReport] = field(default_factory=dict)
    version: str = ""
    method: str = "GMM-MIGT"

    # ---------------- convenience views ----------------
    @property
    def split_names(self) -> List[str]:
        return ["train", "test", "val"] if self.config.get("val") is not None else ["train", "test"]

    def splits(self) -> Dict[str, Dict[str, List[str]]]:
        """``{split: {class: [paths]}}``."""
        return {s: {c: r.paths(s) for c, r in self.classes.items()} for s in self.split_names}

    def counts(self) -> Dict[str, Dict[str, int]]:
        """``{class: {split: count}}``."""
        return {c: {s: r.counts[s] for s in self.split_names} for c, r in self.classes.items()}

    def totals(self) -> Dict[str, int]:
        t = {s: 0 for s in self.split_names}
        for r in self.classes.values():
            for s in self.split_names:
                t[s] += r.counts[s]
        return t

    def skipped(self) -> List[Dict[str, str]]:
        return [dict(s, cls=c) for c, r in self.classes.items() for s in r.skipped]

    def accounting(self) -> Dict[str, int]:
        """``{"found", "assigned", "skipped"}`` image counts over all classes."""
        return {
            "found": sum(r.n_images for r in self.classes.values()),
            "assigned": sum(len(r.items) for r in self.classes.values()),
            "skipped": sum(len(r.skipped) for r in self.classes.values()),
        }

    def verify(self, dataset_root: Optional[str] = None) -> None:
        """Check that no image was lost, duplicated or put in an unknown split.

        Raises :class:`IntegrityError` on the first problem. With
        ``dataset_root``, the partition is also compared with the class
        folders and image files currently on disk. Called automatically by
        ``plan()``, ``run()`` and :meth:`copy_to`.
        """
        valid = set(self.split_names)
        for name, r in self.classes.items():
            bad = [it.path for it in r.items if it.split not in valid]
            if bad:
                raise IntegrityError(f"Class {name!r}: image assigned to an unknown split: {bad[0]}")
            images = list_images(os.path.join(dataset_root, name)) if dataset_root is not None else None
            r.check(images)
        if dataset_root is not None:
            missing = sorted(set(list_classes(dataset_root)) - set(self.classes))
            if missing:
                raise IntegrityError(f"Class folder(s) missing from the partition: {missing}")

    def summary(self) -> str:
        """Plain-text table of strategy, bin sizes and split counts per class."""
        cols = self.split_names
        header = f"{'class':<24} {'n':>6} {'strategy':<27} {'bins':<22} " + " ".join(f"{c:>6}" for c in cols)
        lines = [header, "-" * len(header)]
        for name, r in self.classes.items():
            sizes = ",".join(str(b.size) for b in r.bins) if r.bins else "-"
            counts = " ".join(f"{r.counts[c]:>6}" for c in cols)
            lines.append(f"{name[:24]:<24} {r.n_images:>6} {r.strategy:<27} {sizes[:22]:<22} {counts}")
        tot = self.totals()
        lines.append("-" * len(header))
        n_all = sum(r.n_images for r in self.classes.values())
        lines.append(f"{'TOTAL':<24} {n_all:>6} {'':<27} {'':<22} " + " ".join(f"{tot[c]:>6}" for c in cols))
        acc = self.accounting()
        ok = "OK" if acc["found"] == acc["assigned"] + acc["skipped"] else "MISMATCH"
        lines.append(f"Check: {acc['found']} image(s) found = {acc['assigned']} assigned "
                     f"+ {acc['skipped']} skipped [{ok}]")
        if acc["skipped"]:
            lines.append(f"{acc['skipped']} image(s) skipped and not copied (see .skipped()).")
        return "\n".join(lines)

    # ---------------- serialisation ----------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "version": self.version,
            "config": self.config,
            "classes": {c: asdict(r) for c, r in self.classes.items()},
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SplitResult":
        classes = {}
        for name, r in d["classes"].items():
            r = dict(r)
            r["bins"] = [BinSummary(**b) for b in r.get("bins", [])]
            r["items"] = [ItemRecord(**i) for i in r.get("items", [])]
            classes[name] = ClassReport(**r)
        return cls(config=d["config"], classes=classes, version=d.get("version", ""), method=d.get("method", "GMM-MIGT"))

    @classmethod
    def from_json(cls, path: str) -> "SplitResult":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---------------- materialisation ----------------
    def copy_to(self, output_root: str, write_report: bool = True) -> None:
        """Copy every assigned file into ``output_root/<split>/<class>/``.

        Folders are created for every split and class (also for empty
        classes), as in MIGT. Original files are copied unchanged.

        The target folders must not already contain images (e.g. from an
        earlier run), otherwise an image could end up in two splits. After
        copying, every folder is checked to hold exactly the planned files,
        each with the size of its source; :class:`IntegrityError` is raised
        otherwise.
        """
        self.verify()
        for name in self.classes:
            for s in self.split_names:
                if _image_files(os.path.join(output_root, s, name)):
                    raise ConfigurationError(
                        f"{os.path.join(output_root, s, name)!r} already contains images (an earlier run?). "
                        "Use an empty output_root so that no image ends up in two splits."
                    )
        for name, r in self.classes.items():
            for s in self.split_names:
                os.makedirs(os.path.join(output_root, s, name), exist_ok=True)
            for it in r.items:
                shutil.copy(it.path, os.path.join(output_root, it.split, name))
        for name, r in self.classes.items():
            for s in self.split_names:
                d = os.path.join(output_root, s, name)
                planned = {os.path.normcase(os.path.basename(p)): p for p in r.paths(s)}
                copied = {os.path.normcase(f) for f in _image_files(d)}
                if len(planned) != len(r.paths(s)) or copied != set(planned):
                    raise IntegrityError(
                        f"{d!r} holds {len(copied)} image(s) but {len(r.paths(s))} were planned "
                        "(duplicate file names or a failed copy)."
                    )
                for f, src in planned.items():
                    if os.path.getsize(os.path.join(d, f)) != os.path.getsize(src):
                        raise IntegrityError(f"Copy of {src!r} in {d!r} has a different size.")
        if write_report:
            self.to_json(os.path.join(output_root, REPORT_FILENAME))
