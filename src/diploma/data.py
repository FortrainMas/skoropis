"""Line-level HTR datasets with fixed train/val/test splits.

Every dataset is a flat list of :class:`Sample`. ``group`` is the source page:
lines of one page share hand, ink and often whole phrases, so groups are the
unit of splitting and of bootstrap resampling.

Datasets (``name`` or ``name:split`` in :func:`load_dataset`):

* ``makarov`` -- Makarovsky volost court records, 1913-1914 (``data/strings``),
  both subsets; ``makarov-main`` -- the main text column; ``makarov-short`` --
  the narrow column of notes. Page-level split stored in ``splits/makarov.json``.
* ``peter`` -- Digital Peter (AIJ Contest 2020). The competition split is kept
  for comparability with published results: train -> train, public -> val,
  private -> test. Note that it is line-level, so pages do cross splits.
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from diploma.text import canonical

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DIPLOMA_DATA", REPO_ROOT / "data"))
SPLITS_ROOT = REPO_ROOT / "splits"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    id: str
    image: Path
    text: str
    group: str
    split: str


def load_dataset(
    spec: str, *, data_root: Path | None = None, splits_root: Path | None = None
) -> list[Sample]:
    """Load ``name`` (all splits) or ``name:split``, e.g. ``makarov:test``."""
    name, _, split = spec.partition(":")
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}, expected one of: {', '.join(DATASETS)}")
    if split and split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of: {', '.join(SPLITS)}")
    samples = DATASETS[name](data_root or DATA_ROOT, splits_root or SPLITS_ROOT)
    return [s for s in samples if s.split == split] if split else samples


# --- Makarovsky volost court ---------------------------------------------

MAKAROV_SUBSETS = ("main", "short")


def makarov_pages(data_root: Path) -> set[str]:
    """Ids of all pages that have at least one labeled line."""
    return {
        _makarov_page(label)
        for subset in MAKAROV_SUBSETS
        for label in (data_root / "strings" / subset).glob("*.gt.txt")
    }


def _makarov_page(label: Path) -> str:
    # 133601-179.gt.txt is line 179 of photo IMG_20230824_133601.jpg
    return label.name.split("-", 1)[0]


def _load_makarov(subsets: Iterable[str]) -> Loader:
    def load(data_root: Path, splits_root: Path) -> list[Sample]:
        assignment = read_split(splits_root / "makarov.json")
        samples = []
        for subset in subsets:
            for label in sorted((data_root / "strings" / subset).glob("*.gt.txt")):
                page = _makarov_page(label)
                if page not in assignment:
                    raise LookupError(
                        f"page {page} is not in {splits_root / 'makarov.json'}; "
                        "re-create it with `diploma split --force`"
                    )
                stem = label.name.removesuffix(".gt.txt")
                samples.append(
                    Sample(
                        id=f"{subset}/{stem}",
                        image=label.with_name(f"{stem}.jpg"),
                        text=canonical(label.read_text(encoding="utf-8")),
                        group=page,
                        split=assignment[page],
                    )
                )
        return samples

    return load


# --- Digital Peter -------------------------------------------------------

_PETER_SPLITS = {"train": "train", "public": "val", "private": "test"}


def _load_peter(data_root: Path, splits_root: Path) -> list[Sample]:
    base = data_root / "digital-peter" / "data_anon"
    samples = []
    with open(base / "mapper.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = next((ours for theirs, ours in _PETER_SPLITS.items() if row[theirs] == "1"), None)
            if split is None:  # a few lines were left out of the competition split
                continue
            stem = row["new_name"].removesuffix(".jpg")
            samples.append(
                Sample(
                    id=stem,
                    image=base / "images" / row["new_name"],
                    text=canonical((base / "words" / f"{stem}.txt").read_text(encoding="utf-8")),
                    group=stem.rsplit("_", 1)[0],
                    split=split,
                )
            )
    return sorted(samples, key=lambda s: s.id)


Loader = Callable[[Path, Path], list[Sample]]

DATASETS: dict[str, Loader] = {
    "makarov": _load_makarov(MAKAROV_SUBSETS),
    "makarov-main": _load_makarov(["main"]),
    "makarov-short": _load_makarov(["short"]),
    "peter": _load_peter,
}


# --- Group splits --------------------------------------------------------


def make_group_split(groups: Iterable[str], fractions: Mapping[str, float], seed: int) -> dict[str, list[str]]:
    """Shuffle groups with ``seed`` and cut them by count; ``train`` takes the remainder."""
    order = sorted(set(groups))
    random.Random(seed).shuffle(order)
    split: dict[str, list[str]] = {}
    start = 0
    for name, fraction in fractions.items():
        size = round(fraction * len(order))
        split[name] = sorted(order[start : start + size])
        start += size
    split["train"] = sorted(order[start:])
    return {name: split[name] for name in SPLITS}


def write_split(path: Path, split: Mapping[str, list[str]], *, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, **split}, indent=2) + "\n", encoding="utf-8")


def read_split(path: Path) -> dict[str, str]:
    """Return the ``group -> split`` mapping stored at ``path``."""
    stored = json.loads(path.read_text(encoding="utf-8"))
    assignment: dict[str, str] = {}
    for split in SPLITS:
        for group in stored[split]:
            if assignment.setdefault(group, split) != split:
                raise ValueError(f"{path}: group {group} is in both {assignment[group]} and {split}")
    return assignment
