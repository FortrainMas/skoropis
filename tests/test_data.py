from collections import defaultdict
from pathlib import Path

import pytest

from diploma.data import DATA_ROOT, SPLITS, load_dataset, make_group_split, read_split, write_split


def test_makarov(roots: tuple[Path, Path]) -> None:
    data, splits = roots
    samples = load_dataset("makarov", data_root=data, splits_root=splits)
    assert [(s.id, s.group, s.split) for s in samples] == [
        ("main/100000-1", "100000", "train"),
        ("main/100000-2", "100000", "train"),
        ("main/200000-1", "200000", "test"),
        ("short/100000-3", "100000", "train"),
    ]
    assert samples[2].text == "сторонамъ"
    assert all(s.image.exists() for s in samples)


def test_subset_and_split_filters(roots: tuple[Path, Path]) -> None:
    data, splits = roots
    assert [s.id for s in load_dataset("makarov-short", data_root=data, splits_root=splits)] == ["short/100000-3"]
    assert [s.id for s in load_dataset("makarov:test", data_root=data, splits_root=splits)] == ["main/200000-1"]


def test_unassigned_page_is_an_error(roots: tuple[Path, Path]) -> None:
    data, splits = roots
    write_split(splits / "makarov.json", {"train": ["100000"], "val": [], "test": []}, seed=0)
    with pytest.raises(LookupError, match="200000"):
        load_dataset("makarov", data_root=data, splits_root=splits)


def test_peter_uses_competition_split(roots: tuple[Path, Path]) -> None:
    data, splits = roots
    samples = load_dataset("peter", data_root=data, splits_root=splits)
    assert [(s.id, s.group, s.split, s.text) for s in samples] == [
        ("0_1_0", "0_1", "train", "катеринушка"),
        ("0_1_1", "0_1", "test", "друг мой"),
    ]


@pytest.mark.parametrize("spec", ["nope", "makarov:dev"])
def test_bad_spec(spec: str, roots: tuple[Path, Path]) -> None:
    with pytest.raises((KeyError, ValueError)):
        load_dataset(spec, data_root=roots[0], splits_root=roots[1])


def test_group_split_is_deterministic_and_disjoint() -> None:
    groups = [f"p{i:02}" for i in range(36)]
    split = make_group_split(groups, {"test": 0.2, "val": 0.15}, seed=0)
    assert {k: len(v) for k, v in split.items()} == {"train": 24, "val": 5, "test": 7}
    assert sorted(sum(split.values(), [])) == groups
    assert split == make_group_split(reversed(groups), {"test": 0.2, "val": 0.15}, seed=0)


def test_read_split_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    write_split(path, {"train": ["a"], "val": [], "test": ["a"]}, seed=0)
    with pytest.raises(ValueError, match="both"):
        read_split(path)


@pytest.mark.skipif(not (DATA_ROOT / "strings").exists(), reason="real data not available")
@pytest.mark.parametrize("name", ["makarov", "peter"])
def test_real_data_integrity(name: str) -> None:
    samples = load_dataset(name)
    assert len({s.id for s in samples}) == len(samples)
    assert all(s.text and s.image.exists() for s in samples)
    splits_of_group = defaultdict(set)
    for s in samples:
        splits_of_group[s.group].add(s.split)
    assert {s.split for s in samples} == set(SPLITS)
    if name == "makarov":  # Digital Peter's official split is line-level by design
        assert all(len(v) == 1 for v in splits_of_group.values())
