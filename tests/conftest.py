from pathlib import Path

import pytest
from PIL import Image

from diploma.data import write_split


def _line_image(path: Path, width: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, 10), "white").save(path)


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    """A miniature data/ and splits/ with two makarov pages and three Digital Peter lines."""
    data, splits = tmp_path / "data", tmp_path / "splits"
    lines = {
        ("main", "100000-1"): "Рѣшенiе объявлено",
        ("main", "100000-2"): "судъ",
        ("short", "100000-3"): "№ 82.",
        ("main", "200000-1"): "cторонамъ",
    }
    for (subset, stem), text in lines.items():
        folder = data / "strings" / subset
        _line_image(folder / f"{stem}.jpg", width=10 * len(text))
        (folder / f"{stem}.gt.txt").write_text(text + "\n", encoding="utf-8")
    write_split(splits / "makarov.json", {"train": ["100000"], "val": [], "test": ["200000"]}, seed=0)

    peter = data / "digital-peter" / "data_anon"
    rows = [("0_1_0", "1,0,0", "катеринушка"), ("0_1_1", "0,0,1", "друг мой"), ("0_2_0", "0,0,0", "лишняя")]
    mapper = ",new_name,old_name,train,public,private\n"
    for i, (stem, flags, text) in enumerate(rows):
        _line_image(peter / "images" / f"{stem}.jpg")
        (peter / "words").mkdir(parents=True, exist_ok=True)
        (peter / "words" / f"{stem}.txt").write_text(text, encoding="utf-8")
        mapper += f"{i},{stem}.jpg,old_{stem}.jpg,{flags}\n"
    (peter / "mapper.csv").write_text(mapper, encoding="utf-8")
    return data, splits
