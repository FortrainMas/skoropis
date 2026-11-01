import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from diploma.cli import main
from diploma.data import Sample, load_dataset
from diploma.evaluation import Prediction, Record, coverage_curve, evaluate, summarize


class ByWidth:
    """Recognizes a fixture image by its width (the fixtures draw 10 px per character)."""

    def __init__(self, answers: dict[int, str], confidence: bool = False) -> None:
        self.answers = answers
        self.confidence = confidence

    def predict(self, images: Sequence[Image.Image]) -> list[str | Prediction]:
        texts = [self.answers.get(image.width, "") for image in images]
        if self.confidence:
            return [Prediction(t, confidence=float(len(t))) for t in texts]
        return texts


@pytest.fixture
def makarov(roots: tuple[Path, Path]) -> list[Sample]:
    return load_dataset("makarov", data_root=roots[0], splits_root=roots[1])


def test_perfect_model(makarov: list[Sample], tmp_path: Path) -> None:
    model = ByWidth({10 * len(s.text): s.text for s in makarov})
    report = evaluate(model, makarov, tmp_path / "out", batch_size=3)
    assert report["lines"] == 4 and report["groups"] == 2
    for scores in report["metrics"].values():
        assert scores["cer"] == 0.0 and scores["line_accuracy"] == 1.0
    assert report["confusions"] == []
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["exact"]["cer"] == 0.0
    assert len((tmp_path / "out" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_errors_are_scored_after_canonicalization(makarov: list[Sample]) -> None:
    answers = {10 * len(s.text): s.text for s in makarov}
    answers[10 * len("Рѣшенiе объявлено")] = "Рѣшеніе  объявлено"  # Cyrillic і, double space: not errors
    answers[10 * len("судъ")] = "Суд"
    report = evaluate(ByWidth(answers), makarov)
    exact, modern = report["metrics"]["exact"], report["metrics"]["modern"]
    assert exact["chars"]["substitutions"] == 1 and exact["chars"]["deletions"] == 1
    assert modern["cer"] == 0.0
    assert {(c["ref"], c["hyp"]) for c in report["confusions"]} == {("с", "С"), ("ъ", "")}


def test_wrong_number_of_predictions_is_an_error(makarov: list[Sample]) -> None:
    class Broken:
        def predict(self, images):
            return ["x"]

    with pytest.raises(ValueError, match="1 predictions for 2"):
        evaluate(Broken(), makarov, batch_size=2)


def test_coverage_curve_ranks_by_confidence() -> None:
    records = [
        Record("a", "g", "abcd", "abcd", confidence=0.9),
        Record("b", "g", "abcd", "abXd", confidence=0.5),
        Record("c", "g", "abcd", "XXXX", confidence=0.1),
    ]
    curve = coverage_curve(records, coverages=(0.34, 1.0))
    assert [(p["lines"], p["cer"]) for p in curve] == [(1, 0.0), (3, 5 / 12)]
    assert "coverage" in summarize(records)


def test_cli_eval_and_compare(roots: tuple[Path, Path], tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("diploma.data.DATA_ROOT", roots[0])
    monkeypatch.setattr("diploma.data.SPLITS_ROOT", roots[1])
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "class Echo:\n"
        "    def __init__(self, text):\n"
        "        self.text = text\n"
        "    def predict(self, images):\n"
        "        return [self.text] * len(images)\n",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    main([
        "eval", f"{model_file}:Echo", "-a", "text=сторонамъ",
        "-d", "makarov:test", "-d", "peter:test", "--name", "echo", "--runs", str(runs),
    ])  # fmt: skip
    assert (runs / "echo" / "makarov_test" / "metrics.json").exists()
    main(["compare", "--runs", str(runs), "-d", "makarov:test"])
    table = capsys.readouterr().out.splitlines()[-1]
    assert table.startswith("| echo") and "| 0.00 " in table


def test_cli_sample_is_seeded_and_recorded(roots: tuple[Path, Path], tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("diploma.data.DATA_ROOT", roots[0])
    monkeypatch.setattr("diploma.data.SPLITS_ROOT", roots[1])
    model_file = tmp_path / "model.py"
    model_file.write_text("class Empty:\n    def predict(self, images):\n        return [''] * len(images)\n", encoding="utf-8")

    def run(name: str, seed: int) -> list[str]:
        main(["eval", f"{model_file}:Empty", "-d", "makarov", "--sample", "2", "--seed", str(seed),
              "--name", name, "--runs", str(tmp_path / "runs")])  # fmt: skip
        out = tmp_path / "runs" / name / "makarov"
        report = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert (report["lines"], report["sample"], report["seed"]) == (2, 2, seed)
        assert report["environment"]["python"]
        return [json.loads(l)["id"] for l in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]

    assert run("a", seed=0) == run("b", seed=0)
