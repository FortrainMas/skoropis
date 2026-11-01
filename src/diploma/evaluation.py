"""Model-agnostic evaluation: run a recognizer over samples, score, persist.

A model only has to satisfy :class:`Recognizer`: take a batch of RGB PIL images
of text lines and return one string (or :class:`Prediction` with a confidence)
per image. Resizing, normalization and decoding are the model's business, so
CTC models, seq2seq transformers and VLM wrappers are evaluated identically.

Each evaluated dataset produces a directory with

* ``predictions.jsonl`` -- one line per sample: reference, hypothesis, CER;
* ``metrics.json`` -- CER / WER / line accuracy under every :class:`Norm`, with
  page-bootstrap 95% CIs, top character confusions and, when the model reports
  confidences, error rates of its most confident predictions (the quantity
  that matters for pseudo-labeling).
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from diploma.data import Sample
from diploma.metrics import Tally, align, bootstrap_ci, total
from diploma.text import Norm, canonical, normalize

COVERAGES = (0.1, 0.2, 0.5, 1.0)


@dataclass(frozen=True)
class Prediction:
    text: str
    confidence: float | None = None


class Recognizer(Protocol):
    def predict(self, images: Sequence[Image.Image]) -> Sequence[str | Prediction]: ...


@dataclass(frozen=True)
class Record:
    id: str
    group: str
    ref: str
    hyp: str
    confidence: float | None = None


def evaluate(
    recognizer: Recognizer,
    samples: Sequence[Sample],
    out_dir: Path | None = None,
    *,
    batch_size: int = 16,
    progress: Callable[[int, int], None] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict, score and (if ``out_dir`` is given) save; returns the metrics report."""
    records, seconds = predict(recognizer, samples, batch_size=batch_size, progress=progress)
    report = {
        **(meta or {}),
        **summarize(records),
        "seconds": seconds,
        "lines_per_second": len(records) / seconds if seconds else None,
    }
    if out_dir is not None:
        save(out_dir, records, report)
    return report


def predict(
    recognizer: Recognizer,
    samples: Sequence[Sample],
    *,
    batch_size: int = 16,
    progress: Callable[[int, int], None] | None = None,
    loaders: int = 4,
) -> tuple[list[Record], float]:
    """Run ``recognizer`` over ``samples``; returns records and pure inference time.

    ``loaders`` threads decode the next batch while the model works on the current
    one (Pillow releases the GIL while decoding).
    """
    records: list[Record] = []
    seconds = 0.0
    batches = [samples[start : start + batch_size] for start in range(0, len(samples), batch_size)]
    with ThreadPoolExecutor(max_workers=loaders) as pool:

        def load(k: int) -> list:
            return [pool.submit(_load_rgb, s.image) for s in batches[k]] if k < len(batches) else []

        pending = load(0)
        for k, batch in enumerate(batches):
            images = [future.result() for future in pending]
            pending = load(k + 1)
            tic = time.perf_counter()
            outputs = list(recognizer.predict(images))
            seconds += time.perf_counter() - tic
            if len(outputs) != len(batch):
                raise ValueError(f"recognizer returned {len(outputs)} predictions for {len(batch)} images")
            for sample, output in zip(batch, outputs):
                prediction = output if isinstance(output, Prediction) else Prediction(output)
                records.append(
                    Record(sample.id, sample.group, sample.text, canonical(prediction.text), prediction.confidence)
                )
            if progress is not None:
                progress(len(records), len(samples))
    return records, seconds


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def summarize(records: Sequence[Record], *, resamples: int = 1000, seed: int = 0) -> dict[str, Any]:
    report: dict[str, Any] = {
        "lines": len(records),
        "groups": len({r.group for r in records}),
        "metrics": {
            norm.value: _scores(records, norm, resamples=resamples, seed=seed) for norm in Norm
        },
        "confusions": top_confusions(records),
    }
    if records and all(r.confidence is not None for r in records):
        report["coverage"] = coverage_curve(records)
    return report


def _scores(records: Sequence[Record], norm: Norm, *, resamples: int, seed: int) -> dict[str, Any]:
    by_group: defaultdict[str, Tally] = defaultdict(Tally)
    for r in records:
        by_group[r.group] += Tally.of_line(normalize(r.ref, norm), normalize(r.hyp, norm))
    overall = total(by_group.values())
    return {
        "cer": overall.cer,
        "wer": overall.wer,
        "line_accuracy": overall.line_accuracy,
        "ci95": bootstrap_ci(by_group, resamples=resamples, seed=seed),
        "chars": asdict(overall.chars),
        "words": asdict(overall.words),
    }


def top_confusions(records: Sequence[Record], top: int = 30) -> list[dict[str, Any]]:
    """Most frequent character edits; an empty ``ref`` is an insertion, empty ``hyp`` a deletion."""
    counter: Counter[tuple[str, str]] = Counter()
    for r in records:
        counter.update((e.ref or "", e.hyp or "") for e in align(r.ref, r.hyp) if e.op != "=")
    return [{"ref": ref, "hyp": hyp, "count": n} for (ref, hyp), n in counter.most_common(top)]


def coverage_curve(records: Sequence[Record], coverages: Sequence[float] = COVERAGES) -> list[dict[str, Any]]:
    """Error rates of the ``coverage`` fraction of lines the model is most confident about."""
    ranked = sorted(records, key=lambda r: r.confidence, reverse=True)
    tallies = [Tally.of_line(r.ref, r.hyp) for r in ranked]
    curve = []
    for coverage in coverages:
        k = max(1, round(coverage * len(ranked)))
        kept = total(tallies[:k])
        curve.append(
            {
                "coverage": coverage,
                "lines": k,
                "min_confidence": ranked[k - 1].confidence,
                "cer": kept.cer,
                "line_accuracy": kept.line_accuracy,
            }
        )
    return curve


def save(out_dir: Path, records: Sequence[Record], report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            line = {**asdict(r), "cer": Tally.of_line(r.ref, r.hyp).cer}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    (out_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_reports(runs_dir: Path) -> list[dict[str, Any]]:
    """All ``metrics.json`` reports under ``runs_dir/<run>/<dataset>/``."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob("*/*/metrics.json"))]
