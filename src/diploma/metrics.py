"""Edit-distance metrics and their uncertainty.

Error rates are corpus-level (total edits / total reference length), as in
the Digital Peter competition and most HTR papers; the paper's SER is the
same quantity computed over symbols. Confidence intervals come from a
bootstrap over *groups* (pages), because lines of one page are not
independent samples.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

Op = Literal["=", "S", "D", "I"]
T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class Edit(Generic[T]):
    op: Op
    ref: T | None
    hyp: T | None


def align(ref: Sequence[T], hyp: Sequence[T]) -> list[Edit[T]]:
    """Minimal Levenshtein alignment turning ``ref`` into ``hyp``."""
    n, m = len(ref), len(hyp)
    dist = [list(range(m + 1))]
    for i in range(1, n + 1):
        prev, row, r = dist[-1], [i] * (m + 1), ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (r != hyp[j - 1]))
        dist.append(row)

    edits: list[Edit[T]] = []
    i, j = n, m
    while i or j:
        if i and j and dist[i][j] == dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j = i - 1, j - 1
            edits.append(Edit("=" if ref[i] == hyp[j] else "S", ref[i], hyp[j]))
        elif i and dist[i][j] == dist[i - 1][j] + 1:
            i -= 1
            edits.append(Edit("D", ref[i], None))
        else:
            j -= 1
            edits.append(Edit("I", None, hyp[j]))
    edits.reverse()
    return edits


def rate(errors: int, ref_len: int) -> float:
    """Error rate; an empty reference scores 0 if matched exactly and 1 otherwise."""
    if ref_len == 0:
        return float(errors > 0)
    return errors / ref_len


@dataclass(frozen=True)
class Counts:
    """Edit counts of one or many aligned (reference, hypothesis) pairs."""

    ref_len: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @classmethod
    def of(cls, edits: Sequence[Edit]) -> Counts:
        ops = [e.op for e in edits]
        return cls(
            ref_len=len(ops) - ops.count("I"),
            substitutions=ops.count("S"),
            deletions=ops.count("D"),
            insertions=ops.count("I"),
        )

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return rate(self.errors, self.ref_len)

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            self.ref_len + other.ref_len,
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
        )


@dataclass(frozen=True)
class Tally:
    """Everything needed to compute CER, WER and line accuracy for a set of lines."""

    chars: Counts = Counts()
    words: Counts = Counts()
    lines: int = 0
    exact: int = 0

    @classmethod
    def of_line(cls, ref: str, hyp: str) -> Tally:
        return cls(
            chars=Counts.of(align(ref, hyp)),
            words=Counts.of(align(ref.split(), hyp.split())),
            lines=1,
            exact=int(ref == hyp),
        )

    def __add__(self, other: Tally) -> Tally:
        return Tally(
            self.chars + other.chars,
            self.words + other.words,
            self.lines + other.lines,
            self.exact + other.exact,
        )

    @property
    def cer(self) -> float:
        return self.chars.rate

    @property
    def wer(self) -> float:
        return self.words.rate

    @property
    def line_accuracy(self) -> float:
        return self.exact / self.lines if self.lines else 0.0


def total(tallies: Iterable[Tally]) -> Tally:
    return sum(tallies, Tally())


def bootstrap_ci(
    by_group: Mapping[str, Tally],
    *,
    resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile CI of CER, WER and line accuracy, resampling whole groups."""
    if not by_group:
        raise ValueError("bootstrap needs at least one group")
    # Plain int tuples: summing dataclasses here is the slow path of a whole evaluation.
    groups = [
        (t.chars.errors, t.chars.ref_len, t.words.errors, t.words.ref_len, t.exact, t.lines)
        for t in by_group.values()
    ]
    rng = random.Random(seed)
    cer, wer, accuracy = [], [], []
    for _ in range(resamples):
        char_errors, chars, word_errors, words, exact, lines = map(sum, zip(*rng.choices(groups, k=len(groups))))
        cer.append(rate(char_errors, chars))
        wer.append(rate(word_errors, words))
        accuracy.append(exact / lines)

    def interval(values: list[float]) -> tuple[float, float]:
        values.sort()
        return values[int((1 - level) / 2 * (len(values) - 1))], values[int((1 + level) / 2 * (len(values) - 1))]

    return {"cer": interval(cer), "wer": interval(wer), "line_accuracy": interval(accuracy)}
