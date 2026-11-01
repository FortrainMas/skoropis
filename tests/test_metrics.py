import pytest

from diploma.metrics import Counts, Tally, align, bootstrap_ci, total


@pytest.mark.parametrize(
    ("ref", "hyp", "sub", "dele", "ins"),
    [
        ("kitten", "sitting", 2, 0, 1),
        ("abc", "abc", 0, 0, 0),
        ("abc", "", 0, 3, 0),
        ("", "abc", 0, 0, 3),
        ("судъ", "суд", 0, 1, 0),
    ],
)
def test_align_counts(ref: str, hyp: str, sub: int, dele: int, ins: int) -> None:
    counts = Counts.of(align(ref, hyp))
    assert (counts.substitutions, counts.deletions, counts.insertions) == (sub, dele, ins)
    assert counts.ref_len == len(ref)


def test_alignment_reconstructs_both_strings() -> None:
    edits = align("Макаровскаго", "Макаровского")
    assert "".join(e.ref or "" for e in edits) == "Макаровскаго"
    assert "".join(e.hyp or "" for e in edits) == "Макаровского"
    assert [(e.op, e.ref, e.hyp) for e in edits if e.op != "="] == [("S", "а", "о")]


def test_empty_reference_rate() -> None:
    assert Counts.of(align("", "")).rate == 0.0
    assert Counts.of(align("", "x")).rate == 1.0


def test_tally_is_corpus_level() -> None:
    # 1 error in 10 chars + 1 error in 2 chars: corpus CER 2/12, not the mean of line CERs.
    tally = total([Tally.of_line("abcdefghij", "abcdefghiX"), Tally.of_line("ab", "a")])
    assert tally.cer == pytest.approx(2 / 12)
    assert tally.wer == pytest.approx(1.0)
    assert tally.line_accuracy == 0.0


def test_wer_counts_words() -> None:
    tally = Tally.of_line("по иску крестьянина", "по иску крестьянин")
    assert tally.wer == pytest.approx(1 / 3)
    assert tally.cer == pytest.approx(1 / 19)


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic() -> None:
    by_group = {
        "p1": Tally.of_line("abcdef", "abcdef"),
        "p2": Tally.of_line("abcdef", "abXdef"),
        "p3": Tally.of_line("abcdef", "aXXdef"),
        "p4": Tally.of_line("abcdef", "abcdeX"),
    }
    ci = bootstrap_ci(by_group, resamples=500, seed=1)
    low, high = ci["cer"]
    assert low <= total(by_group.values()).cer <= high
    assert ci == bootstrap_ci(by_group, resamples=500, seed=1)


def test_bootstrap_of_identical_groups_has_zero_width() -> None:
    by_group = {f"p{i}": Tally.of_line("abcd", "abXd") for i in range(5)}
    assert bootstrap_ci(by_group, resamples=100)["cer"] == (0.25, 0.25)
