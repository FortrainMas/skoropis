import pytest

from diploma.text import Norm, canonical, normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Рѣшенiе  объявлено\n", "Рѣшенiе объявлено"),
        ("Рѣшеніе", "Рѣшенiе"),  # Cyrillic і -> the datasets' Latin i
        ("cторонамъ", "сторонамъ"),  # Latin c inside a Cyrillic word
        ("piter sire", "piter sire"),  # genuinely Latin words stay Latin
        ("24–i–18 — 40", "24-i-18 - 40"),
        ("й", "й"),  # NFC
    ],
)
def test_canonical(raw: str, expected: str) -> None:
    assert canonical(raw) == expected


def test_canonical_is_idempotent() -> None:
    text = canonical("Отвѣтчикъ  Хорошевъ cказалъ: «да» – i")
    assert canonical(text) == text


@pytest.mark.parametrize(
    ("norm", "expected"),
    [
        (Norm.EXACT, "Рѣшенiе объявлено 5/II, судъ: объ"),
        (Norm.CASELESS, "рѣшенiе объявлено 5/ii, судъ: объ"),
        (Norm.LETTERS, "рѣшенiе объявлено 5 ii судъ объ"),
        (Norm.MODERN, "решение объявлено 5 ии суд об"),
    ],
)
def test_normalize(norm: Norm, expected: str) -> None:
    assert normalize("Рѣшенiе объявлено 5/II, судъ: объ", norm) == expected


def test_modern_keeps_inner_hard_sign() -> None:
    assert normalize("объявленъ", Norm.MODERN) == "объявлен"
