"""Transcription canonicalization and evaluation-time normalization.

Both datasets write the pre-reform letter ``і`` as Latin ``i``. We keep that
convention and fold every other encoding variant of a visually identical
character into one code point, so the alphabet is not split by typing noise
(a Latin ``c`` inside a Cyrillic word, an en dash instead of a hyphen, ...).
``canonical`` is applied to labels and model outputs alike.

``normalize`` produces the coarser views used only for reporting metrics.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

_FOLD = str.maketrans({"і": "i", "І": "I", **{dash: "-" for dash in "‐‑‒–—―"}})
# Latin look-alikes of Cyrillic letters. 'i' is absent on purpose: it *is* the dataset's і.
_LATIN_TO_CYRILLIC = str.maketrans("aceopxyABCEHKMOPTX", "асеорхуАВСЕНКМОРТХ")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

_PUNCTUATION = re.compile(r"[^\w\s]|_")
_MODERN = str.maketrans({"ѣ": "е", "i": "и", "ѳ": "ф", "ѵ": "и"})
_FINAL_HARD_SIGN = re.compile(r"ъ\b")


class Norm(StrEnum):
    """How strictly a hypothesis is compared to the reference."""

    EXACT = "exact"  # canonical form, every character counts
    CASELESS = "caseless"  # ignore letter case
    LETTERS = "letters"  # caseless, punctuation and special marks dropped
    MODERN = "modern"  # letters + pre-reform letters mapped to modern ones, final ъ dropped


def canonical(text: str) -> str:
    """Return the single representation used for labels and predictions."""
    text = unicodedata.normalize("NFC", text).translate(_FOLD)
    return " ".join(_fold_mixed_script(word) for word in text.split())


def _fold_mixed_script(word: str) -> str:
    return word.translate(_LATIN_TO_CYRILLIC) if _CYRILLIC.search(word) else word


def normalize(text: str, norm: Norm) -> str:
    """Map a canonical string to the comparison view ``norm``."""
    if norm is Norm.EXACT:
        return text
    text = text.lower()
    if norm is Norm.CASELESS:
        return text
    text = " ".join(_PUNCTUATION.sub(" ", text).split())
    if norm is Norm.LETTERS:
        return text
    return " ".join(_FINAL_HARD_SIGN.sub("", text.translate(_MODERN)).split())
