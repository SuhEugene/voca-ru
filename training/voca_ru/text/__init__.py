"""Russian text frontend: normalisation -> stress -> phonemes -> token ids."""

from .normalize import normalize
from .stress import Stressor, stress_text
from .g2p import phonemize
from .symbols_ru import SYMBOLS, symbol_id, is_vowel, is_phone

__all__ = [
    "analyze",
    "normalize",
    "stress_text",
    "phonemize",
    "Stressor",
    "SYMBOLS",
    "symbol_id",
    "is_vowel",
    "is_phone",
]


def analyze(text, stressor=None, oov_words=None):
    """Raw sentence -> the fields the CLI writes for one JSONL row."""
    from .pipeline import analyze as _analyze

    return _analyze(text, stressor=stressor, oov_words=oov_words)
