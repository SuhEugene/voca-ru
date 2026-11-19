"""Russian front end adapter for the ported tiny-voca stack.

The token inventory and the text pipeline live in ``voca_ru.text`` (F3); this
module only gives the upstream training code the names it expects
(``Phonemized``, ``phonemize``, ``frontend_fingerprint`` …), so tokens cannot
drift between annotation, training and the Rust frontend.
"""

from dataclasses import dataclass
import hashlib
import json

from voca_ru.text import normalize, phonemize as _phonemize, stress_text
from voca_ru.text.symbols_ru import (
    ALIGNER_LABEL, BOS, CONSONANTS, EOS, PAD, PHONE_SET, PUNCTUATION,
    SPECIALS, SP, SYMBOL_TO_ID, SYMBOLS, UNK, VOWEL_SET, VOWELS,
)

FRONTEND_VERSION = 1
FRONTEND_NAME = "voca_ru.text"

PHONES = tuple(CONSONANTS) + tuple(VOWELS)
PHONE_IDS = frozenset(SYMBOL_TO_ID[p] for p in PHONES)
VOICELESS = frozenset((
    "p", "pʲ", "t", "tʲ", "k", "kʲ", "ts", "tʃʲ", "f", "fʲ",
    "s", "sʲ", "ʃ", "ɕ", "x",
))


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple

    def __len__(self):
        return len(self.ids)


def frontend_identity():
    return {"frontend_version": FRONTEND_VERSION, "phonemizer": FRONTEND_NAME,
            "symbols": list(SYMBOLS)}


def frontend_fingerprint():
    return hashlib.sha256(json.dumps(frontend_identity(), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def frontend_runtime():
    """What produced the phonemes: recorded so stale annotations are detected."""
    try:
        from voca_ru.text.g2p import espeak
        version = espeak().version
    except Exception:
        version = "unavailable"
    return {"phonemizer": FRONTEND_NAME, "espeak_version": version}


def phonemize(text, stress=True):
    """Raw or stressed Russian text -> :class:`Phonemized`.

    Text without combining acute is normalised and stressed first; text that
    already carries the marks (manifest ``norm_text``) is used as given.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    if "\u0301" in text:
        prepared = text
    else:
        prepared = normalize(text)
        if stress:
            prepared, _ = stress_text(prepared)
    symbols, stress_levels = _phonemize(prepared)
    return Phonemized(tuple(symbols), tuple(SYMBOL_TO_ID[s] for s in symbols),
                      tuple(stress_levels), ())


def is_phone(token_ids):
    return [int(i) in PHONE_IDS for i in token_ids]
