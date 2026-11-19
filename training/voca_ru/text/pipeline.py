"""Raw sentence -> all fields one JSONL row needs."""

from .normalize import normalize
from .g2p import phonemize
from .stress import Stressor
from .symbols_ru import is_phone, strip_stress, symbol_id

_DEFAULT_STRESSOR = None


def default_stressor():
    global _DEFAULT_STRESSOR
    if _DEFAULT_STRESSOR is None:
        _DEFAULT_STRESSOR = Stressor().load()
    return _DEFAULT_STRESSOR


def analyze(text, stressor=None, oov_words=None):
    """``text`` -> dict with norm_text, stressed_text, phonemes, stress, tokens, oov.

    ``norm_text`` follows the manifest contract (§5): it carries the stress
    marks. ``plain_text`` is the same string without them.
    """
    plain = normalize(text)
    stressor = stressor if stressor is not None else default_stressor()
    stressed, oov = stressor.stressed(plain)
    symbols, levels = phonemize(stressed)
    return {
        "plain_text": plain,
        "stressed_text": stressed,
        "norm_text": stressed,
        "phonemes": symbols,
        "stress": levels,
        "tokens": [symbol_id(symbol) for symbol in symbols],
        "oov": bool(oov),
        "phone_count": sum(1 for symbol in symbols if is_phone(symbol)),
    }
