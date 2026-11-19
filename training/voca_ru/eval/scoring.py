"""Text normalisation used for scoring only (not for synthesis).

The contract (§5) puts the single text normaliser in ``training/voca_ru/text``
(F3).  ``score_text`` imports it and adds only the comparison cleanup that must
not leak into synthesis: fold ё→е, drop stress marks, punctuation and case.
If the F3 package is absent the shim below runs instead, and every report says
so through ``implementation``.
"""

from __future__ import annotations

import re
import unicodedata

try:  # F3 frontend
    from voca_ru.text import normalize as _f3_normalize

    implementation = "voca_ru.text.normalize + eval cleanup"
except ImportError:
    _f3_normalize = None
    implementation = "eval.shim (TODO F3)"

_ONES = (
    "ноль один два три четыре пять шесть семь восемь девять десять "
    "одиннадцать двенадцать тринадцать четырнадцать пятнадцать "
    "шестнадцать семнадцать восемнадцать девятнадцать"
).split()
_TENS = ("", "", "двадцать", "тридцать", "сорок", "пятьдесят",
         "шестьдесят", "семьдесят", "восемьдесят", "девяносто")
_HUNDREDS = ("", "сто", "двести", "триста", "четыреста", "пятьсот",
             "шестьсот", "семьсот", "восемьсот", "девятьсот")
_SCALE = (("тысяча", "тысячи", "тысяч"), ("миллион", "миллиона", "миллионов"))


def _plural(n: int, forms: tuple[str, str, str]) -> str:
    n = abs(n) % 100
    if 11 <= n <= 19:
        return forms[2]
    n %= 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def _triple(n: int, feminine: bool = False) -> list[str]:
    words: list[str] = []
    if n >= 100:
        words.append(_HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        words.append(_TENS[n // 10])
        n %= 10
    if n:
        if feminine and n == 1:
            words.append("одна")
        elif feminine and n == 2:
            words.append("две")
        else:
            words.append(_ONES[n])
    return [w for w in words if w]


def num2words_ru(n: int) -> str:
    """Cardinal integer 0..999 999 999, no case agreement (see report)."""
    if n < 0 or n > 999_999_999:
        return str(n)
    if n == 0:
        return "ноль"
    out: list[str] = []
    millions, rest = divmod(n, 1_000_000)
    thousands, units = divmod(rest, 1_000)
    if millions:
        out += _triple(millions)
        out.append(_plural(millions, _SCALE[1]))
    if thousands:
        out += _triple(thousands, feminine=True)
        out.append(_plural(thousands, _SCALE[0]))
    if units:
        out += _triple(units)
    return " ".join(out)


def _shim(text: str) -> str:
    # Scoring folds ё→е: ASR models almost always write е. Synthesis still
    # keeps ё (contract §5); this fold exists only in the comparison path.
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"[\u0300-\u036f]", "", text)  # combining stress marks
    text = re.sub(r"\d+", lambda m: num2words_ru(int(m.group())), text)
    text = text.lower()
    text = re.sub(r"[^а-яa-z]+", " ", text)
    return " ".join(text.split())


def _cleanup(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"[\u0300-\u036f]", "", text)
    text = text.lower()
    text = re.sub(r"[^а-яa-z]+", " ", text)
    return " ".join(text.split())


def score_text(text: str) -> str:
    """Normalise one side of a comparison; symmetric and idempotent."""
    if _f3_normalize is not None:
        return _cleanup(_f3_normalize(text))
    return _shim(text)
