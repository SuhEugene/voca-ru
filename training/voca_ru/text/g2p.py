"""eSpeak NG Russian G2P through the dylib shipped by ``espeakng-loader``.

The Rust frontend will not ship eSpeak (GPL); this module exists for teacher
annotation and training labels only. eSpeak 1.52.0 is pinned — 1.51 emits
different Russian labels (docs/aligner-mapping.md §1).

eSpeak ignores combining acute, so stress from RUAccent is transferred onto the
eSpeak vowel by ordinal: the n-th vowel letter of the word maps to the n-th
vowel phone. When those counts disagree the eSpeak mark is kept and the word is
reported as a stress-transfer miss.
"""

import ctypes
import re

from .symbols_ru import (
    PHONE_MAP, SP, BOS, EOS, UNK, is_phone, is_vowel, punctuation,
    PUNCTUATION_SET, STRESS, strip_stress,
)

VOWEL_LETTERS = "аеёиоуыэюя"
WORD_RE = re.compile(r"[а-яё\u0301]+(?:-[а-яё\u0301]+)*")
LANGUAGE_TAG = re.compile(r"\((?:[a-z]{2,3})\)")
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}

CONSONANT_CHARS = set("bdfɡjklmnpqrstvxzʃʒɕʑɭŋhθðɹɻw")
VOWEL_CHARS = set("ʌaɑɪiɛeəouyɵɔʊɐæøœɜ")


class EspeakRu:
    """One process-wide handle to libespeak-ng with the Russian voice."""

    def __init__(self, library_path=None, data_path=None):
        import espeakng_loader

        self.lib = ctypes.CDLL(library_path or espeakng_loader.get_library_path())
        data = data_path or espeakng_loader.get_data_path()
        self.lib.espeak_Initialize.restype = ctypes.c_int
        self.lib.espeak_Initialize.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        rate = self.lib.espeak_Initialize(2, 0, data.encode(), 0)  # synchronous
        if rate <= 0:
            raise RuntimeError("espeak_Initialize failed")
        if self.lib.espeak_SetVoiceByName(b"ru") != 0:
            raise RuntimeError("espeak SetVoiceByName(ru) failed")
        self.lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        self.lib.espeak_TextToPhonemes.argtypes = [
            ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
        self.lib.espeak_Info.restype = ctypes.c_char_p
        info = self.lib.espeak_Info(None)
        self.version = info.decode(errors="replace") if info else "unknown"

    def raw(self, text):
        """IPA string for one text piece (words separated by spaces)."""
        pointer = ctypes.c_char_p(text.encode("utf-8"))
        chunks = []
        while pointer:
            out = self.lib.espeak_TextToPhonemes(ctypes.byref(pointer), 1, 2)
            if not out:
                break
            chunks.append(out.decode("utf-8", errors="replace"))
        return " ".join(chunks)


_ESPEAK = None


def espeak():
    global _ESPEAK
    if _ESPEAK is None:
        _ESPEAK = EspeakRu()
    return _ESPEAK


def split_units(ipa):
    """IPA string -> (units, dropped) with ʲ/ː kept on the phone.

    eSpeak 1.52 emits a small alphabet. Affricates are two code points
    (``tʃ``, ``dʒ``, ``ts``); palatalisation is U+02B2 attached to a
    consonant. ``^`` and ``"`` are build artefacts and are dropped here.
    """
    units = []
    dropped = []
    i = 0
    while i < len(ipa):
        char = ipa[i]
        if char in STRESS_MARKS:
            units.append((char, 0))
            i += 1
            continue
        if char in "^\"":
            i += 1
            continue
        unit = char
        if char == "t" and i + 1 < len(ipa) and ipa[i + 1] in "ʃs":
            unit = ipa[i:i + 2]
            i += 1
        elif char == "d" and i + 1 < len(ipa) and ipa[i + 1] == "ʒ":
            unit = "dʒ"
            i += 1
        elif char == "u" and i + 1 < len(ipa) and ipa[i + 1] == '"':
            unit = 'u"'
            i += 1
        elif char == "ɪ" and i + 1 < len(ipa) and ipa[i + 1] == "^":
            unit = "ɪ^"
            i += 1
        i += 1
        while i < len(ipa) and ipa[i] in "ʲː":
            unit += ipa[i]
            i += 1
        if unit in PHONE_MAP:
            units.append((unit, 0))
        else:
            dropped.append(unit)
    return units, dropped


def _target_vowel_index(stressed_word):
    plain = strip_stress(stressed_word)
    index = None
    for i, char in enumerate(stressed_word):
        if char == STRESS:
            index = sum(1 for c in plain[:i] if c.lower() in VOWEL_LETTERS) - 1
            break
    return plain, index


def parse_word(ipa_word, target_index=None, drop=None):
    """One eSpeak word -> [(symbol, stress)] with RUAccent's stress applied."""
    units, dropped = split_units(ipa_word)
    if drop is not None:
        drop.update(dropped)
    phones = []
    pending = 0
    for unit, _ in units:
        if unit in STRESS_MARKS:
            pending = STRESS_MARKS[unit]
            continue
        symbol = PHONE_MAP[unit]
        phones.append([symbol, pending if is_vowel(symbol) else 0])
        if is_vowel(symbol):
            pending = 0
    if target_index is not None:
        vowel_slots = [i for i, (symbol, _) in enumerate(phones) if is_vowel(symbol)]
        if 0 <= target_index < len(vowel_slots):
            for slot in phones:
                slot[1] = 0
            phones[vowel_slots[target_index]][1] = 1
    return [(symbol, stress) for symbol, stress in phones]


class G2P:
    """Word-level cache around one espeak handle."""

    def __init__(self, engine=None):
        self.engine = engine or espeak()
        self._cache = {}
        self.dropped = set()
        self.transfer_misses = 0

    def word(self, stressed_word):
        plain, target = _target_vowel_index(stressed_word)
        key = (plain, target)
        if key in self._cache:
            return self._cache[key]
        ipa = self.engine.raw(plain)
        units, dropped = split_units(ipa)
        self.dropped.update(dropped)
        words = ipa.split()
        if len(words) > 1:
            # eSpeak split a hyphenated word; treat each part separately.
            phones = []
            for part in words:
                if phones:
                    phones.append((SP, 0))
                phones.extend(parse_word(part, None, self.dropped))
            if target is not None:
                slots = [i for i, (s, _) in enumerate(phones) if is_vowel(s)]
                if target < len(slots):
                    for i, (s, _) in enumerate(phones):
                        if is_vowel(s):
                            phones[i] = (s, 0)
                    s = phones[slots[target]][0]
                    phones[slots[target]] = (s, 1)
                else:
                    self.transfer_misses += 1
        else:
            phones = parse_word(ipa, target, self.dropped) if ipa else []
            if target is not None and ipa:
                slots = sum(1 for symbol, _ in phones if is_vowel(symbol))
                if target >= slots:
                    self.transfer_misses += 1
        self._cache[key] = phones
        return phones

    def pieces(self, stressed_text):
        """Text -> ('punct', mark) / ('word', phones) pieces."""
        out = []
        for match in re.finditer(r"[а-яё\u0301]+(?:-[а-яё\u0301]+)*|[^\sа-яё\u0301]", stressed_text):
            token = match.group(0)
            if WORD_RE.fullmatch(token):
                out.append(("word", self.word(token)))
            else:
                mark = punctuation(token)
                if mark:
                    out.append(("punct", mark))
        return out

    def tokenize(self, stressed_text):
        """Mirror Rust ``symbols::tokenize``: BOS, <sp>, punctuation, EOS."""
        symbols = [BOS]
        stress = [0]
        has_phone = False
        for kind, value in self.pieces(stressed_text):
            if kind == "punct":
                if symbols[-1] != value:
                    symbols.append(value)
                    stress.append(0)
            else:
                if is_phone(symbols[-1]) or symbols[-1] == UNK:
                    symbols.append(SP)
                    stress.append(0)
                for phone, level in value:
                    if phone == SP:
                        if symbols[-1] != SP:
                            symbols.append(SP)
                            stress.append(0)
                        continue
                    if is_phone(phone) or phone == UNK:
                        has_phone = True
                    symbols.append(phone)
                    stress.append(level)
        symbols.append(EOS)
        stress.append(0)
        if not has_phone:
            raise ValueError("the text has nothing to pronounce")
        return symbols, stress


_G2P = None


def phonemize(stressed_text):
    """Stressed text -> (symbols, stress levels). Cached per word."""
    global _G2P
    if _G2P is None:
        _G2P = G2P()
    return _G2P.tokenize(stressed_text)
