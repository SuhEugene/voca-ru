"""eSpeak NG Russian phonemes, exactly as upstream `tiny_voca.text` parses them.

Only the parts the alignment spike needs: run eSpeak on one text piece, split on
the separator, strip stress marks into a parallel 0/1/2 feature.

Pinned to espeak-ng 1.52.0 (Homebrew). The aligner labels were made with the
2021 eSpeak label set, so the mapping from what today's binary prints to the
model vocabulary lives in `mapping.py`, not here.
"""

import re
import subprocess
import unicodedata

SEPARATOR = "_"
VOICE = "ru"
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}
# eSpeak marks a language switch inline, e.g. ``(en)_t_ˈɛ_s_t_(ru)``. Upstream
# strips the tags themselves; the foreign phones that follow stay as artefacts.
LANGUAGE_TAG = re.compile(r"\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)")


def parse_espeak(output):
    """``output`` -> list of words, each a list of ``(phone, stress)``.

    eSpeak prints an empty unit for some boundaries (``s__rʲ``), so empty
    tokens are dropped.
    """
    output = LANGUAGE_TAG.sub("", output)
    words = []
    for word in output.split():
        units = []
        for unit in word.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            unit = "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()")
            if unit:
                units.append((unit, stress))
        if units:
            words.append(units)
    return words


def phonemize(text):
    """One text piece -> list of words of ``(phone, stress)``."""
    result = subprocess.run(
        ["espeak-ng", "-q", "--ipa", "--sep=" + SEPARATOR, "-v", VOICE, "--stdin"],
        input=text.rstrip("\n") + "\n", text=True, capture_output=True, check=True, timeout=30)
    return parse_espeak(result.stdout)


def phonemize_sentences(sentences):
    """Phonemize a list of sentences, one eSpeak call each.

    Not batched: eSpeak wraps its output at a fixed column, so joining
    sentences on stdin splits words across lines and loses tokens.
    """
    return [phonemize(text) for text in sentences]


def inventory(words):
    """Counter of phone -> occurrences over parsed output."""
    from collections import Counter
    phones = Counter(phone for word in words for phone, _ in word)
    return phones
