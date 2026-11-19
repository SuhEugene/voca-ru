"""The 76-symbol Russian inventory (contract: docs/contracts.md §4).

Layout is append-only and shared with the Rust frontend (F7):

    0..4    5 specials        <pad> <bos> <eos> <sp> <unk>
    5..27    23 punctuation
    28..63   36 consonants
    64..75   12 vowels

Ids 0..16 are identical to the upstream English table, so the specials and the
first twelve punctuation marks never move. Russian phones are appended after
the punctuation block: ``FIRST_PHONE = 28``, ``FIRST_VOWEL = 64``.

Every symbol is one token. Palatalisation is part of the consonant symbol
(``tʲ``), not a separate token: eSpeak and the F6 aligner both emit it that
way, and 36 consonants + 12 vowels fit the 48-phone budget.
"""

SPECIALS = ["<pad>", "<bos>", "<eos>", "<sp>", "<unk>"]

# Upstream ids 5..16, then the 11 Russian additions, in this exact order.
PUNCTUATION = [
    ",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…",
    "«", "»", "-", "–", "№", "%", "/", "[", "]", "{", "}",
]

# 36 consonants, hard/soft pairs adjacent.
CONSONANTS = [
    "b", "bʲ", "v", "vʲ", "ɡ", "ɡʲ", "d", "dʲ", "ʒ", "z", "ʑ", "j",
    "k", "kʲ", "ɭ", "ɭʲ", "m", "mʲ", "n", "nʲ", "p", "pʲ", "r", "rʲ",
    "s", "sʲ", "t", "tʲ", "ts", "tʃʲ", "f", "fʲ", "x", "ʃ", "ɕ", "dʒ",
]

# 12 vowels: unreduced and reduced allophones stay distinct (a real cue).
VOWELS = ["ʌ", "a", "ɑ", "ɪ", "i", "ɛ", "e", "ə", "o", "u", "y", "ɵ"]

SYMBOLS = SPECIALS + PUNCTUATION + CONSONANTS + VOWELS
assert len(SYMBOLS) == 76 and len(set(SYMBOLS)) == 76

FIRST_PUNCTUATION = 5
FIRST_PHONE = 28
FIRST_VOWEL = 64
PAD, BOS, EOS, SP, UNK = SPECIALS
PAD_ID, BOS_ID, EOS_ID, SP_ID, UNK_ID = range(5)

SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
PUNCTUATION_SET = set(PUNCTUATION)
PHONE_SET = set(CONSONANTS) | set(VOWELS)
VOWEL_SET = set(VOWELS)


def symbol_id(symbol):
    return SYMBOL_TO_ID[symbol]


def is_phone(symbol):
    return symbol in PHONE_SET


def is_vowel(symbol):
    return symbol in VOWEL_SET


def punctuation(mark):
    """Inventory copy of a punctuation mark, or None."""
    return mark if mark in PUNCTUATION_SET else None


# eSpeak-ru unit -> training symbol. Anything not listed is dropped after the
# cleanup table below. Sources: docs/aligner-mapping.md (F6), plus the units
# measured over `data/scripts-ru` with espeak-ng 1.52.
PHONE_MAP = {
    # identity (all 48 inventory phones that eSpeak 1.52 emits)
    **{p: p for p in CONSONANTS},
    **{p: p for p in VOWELS},
    # canonical merges decided by F6 and kept by F3
    "u\"": "u",          # /u/ after a soft consonant (ю)
    "ɪ^": "i",           # /и/ allophone after a hard consonant
    "tʃ": "tʃʲ",         # foreign /tʃ/; Russian ч is always soft
    "dʒʲ": "dʒ",         # no soft dʒ in the inventory
    "ɔ": "o",
    "iː": "i",
    "eɪ": "e",
    "aɪ": "i",
    "əʊ": "o",
    "ɹ": "r",
    "ɻ": "r",
    "ɑː": "ɑ",
    "θ": "t",
    "ð": "d",
    "l": "ɭ",
    "əl": "ɭ",
    "ɜː": "e",
    "uː": "u",
    "w": "v",
    "ø": "o",
    "øː": "o",
    "œ": "o",
    "ɒ": "ɑ",
    "h": "x",
    "eə": "e",
    "n̩": "n",
    "l̩": "ɭ",
    "m̩": "m",
    "ɾ": "r",
    "ŋ": "n",
    "ɐ": "ʌ",
    "zʲ": "ʑ",          # F6 canonical; eSpeak 1.52 uses ʑ for зь anyway
    "ɕː": "ɕ",          # long /ɕ/ (щ) has no length variant
    # eSpeak internal names seen in other builds (docs/aligner-mapping.md §4)
    "a#": "a",
    "u#": "u",
    "I#": "ɪ",
    "a#ː": "a",
    "@-": "ə",
    "8": "ɵ",
}

# Symbol -> label in facebook/wav2vec2-lv-60-espeak-cv-ft vocab.json, per
# docs/aligner-mapping.md. F6 owns this table; F3 only repeats it so the
# annotation step cannot drift from the inventory.
ALIGNER_LABEL = {
    **{p: p for p in CONSONANTS},
    **{p: p for p in VOWELS},
    # F6 canonical decisions win over the identity default.
    "a": "ʌ", "ɪ": "i", "ɛ": "e", "ə": "ʌ", "ɵ": "o",
}
ALIGNER_LABEL.update({
    "ʌ": "ʌ", "ɑ": "ɑ", "i": "i", "e": "e", "o": "o", "u": "u",
    "y": "y", "ʑ": "ʑ", "ɕ": "ɕ", "dʒ": "dʒ", "tʃʲ": "tʃʲ",
    "ts": "ts", "ɭʲ": "ɭʲ", "ɡ": "ɡ",
})

# Combining acute is the only stress notation the manifest stores.
STRESS = "\u0301"


def strip_stress(text):
    return text.replace(STRESS, "")


def mark_vowel(text, index):
    """Put U+0301 after the vowel at 0-based ``index`` among vowels."""
    vowels = "аеёиоуыэюя"
    seen = -1
    for pos, char in enumerate(text):
        if char.lower() in vowels:
            seen += 1
            if seen == index:
                return text[: pos + 1] + STRESS + text[pos + 1:]
    return text


def vowel_positions(text):
    """Indices of vowel letters in ``text`` (ё counts as its own letter)."""
    vowels = "аеёиоуыэюя"
    return [i for i, char in enumerate(text) if char.lower() in vowels]
