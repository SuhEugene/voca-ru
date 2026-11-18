"""eSpeak-ru phone -> aligner label map.

The aligner (`facebook/wav2vec2-lv-60-espeak-cv-ft`) was fine-tuned in 2021 on
eSpeak phoneme labels. Today's eSpeak 1.52 output differs, and Russian brings
phones the English inventory never used. Every difference is decided here, once,
and the same table is used for annotation and for measuring the aligner.

Two kinds of entries:

* `FALLBACK` — phone absent from `vocab.json`; mapped to the nearest label the
  model actually has. Nothing else can be aligned.
* `CANONICAL` — the phone *is* a label but a different label is the documented
  eSpeak-ru convention (`u"`, `ɪ^`, `zʲ`, `ɕː`). `calibrate.py` checks these
  choices against real audio and the report records the outcome.

`ARTEFACT` marks phones that only appear for foreign/Latin text; F3 decides
whether the corpus keeps them. They still need a label so a clip does not fail.

Run `python scripts/align_spike/calibrate.py` to re-measure every choice against
the test clips; it prints mean log-probability per candidate and the top-5
labels the model preferred.
"""

# Phones absent from vocab.json (2021 eSpeak label set) -> nearest label.
FALLBACK = {
    # Modern eSpeak-ru spellings the old model never saw.
    "zʲ": "ʑ",          # R5 probe: voiced palatal fricative, ʑ is the label
    "ɕː": "ɕ",          # long ɕ (щ); the model has no length mark for it
    "a#": "a",          # word-final allophone artefacts
    "u#": "u",
    "I#": "ɪ",
    "a#ː": "aː",
    "@-": "ə",
    "8": "o",           # eSpeak internal name for ё's vowel
    "ɨː": "ɨ",
    # Foreign phones from Latin text in otherwise Russian sentences.
    "ɹ": "r",
    "ɻ": "r",
    "θ": "t",
    "ð": "d",
    "ø": "o",
    "øː": "oː",
    "œ": "o",
    "əʊ": "o",
    "eɪ": "e",
    "aɪ": "i",
    "aɪə": "aɪ",
    "ɔ": "o",
    "ɑː": "ɑ",
    "iː": "i",
    "ɜː": "ɜ",
    "tʃ": "tʃʲ",
    "dʒʲ": "dʒ",
    "əl": "l",
    "w": "v",
    "n̩": "n",
    "l̩": "l",
    "m̩": "m",
    "ɾ": "r",
}

# Phones that are labels but a different label aligns better (measured:
# `.cache/align-spike/alignment-fleurs*.json`, numbers in the report).
CANONICAL = {
    "u\"": "u",         # eSpeak-ru /u/ after a soft consonant; u wins 1.49 vs -3.99
    "ɪ^": "ɪ",          # eSpeak-ru allophone of /i/; too rare to decide (<30 in 775 sentences)
    "ɵ": "o",           # ё: model has ɵ but picks o on 15 of 21 spans (-0.86 vs -6.60)
    "ə": "ʌ",           # reduced vowel: 24 of 32 spans pick ʌ (-0.96 vs -5.39)
    "ɪ": "i",           # и/е reduced: 296 of 503 spans pick i (-1.24 vs -2.24)
    "a": "ʌ",           # /a/ after a hard consonant: 269 of 327 pick ʌ (-0.85 vs -1.84)
    "ɛ": "e",           # э: e -1.95 vs ɛ -2.01, ɛ top-1 only 12/89
    # Foreign phones (Latin words) — kept alignable until F3 decides the text
    # policy. Scores for each pair are in the report.
    "eɪ": "e",
    "əʊ": "o",
    "aɪ": "i",
    "ɔ": "o",
    "ɑː": "ɑ",
    "iː": "i",
    "ɜː": "ɜ",
    "tʃ": "tʃʲ",
    "dʒʲ": "dʒ",
    "əl": "l",
    "w": "v",
    "ɹ": "r",
    "ɻ": "r",
    "θ": "t",
    "ð": "d",
    "ø": "o",
    "øː": "oː",
    "œ": "o",
    "ɪ^": "i",          # thin evidence (<30 occurrences in 775 sentences); follows ɪ→i
}

# One eSpeak phone that the 2021 label set wrote as two phones.
# Measured 2026-11-19 on 110 FLEURS clips: splitting `ja` into j+a and `ju` into
# j+u makes the phone probability worse (ja 0.119 -> 0.036, ju 0.065 -> 0.038)
# because the model emits blank across the glide either way. Kept empty; the
# code path stays so the experiment can be rerun, but the contract uses the
# single eSpeak label.
EXPAND = {
    # "ja": ("j", "a"),
    # "ju": ("j", "u"),
}

# Only reachable through foreign text. Kept for completeness, not for training.
ARTEFACT = frozenset(("ɹ", "ɻ", "θ", "ð", "ø", "øː", "œ", "əʊ", "aɪə", "n̩",
                      "l̩", "m̩", "ɾ"))

STRESS_LEVELS = (0, 1, 2)

# Vowels as eSpeak-ru emits them (plus the foreign ones above). Used only to
# check that a stressed vowel got frames and voicing.
VOWELS = frozenset(("ʌ", "i", "ɪ", "o", "ɑ", "y", "e", "a", "u", "ja", "ju",
                    "ɛ", "ə", "ɔ", "u\"", "ɵ", "iː", "ɜː", "aɪ", "əʊ", "ø"))

VOICELESS = frozenset(("p", "pʲ", "t", "tʲ", "k", "kʲ", "f", "fʲ", "s", "sʲ",
                       "ʃ", "ɕ", "x", "ts", "tʃʲ", "c"))
