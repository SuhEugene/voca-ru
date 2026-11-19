import unittest

import _support  # noqa: F401
from voca_ru.text.symbols_ru import (
    ALIGNER_LABEL, CONSONANTS, FIRST_PHONE, FIRST_PUNCTUATION, FIRST_VOWEL,
    PAD, BOS, EOS, SP, UNK, SPECIALS, SYMBOLS, SYMBOL_TO_ID, VOWELS,
    is_phone, is_vowel,
)

ESPEAK_UNITS = {
    "ʌ", "i", "n", "t", "s", "ɪ", "ɑ", "o", "r", "y", "v", "j", "k", "a",
    "e", "p", "ɭ", "u", "m", "ɭʲ", "tʲ", "nʲ", "d", "rʲ", "b", "sʲ", "z",
    "x", "mʲ", "ɡ", "tʃʲ", "ts", "f", "dʲ", "vʲ", "ɛ", "ʃ", "ʒ", "kʲ",
    "ɕ", "pʲ", "ɡʲ", "ə", "bʲ", "ʑ", "ɵ", "fʲ", "dʒ",
}


class InventoryTest(unittest.TestCase):
    def test_size_and_ids(self):
        self.assertEqual(len(SYMBOLS), 76)
        self.assertEqual(len(set(SYMBOLS)), 76)
        self.assertEqual(sorted(SYMBOL_TO_ID.values()), list(range(76)))

    def test_layout(self):
        self.assertEqual(SYMBOLS[:5], ["<pad>", "<bos>", "<eos>", "<sp>", "<unk>"])
        self.assertEqual((PAD, BOS, EOS, SP, UNK), tuple(SPECIALS))
        self.assertEqual(FIRST_PUNCTUATION, 5)
        self.assertEqual(FIRST_PHONE, 5 + 23)
        self.assertEqual(FIRST_VOWEL, FIRST_PHONE + 36)
        self.assertEqual(FIRST_VOWEL + len(VOWELS), 76)

    def test_upstream_prefix_is_stable(self):
        upstream = [",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…"]
        self.assertEqual(SYMBOLS[5:17], upstream)

    def test_phone_counts(self):
        self.assertEqual(len(CONSONANTS), 36)
        self.assertEqual(len(VOWELS), 12)
        self.assertEqual(len(set(CONSONANTS) | set(VOWELS)), 48)
        for phone in CONSONANTS:
            self.assertTrue(is_phone(phone) and not is_vowel(phone))
        for vowel in VOWELS:
            self.assertTrue(is_phone(vowel) and is_vowel(vowel))

    def test_every_espeak_unit_has_a_training_phone(self):
        missing = ESPEAK_UNITS - set(CONSONANTS) - set(VOWELS)
        self.assertEqual(missing, set())

    def test_aligner_label_covers_inventory(self):
        for phone in CONSONANTS + VOWELS:
            self.assertIn(phone, ALIGNER_LABEL)
        self.assertEqual(ALIGNER_LABEL["ɵ"], "o")
        self.assertEqual(ALIGNER_LABEL["ə"], "ʌ")
        self.assertEqual(ALIGNER_LABEL["ɪ"], "i")
        self.assertEqual(ALIGNER_LABEL["a"], "ʌ")
        self.assertEqual(ALIGNER_LABEL["ɛ"], "e")


if __name__ == "__main__":
    unittest.main()
