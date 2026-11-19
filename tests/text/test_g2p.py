import unittest

import _support  # noqa: F401
from voca_ru.text.g2p import phonemize, split_units, espeak
from voca_ru.text.symbols_ru import (
    BOS, EOS, SP, SYMBOLS, SYMBOL_TO_ID, is_phone, is_vowel,
)
from voca_ru.text.stress import fallback_stress


class UnitsTest(unittest.TestCase):
    def test_affricates_and_palatalisation(self):
        units, dropped = split_units("tʃʲˈɑj")
        self.assertEqual([u for u, _ in units], ["tʃʲ", "ˈ", "ɑ", "j"])
        self.assertEqual(dropped, [])

    def test_u_umlaut_artefact(self):
        units, _ = split_units('ɭʲu"bˈof')
        self.assertEqual([u for u, _ in units], ["ɭʲ", 'u"', "b", "ˈ", "o", "f"])

    def test_unknown_unit_is_reported(self):
        _, dropped = split_units("q")
        self.assertEqual(dropped, ["q"])


class PhonemizeTest(unittest.TestCase):
    def test_version_pinned(self):
        self.assertTrue(espeak().version.startswith("1.52"), espeak().version)

    def test_privet(self):
        symbols, stress = phonemize("приве́т")
        self.assertEqual(symbols, [BOS, "p", "rʲ", "i", "vʲ", "e", "t", EOS])
        self.assertEqual(stress[symbols.index("e")], 1)

    def test_segodnya(self):
        symbols, _ = phonemize("сего́дня")
        self.assertEqual(symbols[1:-1], ["sʲ", "i", "v", "o", "d", "nʲ", "ʌ"])

    def test_yo_is_o_umlaut(self):
        symbols, stress = phonemize("ёлка")
        self.assertIn("ɵ", symbols)
        self.assertEqual(stress[symbols.index("ɵ")], 1)

    def test_stress_transfer_beats_espeak_default(self):
        # eSpeak's own reading is "за́мок"; the mark must be movable.
        symbols, stress = phonemize("замо́к")
        vowel_slots = [(i, stress[i]) for i, s in enumerate(symbols) if is_vowel(s)]
        self.assertEqual(sum(level for _, level in vowel_slots), 1)
        slot = [i for i, level in vowel_slots if level == 1][0]
        self.assertEqual(symbols[slot], "o")

    def test_punctuation_and_spacing(self):
        symbols, stress = phonemize("приве́т, как дела́?")
        self.assertEqual(symbols, [
            BOS, "p", "rʲ", "i", "vʲ", "e", "t", ",", "k", "ɑ", "k", SP,
            "dʲ", "e", "ɭ", "a", "?", EOS])
        self.assertEqual(stress.count(1), 3)

    def test_hyphenated_word(self):
        symbols, _ = phonemize("по-мо́ему")
        self.assertNotIn("-", symbols)
        self.assertIn("ʌ", symbols)

    def test_all_symbols_in_inventory(self):
        symbols, _ = phonemize("сего́дня мы́ встре́тимся в пятна́дцать три́дцать")
        for symbol in symbols:
            self.assertIn(symbol, SYMBOL_TO_ID)
            self.assertIn(SYMBOL_TO_ID[symbol], range(76))

    def test_fallback_stress_tokenises(self):
        text = fallback_stress("красота")
        symbols, stress = phonemize(text)
        self.assertEqual(sum(stress), 1)
        self.assertTrue(all(is_phone(s) or not is_vowel(s) for s in symbols))


if __name__ == "__main__":
    unittest.main()
