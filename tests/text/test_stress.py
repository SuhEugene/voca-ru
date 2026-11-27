import unittest

import _support  # noqa: F401
from voca_ru.text.stress import Stressor, fallback_stress, words_with_marks
from voca_ru.text.symbols_ru import STRESS


class FakeStressor(Stressor):
    """Dictionary-subset stressor; no download and no ONNX needed."""

    def __init__(self, accents, omographs=None):
        super().__init__(use_model=False)
        self.accents = dict(accents)
        self.omographs = dict(omographs or {})
        self.yo_words = {}


class StressTest(unittest.TestCase):
    def setUp(self):
        self.stressor = FakeStressor({
            "сегодня": "сег+одня",
            "машина": "маш+ина",
            "молоко": "молок+о",
            "замок": "з+амок",
            "ёж": "+ёж",
            "на": "н+а",
            "до": "д+о",
        })

    def test_dictionary_mark(self):
        text, oov = self.stressor.stressed("сегодня машина")
        self.assertEqual(text, "сего́дня маши́на")
        self.assertFalse(oov)

    def test_yo_marked(self):
        text, _ = self.stressor.stressed("ёж")
        self.assertEqual(text, "ё́ж")

    def test_clitics_stay_unmarked(self):
        text, _ = self.stressor.stressed("на до")
        self.assertEqual(text, "на до")
        self.assertNotIn(STRESS, text)

    def test_fallback_oov_is_flagged(self):
        text, oov = self.stressor.stressed("красота")
        self.assertTrue(oov)
        self.assertEqual(text, "красо́та")

    def test_manual_mark_wins(self):
        # the dictionary says з+амок; a hand-written acute must survive
        text, oov = self.stressor.stressed("замо́к")
        self.assertEqual(text, "замо́к")
        self.assertFalse(oov)

    def test_plus_before_vowel_marks_stress(self):
        text, _ = self.stressor.stressed("зам+ок")
        self.assertEqual(text, "замо́к")

    def test_hyphenated_word(self):
        text, _ = self.stressor.stressed("когда-нибудь")
        self.assertIn("-", text)
        self.assertEqual(text.count(STRESS), 2)

    def test_idempotent(self):
        once, _ = self.stressor.stressed("сегодня красивая машина")
        twice, _ = self.stressor.stressed(once)
        self.assertEqual(once, twice)

    def test_every_non_clitic_word_has_one_mark(self):
        text, _ = self.stressor.stressed("сегодня машина красивая на до")
        marked = words_with_marks(text)
        for word, index in marked:
            if word in {"на", "до"}:
                self.assertIsNone(index, word)
            else:
                self.assertIsNotNone(index, word)

    def test_fallback_suffixes(self):
        self.assertEqual(fallback_stress("информация"), "информа́ция")
        self.assertEqual(fallback_stress("красота"), "красо́та")
        self.assertEqual(fallback_stress("свёкла").count(STRESS), 1)

    def test_unloaded_stressor_uses_fallback(self):
        text, oov = Stressor().stressed("красота")
        self.assertTrue(oov)
        self.assertEqual(text, "красо́та")


if __name__ == "__main__":
    unittest.main()
