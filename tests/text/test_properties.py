import json
import re
import unittest

import _support
from voca_ru.text.g2p import phonemize
from voca_ru.text.normalize import normalize
from voca_ru.text.pipeline import analyze
from voca_ru.text.stress import CLITICS, Stressor, words_with_marks
from voca_ru.text.symbols_ru import STRESS, SYMBOL_TO_ID, is_phone


def corpus(limit=40):
    path = _support.ROOT / "data" / "scripts-ru" / "batch1.jsonl"
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line)["text"])
            if len(rows) >= limit:
                break
    return rows


class PropertiesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sentences = corpus()
        cls.stressor = Stressor()  # unloaded: deterministic fallback only

    def test_no_digits_or_latin_after_normalisation(self):
        for text in self.sentences:
            result = normalize(text)
            self.assertIsNone(re.search(r"[0-9]", result), result)
            self.assertIsNone(re.search(r"[a-z]", result), result)

    def test_normalisation_idempotent(self):
        for text in self.sentences:
            once = normalize(text)
            self.assertEqual(normalize(once), once)

    def test_stress_idempotent(self):
        for text in self.sentences[:15]:
            once, _ = self.stressor.stressed(normalize(text))
            twice, _ = self.stressor.stressed(once)
            self.assertEqual(once, twice)

    def test_one_mark_per_non_clitic_word(self):
        for text in self.sentences[:15]:
            marked, _ = self.stressor.stressed(normalize(text))
            words = words_with_marks(marked)
            expected = sum(
                1 for word, _ in words if word.lower() not in CLITICS)
            self.assertEqual(marked.count(STRESS), expected, marked)
            for word, index in words:
                if word.lower() in CLITICS:
                    self.assertIsNone(index, word)
                else:
                    self.assertIsNotNone(index, word)

    def test_token_ids_in_range(self):
        for text in self.sentences[:20]:
            result = analyze(text, stressor=self.stressor)
            self.assertTrue(result["tokens"])
            for token in result["tokens"]:
                self.assertIn(token, range(76))
            self.assertEqual(len(result["tokens"]), len(result["phonemes"]))
            self.assertEqual(len(result["tokens"]), len(result["stress"]))

    def test_every_symbol_is_in_inventory(self):
        for text in self.sentences[:20]:
            result = analyze(text, stressor=self.stressor)
            for symbol in result["phonemes"]:
                self.assertIn(symbol, SYMBOL_TO_ID)

    def test_phones_are_pronounceable(self):
        for text in self.sentences[:20]:
            symbols, _ = phonemize(self.stressor.stressed(normalize(text))[0])
            self.assertTrue(any(is_phone(symbol) for symbol in symbols), symbols[1:-1])


if __name__ == "__main__":
    unittest.main()
