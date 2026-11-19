from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from voca_ru.eval import (  # noqa: E402
    FakeASR,
    cer,
    evaluate,
    load_stress_set,
    score_text,
    wer,
)
from voca_ru.eval.__main__ import main  # noqa: E402
from voca_ru.eval.metrics import edit_counts  # noqa: E402

try:
    from fixtures import FIXTURES, make_fixtures
except ImportError:  # discover -s tests/eval keeps the dir on sys.path
    from tests.eval.fixtures import FIXTURES, make_fixtures


class ScoringTest(unittest.TestCase):
    def test_fold_and_punctuation(self):
        self.assertEqual(score_text("Привет, МИР!"), "привет мир")
        self.assertEqual(score_text("Пётр"), "петр")

    def test_stress_and_digits(self):
        self.assertEqual(score_text("замо́к"), "замок")
        self.assertEqual(score_text("23 января"), "двадцать третьего января")
        self.assertEqual(score_text("1000"), "тысяча")

    def test_idempotent(self):
        text = "В 1917 году было 3 дома."
        self.assertEqual(score_text(score_text(text)), score_text(text))


class MetricsTest(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(wer("один два три", "один два три"), 0.0)
        self.assertEqual(cer("один два", "один два"), 0.0)

    def test_fully_wrong(self):
        self.assertEqual(wer("один два три", "альфа бета гамма"), 1.0)

    def test_substitution_and_insertion(self):
        self.assertEqual(edit_counts("один два три четыре", "один три три четыре"),
                         (1, 4, 1, 0))
        errors, ref_len, _, ins = edit_counts("один два", "один два три")
        self.assertEqual((errors, ref_len, ins), (1, 2, 1))

    def test_cer(self):
        self.assertAlmostEqual(cer("кот", "кит"), 1 / 3)


class FixtureSelfCheck(unittest.TestCase):
    """Five tone/noise clips, no speech: the harness must say 0 % or 100 %."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        make_fixtures(root)
        cls.root = root
        mismatch = {"tone_other": "альфа бета гамма дельта",
                    "noise_other": "альфа бета гамма"}
        transcripts = {}
        rows = []
        for name, text, _seconds, _kind in FIXTURES:
            audio = root / f"{name}.wav"
            transcripts[str(audio.resolve())] = mismatch.get(name, text)
            rows.append({"id": name, "text": text, "audio": audio.name})
        cls.rows = rows
        cls.report = evaluate(rows, FakeASR(transcripts), root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_row_wer(self):
        by_id = {row["id"]: row for row in self.report["rows"]}
        for name, _text, _seconds, _kind in FIXTURES:
            expected = 1.0 if name.endswith("_other") else 0.0
            self.assertEqual(by_id[name]["wer"], expected, name)

    def test_all_mismatches_are_100pct(self):
        mismatches = [r for r in self.report["rows"] if r["id"].endswith("_other")]
        self.assertEqual(len(mismatches), 2)
        self.assertTrue(all(r["wer"] == 1.0 for r in mismatches))

    def test_timing_recorded(self):
        self.assertGreater(self.report["overall"]["audio_seconds"], 2.5)
        self.assertIn("asr_seconds", self.report["overall"])

    def test_synthesis_timing_fields(self):
        row = {"id": "t", "text": "один два", "audio": self.rows[0]["audio"],
               "synth_seconds": 1.0, "first_audio_seconds": 0.25}
        report = evaluate(
            [row],
            FakeASR({str((self.root / row["audio"]).resolve()): "один два"}),
            self.root,
        )
        self.assertEqual(report["synthesis"]["rows"], 1)
        self.assertAlmostEqual(report["synthesis"]["first_audio_seconds_mean"], 0.25)
        self.assertGreater(report["synthesis"]["rtf"], 0.0)

    def test_category_breakdown_and_single_pair_cli(self):
        report = evaluate(
            [{"id": "x", "text": "один два", "audio": self.rows[0]["audio"],
              "category": "number"},
             {"id": "y", "text": "три", "audio": self.rows[0]["audio"],
              "category": "date"}],
            FakeASR({str((self.root / self.rows[0]["audio"]).resolve()): "один два"}),
            self.root,
        )
        self.assertEqual(set(report["categories"]), {"number", "date"})

        out = self.root / "single.json"
        transcripts = self.root / "fake.json"
        transcripts.write_text(json.dumps(
            {str((self.root / self.rows[0]["audio"]).resolve()): "один два три"}))
        code = main(["--audio", str(self.root / self.rows[0]["audio"]),
                     "--text", "один два три", "--backend", "fake",
                     "--fake-transcripts", str(transcripts), "--out", str(out),
                     "--md", str(self.root / "single.md")])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_text())["overall"]["wer"], 0.0)


class StressSetTest(unittest.TestCase):
    def test_set(self):
        items = load_stress_set()
        self.assertEqual(len(items), 200)
        self.assertEqual(len({i["id"] for i in items}), 200)
        groups = {i["group"] for i in items}
        self.assertIn("homograph", groups)

    def test_evaluate_without_detector_is_explicit(self):
        from voca_ru.eval import evaluate_stress

        items = load_stress_set()[:3]
        transcripts = {item["id"]: item["word"] for item in items}
        result = evaluate_stress(items, transcripts)
        self.assertEqual(result["word_miss_rate"], 0.0)
        self.assertIsNone(result["stress_error_rate"])
        self.assertIsNone(result["detector"])
        self.assertIn("no stress marks", result["note"])


if __name__ == "__main__":
    unittest.main()
