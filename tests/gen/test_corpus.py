"""Self-checks for the corpus builder's model-free logic.

    . scripts/cache-env.sh
    PYTHONPATH=training .cache/venvs/r4/bin/python -m unittest discover -s tests/gen
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for path in (os.path.join(REPO, "training"), os.path.join(REPO, "scripts/gen")):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_corpus as bc  # noqa: E402
import screen  # noqa: E402

SR = 24000


class AudioTests(unittest.TestCase):
    def test_trim_keeps_pad_and_signal(self):
        silence = np.zeros(SR // 2, np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR).astype(np.float32)
        wav = np.concatenate([silence, tone, silence])
        out, lead, tail = screen.trim_silence(wav, SR)
        self.assertGreater(lead, 0.45)
        self.assertGreaterEqual(tail, 0.45)
        self.assertAlmostEqual(len(out), (1.0 + 2 * screen.TRIM_PAD) * SR, delta=2)
        self.assertAlmostEqual(float(np.abs(out).max()), 0.5, places=3)

    def test_peak_normalise(self):
        wav = 0.2 * np.sin(np.linspace(0, 10, SR)).astype(np.float32)
        out, gain = screen.peak_normalize(wav)
        self.assertAlmostEqual(float(np.abs(out).max()), 0.95, places=4)
        self.assertGreater(gain, 0)

    def test_prepare_flags_raw_clipping(self):
        wav = np.ones(SR, np.float32)  # constant 1.0 clips
        shaped, stats = screen.prepare_audio(wav)
        self.assertTrue(stats["raw_clipped"])
        self.assertFalse(stats["clipped"])
        self.assertAlmostEqual(stats["peak"], 0.95, places=3)


class BucketTests(unittest.TestCase):
    def test_batches_sorted_and_bounded(self):
        rows = [{"id": f"{i:03d}", "plain_text": "x" * n} for i, n in enumerate([9, 3, 7, 1, 5])]
        batches = bc.make_batches(rows, 2)
        self.assertEqual([len(b) for b in batches], [2, 2, 1])
        lengths = [len(r["plain_text"]) for b in batches for r in b]
        self.assertEqual(lengths, sorted(lengths))

    def test_seed_is_stable_and_attempt_sensitive(self):
        a = bc.seed_for(["a", "b"], "f_neutral", 0, 1234)
        self.assertEqual(a, bc.seed_for(["a", "b"], "f_neutral", 0, 1234))
        self.assertNotEqual(a, bc.seed_for(["a", "b"], "f_neutral", 1, 1234))
        self.assertNotEqual(a, bc.seed_for(["a", "b"], "m_neutral", 0, 1234))
        self.assertLess(a, 2**31)

    def test_pilot_is_20_across_categories(self):
        rows = []
        for cat, n in [("number", 20), ("date", 20), ("money", 20), ("name", 20),
                       ("question", 20), ("conversation", 20), ("reading", 20)]:
            for i in range(n):
                rows.append({"id": f"{cat}{i:03d}",
                             "text": f"Это тестовое предложение номер {i} для категории.",
                             "category": cat})
        pilot = bc.select_pilot(rows)
        self.assertEqual(len(pilot), 20)
        cats = {r["category"] for r in pilot}
        self.assertEqual(cats, {"number", "date", "money", "name", "question",
                                "conversation", "reading"})


class FakeMeter:
    """Only the three helpers ``stress_check`` borrows from meter2."""
    V = "аеёиоуыэюя"
    ACUTE = "\u0301"

    @staticmethod
    def plain_text(text):
        return text.replace(FakeMeter.ACUTE, "").replace("+", "")

    @staticmethod
    def vowel_pos(word):
        return [i for i, ch in enumerate(word.lower()) if ch in FakeMeter.V]

    @staticmethod
    def marked_index(word):
        n, hit, idx = 0, False, None
        w = word.lower()
        for j, ch in enumerate(w):
            if ch == "+":
                hit = True
                continue
            if ch in FakeMeter.V:
                if hit or (j + 1 < len(w) and w[j + 1] == FakeMeter.ACUTE):
                    idx = n
                hit = False
                n += 1
        return idx


class FakeJudge:
    _meter2 = FakeMeter

    def __init__(self, best):
        self.best = best

    def judge(self, wav, sr, text):
        words = screen._WORD_MARKED.findall(text)
        return [{"best": self.best[i] if i < len(self.best) else None,
                 "margin": 0.9, "align": 0.99} for i in range(len(words))]


class StressTests(unittest.TestCase):
    def test_correct_marks_pass(self):
        text = "дома́ рабо́та хорошо́"
        judge = FakeJudge([1, 1, 2])  # до-ма, ра-бо-та, хо-ро-шо
        out = screen.stress_check(np.zeros(10, np.float32), SR, text, judge)
        self.assertTrue(out["stress_ok"])
        self.assertEqual(out["stress_checked"], 3)
        self.assertEqual(out["stress_errors"], 0)

    def test_shifted_marks_fail(self):
        text = "дома́ рабо́та хорошо́"
        judge = FakeJudge([0, 2, 0])  # every stress in the wrong place
        out = screen.stress_check(np.zeros(10, np.float32), SR, text, judge)
        self.assertFalse(out["stress_ok"])
        self.assertEqual(out["stress_errors"], 3)


class FakeASR:
    def __init__(self, text):
        self.text = text

    def transcribe(self, path):
        return self.text


class ScreenTests(unittest.TestCase):
    def test_good_clip_and_duration_reject(self):
        text = "Сегодня хорошая погода."
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip.flac")
            sf.write(path, np.zeros(SR, np.float32) + 0.1, SR, subtype="PCM_16")
            stats = screen.audio_stats(np.zeros(SR, np.float32) + 0.1, SR)
            stats["raw_clipped"] = False
            good = screen.screen_clip(path, text=text, stressed_text=text, stats=stats,
                                      asr=FakeASR(text), stress=None)
            self.assertTrue(good["ok"], good["reject_reasons"])
            self.assertEqual(good["wer"], 0.0)

            bad = screen.screen_clip(os.path.abspath(__file__), text=text,
                                     stressed_text=text, stats={**stats, "seconds": 40.0},
                                     asr=FakeASR(text), stress=None)
            self.assertFalse(bad["ok"])
            self.assertTrue(any("outside" in r for r in bad["reject_reasons"]))


if __name__ == "__main__":
    unittest.main()
