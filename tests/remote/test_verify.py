"""Manifest / FLAC verification (the logic publish.sh refuses to upload without)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "lib"))

import verify  # noqa: E402

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - the test env has soundfile
    sf = None


def write_flac(path, seconds=1.0, sr=24000, channels=1):
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    samples = np.zeros((int(seconds * sr), channels), dtype="float32")
    sf.write(path, samples, sr, subtype="PCM_16")


def row(cid="t1_f_neutral", voice="f_neutral", seconds=1.0, **extra):
    base = {
        "id": cid,
        "text_id": cid.rsplit("_", 1)[0],
        "text": "привет",
        "audio": f"audio/{voice}/{cid}.flac",
        "seconds": seconds,
        "voice": voice,
        "wer": 0.0,
        "stress_checked": 4,
        "stress_errors": 1,
        "category": "conversation",
        "split": "train",
    }
    base.update(extra)
    return base


class CorpusFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        os.makedirs(os.path.join(self.dir, "audio", "f_neutral"), exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows, review=()):
        with open(os.path.join(self.dir, "manifest.jsonl"), "w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        if review:
            with open(os.path.join(self.dir, "review.jsonl"), "w", encoding="utf-8") as handle:
                for item in review:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    @unittest.skipIf(sf is None, "soundfile not installed")
    def test_valid_corpus_passes_and_reports_stats(self):
        write_flac(os.path.join(self.dir, "audio/f_neutral/t1_f_neutral.flac"), 1.0)
        write_flac(os.path.join(self.dir, "audio/f_neutral/t2_f_neutral.flac"), 2.0)
        self.write([row(), row("t2_f_neutral", seconds=2.0)],
                   [{"id": "t3", "reject_reasons": ["wer 0.400 > 0.15"]}])
        result = verify.verify(self.dir)
        self.assertTrue(result["ok"], result["errors"])
        stats = result["stats"]
        self.assertEqual(stats["clips"], 2)
        self.assertEqual(stats["dropped"], 1)
        self.assertEqual(stats["voices"], {"f_neutral": 2})
        self.assertAlmostEqual(stats["audio_seconds"], 3.0, places=3)
        self.assertEqual(stats["stress_checked"], 8)
        self.assertEqual(stats["stress_errors"], 2)
        self.assertAlmostEqual(stats["stress_error_pct"], 25.0, places=2)
        self.assertEqual(stats["wer_mean"], 0.0)

    def test_missing_audio_file_fails(self):
        self.write([row()])
        result = verify.verify(self.dir)
        self.assertFalse(result["ok"])
        self.assertTrue(any("audio file missing" in e for e in result["errors"]))

    def test_orphan_flac_fails(self):
        write_flac(os.path.join(self.dir, "audio/f_neutral/t1_f_neutral.flac"), 1.0)
        write_flac(os.path.join(self.dir, "audio/f_neutral/orphan.flac"), 1.0)
        self.write([row()])
        result = verify.verify(self.dir)
        self.assertFalse(result["ok"])
        self.assertTrue(any("without a manifest row" in e for e in result["errors"]))

    def test_duplicate_id_fails(self):
        write_flac(os.path.join(self.dir, "audio/f_neutral/t1_f_neutral.flac"), 1.0)
        self.write([row(), row()])
        result = verify.verify(self.dir)
        self.assertFalse(result["ok"])
        self.assertTrue(any("duplicate id" in e for e in result["errors"]))

    @unittest.skipIf(sf is None, "soundfile not installed")
    def test_duration_mismatch_fails(self):
        write_flac(os.path.join(self.dir, "audio/f_neutral/t1_f_neutral.flac"), 3.0)
        self.write([row(seconds=1.0)])
        result = verify.verify(self.dir)
        self.assertFalse(result["ok"])
        self.assertTrue(any("duration" in e for e in result["errors"]))

    @unittest.skipIf(sf is None, "soundfile not installed")
    def test_wrong_sample_rate_fails(self):
        write_flac(os.path.join(self.dir, "audio/f_neutral/t1_f_neutral.flac"), 1.0, sr=16000)
        self.write([row(seconds=1.0)])
        result = verify.verify(self.dir)
        self.assertFalse(result["ok"])
        self.assertTrue(any("sample rate" in e for e in result["errors"]))

    def test_path_traversal_fails(self):
        self.write([row(audio="../../etc/passwd")])
        result = verify.verify(self.dir)
        self.assertFalse(result["ok"])
        self.assertTrue(any("escapes the corpus" in e for e in result["errors"]))

    @unittest.skipIf(sf is None, "soundfile not installed")
    def test_dataset_card_contains_licence_and_shas(self):
        write_flac(os.path.join(self.dir, "audio/f_neutral/t1_f_neutral.flac"), 1.0)
        self.write([row()])
        stats = verify.verify(self.dir)["stats"]
        meta = {
            "repo": "me/corpus", "teacher_repo": "Qwen/base",
            "adapter_repo": "me/adapter", "year": "2026",
            "generated_on": "2026-01-01", "teacher_sha": "a" * 64,
            "tokenizer_sha": "b" * 64, "adapter_sha": "c" * 64,
            "head_sha": "d" * 64, "xlsr_sha": "e" * 64, "whisper_sha": "f" * 64,
            "sentences": 1, "batch_size": 16,
        }
        card = verify.dataset_card(stats, meta)
        for needle in ("research-only", "Qwen/base", "me/adapter",
                       "a" * 64, "24000 Hz", "bibtex", "2026-01-01"):
            self.assertIn(needle, card)


if __name__ == "__main__":
    unittest.main()
