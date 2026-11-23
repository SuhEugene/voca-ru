"""Smoke-result validation (runs here; the smoke itself needs the GPU)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "lib"))

import smoke  # noqa: E402


def make_smoke(root, *, adapter=".cache/models/qwen3-tts-ru-stress/lora",
               stressed="приве́т ми́р", wer=0.0, stress_ok=True):
    import numpy as np
    import soundfile as sf

    os.makedirs(os.path.join(root, "audio", "f_neutral"), exist_ok=True)
    audio = os.path.join(root, "audio", "f_neutral", "t1_f_neutral.flac")
    sf.write(audio, np.zeros(24000, dtype="float32"), 24000, subtype="PCM_16")
    row = {
        "id": "t1_f_neutral", "audio": "audio/f_neutral/t1_f_neutral.flac",
        "teacher_adapter": adapter, "stressed_text": stressed, "wer": wer,
        "stress_ok": stress_ok, "stress_checked": 3, "stress_errors": 0,
    }
    with open(os.path.join(root, "manifest.jsonl"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class SmokeCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_good_smoke_passes(self):
        make_smoke(self.tmp.name)
        ok, lines = smoke.check(self.tmp.name)
        self.assertTrue(ok, lines)
        self.assertTrue(any("WER 0.00" in line for line in lines))
        self.assertTrue(any("stress adapter loaded" in line for line in lines))

    def test_base_without_adapter_fails(self):
        make_smoke(self.tmp.name, adapter="none (LoRA failed: RuntimeError)")
        ok, lines = smoke.check(self.tmp.name)
        self.assertFalse(ok)
        self.assertTrue(any("not the stress LoRA" in line for line in lines))

    def test_missing_stress_marks_fail(self):
        make_smoke(self.tmp.name, stressed="привет мир")
        ok, lines = smoke.check(self.tmp.name)
        self.assertFalse(ok)
        self.assertTrue(any("stress marks missing" in line for line in lines))

    def test_empty_manifest_fails(self):
        os.makedirs(self.tmp.name, exist_ok=True)
        ok, _lines = smoke.check(self.tmp.name)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
