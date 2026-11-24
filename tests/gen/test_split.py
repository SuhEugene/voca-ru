"""Self-checks for the two-phase corpus build (model-free).

    . scripts/cache-env.sh
    PYTHONPATH=training .cache/venvs/r4/bin/python -m unittest discover -s tests/gen
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for path in (os.path.join(REPO, "training"), os.path.join(REPO, "scripts/gen")):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_corpus as bc  # noqa: E402
import split_corpus as sc  # noqa: E402

SR = 24000
TEXT = "Сегодня хорошая погода"


class FakeTeacher:
    """Short tone per clip; records every call."""

    def __init__(self):
        self.calls = []

    def generate(self, voice, texts, seed):
        n = SR
        wavs = [(0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)).astype(np.float32)
                for _ in texts]
        self.calls.append({"voice": voice, "texts": list(texts), "seed": seed})
        return wavs, SR, 0.01 * len(texts)

    @property
    def cids(self):
        return [f"{t}_x" for c in self.calls for t in c["texts"]]


class FakeASR:
    """Path basename -> transcript, so a test decides which clips fail WER."""

    def __init__(self, transcripts=None):
        self.transcripts = transcripts or {}
        self.calls = []

    def transcribe(self, path):
        self.calls.append(path)
        return self.transcripts.get(os.path.basename(path), TEXT)


def make_rows(n=4, prefix="a"):
    rows = []
    for i in range(n):
        rows.append({"id": f"{prefix}{i}", "text": TEXT, "plain_text": TEXT,
                     "norm_text": TEXT, "stressed_text": TEXT,
                     "phonemes": ["<bos>", "x", "<eos>"], "tokens": [1, 2, 3],
                     "category": "question", "split": "train"})
    return rows


def run_generate(out, rows, voice="f_neutral", batch_size=2, wave=1, ids=None,
                 teacher=None):
    teacher = teacher or FakeTeacher()
    sc.run_generate(rows=rows, voices=[voice], batch_size=batch_size, base_seed=1234,
                    out=out, teacher=teacher, wave=wave,
                    ids=ids, adapter="fake-lora")
    return teacher


def raw_rows(out):
    return sc.read_jsonl(os.path.join(out, "raw.jsonl"))


class GenerateTests(unittest.TestCase):
    def test_generate_only_never_screens(self):
        with tempfile.TemporaryDirectory() as out:
            with mock.patch.object(sc.screen, "screen_clip",
                                   side_effect=AssertionError("screened")):
                run_generate(out, make_rows(3), batch_size=2)
            rows = raw_rows(out)
            self.assertEqual(len(rows), 3)
            required = {"id", "voice", "text", "stressed_text", "phonemes", "tokens",
                        "audio", "seconds", "seed", "attempt", "wave"}
            self.assertTrue(required <= set(rows[0]))
            self.assertEqual(rows[0]["wave"], 1)
            self.assertTrue(os.path.isfile(os.path.join(out, rows[0]["audio"])))
            self.assertFalse(os.path.exists(os.path.join(out, "manifest.jsonl")))

    def test_resume_skips_known_ids(self):
        with tempfile.TemporaryDirectory() as out:
            rows = make_rows(4)
            run_generate(out, rows, batch_size=2)
            teacher = run_generate(out, rows, batch_size=2)
            self.assertEqual(teacher.calls, [])
            self.assertEqual(len(raw_rows(out)), 4)

    def test_peak_is_printed_per_batch(self):
        with tempfile.TemporaryDirectory() as out, \
                mock.patch("builtins.print") as printed:
            run_generate(out, make_rows(4), batch_size=2)
        lines = [c.args[0] for c in printed.call_args_list if c.args]
        self.assertTrue(any("peak_device_gib" in str(line) for line in lines))


class WaveTests(unittest.TestCase):
    def test_wave_file_selects_exactly_rejected_ids(self):
        with tempfile.TemporaryDirectory() as out:
            rows = make_rows(4)
            run_generate(out, rows, batch_size=4)
            asr = FakeASR({f"a1_f_neutral.flac": "", f"a3_f_neutral.flac": ""})
            sc.run_screen(out=out, asr=asr, stress=None, max_wer=0.15)
            ids_file = os.path.join(out, "wave2.ids")
            report = sc.run_wave_report(out, ids_file)
            self.assertEqual(report["rejected"], 2)
            self.assertEqual(sc.read_ids_file(ids_file),
                             ["a1_f_neutral", "a3_f_neutral"])

            teacher = run_generate(out, rows, batch_size=4, wave=2,
                                   ids=sc.read_ids_file(ids_file))
            self.assertEqual(len(teacher.calls), 2)
            self.assertTrue(all(len(c["texts"]) == 1 for c in teacher.calls))
            waves = {r["id"]: r["wave"] for r in raw_rows(out)}
            self.assertEqual(waves["a1_f_neutral"], 2)
            self.assertEqual(waves["a3_f_neutral"], 2)
            self.assertEqual(waves["a0_f_neutral"], 1)
            # the retry seed matches the sequential single-clip retry seed
            self.assertEqual(teacher.calls[0]["seed"],
                             bc.seed_for(["a1"], "f_neutral", 1, 1234))

    def test_empty_wave(self):
        with tempfile.TemporaryDirectory() as out:
            run_generate(out, make_rows(2), batch_size=2)
            sc.run_screen(out=out, asr=FakeASR(), stress=None)
            ids_file = os.path.join(out, "wave2.ids")
            report = sc.run_wave_report(out, ids_file)
            self.assertEqual(report["rejected"], 0)
            self.assertEqual(sc.read_ids_file(ids_file), [])
            teacher = run_generate(out, make_rows(2), wave=2, ids=[])
            self.assertEqual(teacher.calls, [])


class ScreenTests(unittest.TestCase):
    def test_screen_never_rescans(self):
        with tempfile.TemporaryDirectory() as out:
            run_generate(out, make_rows(3), batch_size=3)
            asr = FakeASR({"a1_f_neutral.flac": ""})
            first = sc.run_screen(out=out, asr=asr, stress=None)
            self.assertEqual(first["screened"], 3)
            self.assertEqual(first["accepted"], 2)
            self.assertEqual(first["rejected"], 1)
            self.assertEqual(len(asr.calls), 3)

            second = sc.run_screen(out=out, asr=asr, stress=None)
            self.assertEqual(second["screened"], 0)
            self.assertEqual(len(asr.calls), 3)  # no rescan

    def test_reason_histogram_and_accepted_schema(self):
        with tempfile.TemporaryDirectory() as out:
            run_generate(out, make_rows(3), batch_size=3)
            asr = FakeASR({"a0_f_neutral.flac": "", "a1_f_neutral.flac": ""})
            summary = sc.run_screen(out=out, asr=asr, stress=None)
            self.assertEqual(summary["reject_reasons"], {"wer N > N": 2})
            manifest = sc.read_jsonl(os.path.join(out, "manifest.jsonl"))
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest[0]["teacher_adapter"], "fake-lora")
            self.assertEqual(manifest[0]["attempt"], 1)
            self.assertEqual(manifest[0]["text_id"], "a2")
            self.assertTrue(os.path.isfile(os.path.join(out, manifest[0]["audio"])))
            rejected = os.path.join(out, "rejected", "f_neutral", "a0_f_neutral.flac")
            self.assertTrue(os.path.isfile(rejected))
            self.assertFalse(os.path.exists(
                os.path.join(out, "audio", "f_neutral", "a0_f_neutral.flac")))

    def test_second_wave_is_rescreened(self):
        with tempfile.TemporaryDirectory() as out:
            rows = make_rows(2)
            run_generate(out, rows, batch_size=2)
            asr = FakeASR({"a0_f_neutral.flac": ""})
            sc.run_screen(out=out, asr=asr, stress=None)
            self.assertEqual(len(sc.read_jsonl(os.path.join(out, "review.jsonl"))), 1)
            # wave 2 regenerates the reject; the verdict must not be reused
            run_generate(out, rows, batch_size=2, wave=2, ids=["a0_f_neutral"])
            asr.transcripts = {}
            summary = sc.run_screen(out=out, asr=asr, stress=None)
            self.assertEqual(summary["screened"], 1)
            self.assertEqual(summary["accepted"], 1)
            review = sc.read_jsonl(os.path.join(out, "review.jsonl"))
            self.assertEqual(len(review), 0)  # stale verdict removed, not duplicated


if __name__ == "__main__":
    unittest.main()
