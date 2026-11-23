"""Health-check threshold logic on fake snapshots (no GPU, no corpus)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "lib"))

import health  # noqa: E402


def status(**overrides):
    base = {
        "clips_done": 100,
        "clips_expected": 16000,
        "clips_per_min": 50.0,
        "eta_hours": 5.0,
        "wer_mean": 0.01,
        "wer_p95": 0.04,
        "dropped": 2,
        "stress_fail_pct": 0.5,
        "gpu_temp_c": 70,
        "vram_used_gb": 12.0,
        "disk_free_gb": 80.0,
        "last_clip_age_s": 30.0,
    }
    base.update(overrides)
    return base


class EvaluateTests(unittest.TestCase):
    def test_healthy_snapshot_has_no_alarms(self):
        self.assertEqual(health.evaluate(status(), baseline=50.0), [])

    def test_no_new_clip_for_20_minutes(self):
        alarms = health.evaluate(status(last_clip_age_s=1200.1), baseline=50.0)
        self.assertTrue(any("no new clip" in a for a in alarms))

    def test_no_clip_alarm_exactly_at_limit_is_quiet(self):
        self.assertEqual(health.evaluate(status(last_clip_age_s=1200), baseline=50.0), [])

    def test_no_clip_alarm_suppressed_when_finished(self):
        done = status(clips_done=16000, last_clip_age_s=99999)
        self.assertEqual(health.evaluate(done, baseline=50.0), [])

    def test_rate_drop_below_half_of_baseline(self):
        alarms = health.evaluate(status(clips_per_min=24.9), baseline=50.0)
        self.assertTrue(any("clips/min" in a for a in alarms))
        self.assertEqual(health.evaluate(status(clips_per_min=25.0), baseline=50.0), [])

    def test_no_rate_alarm_without_baseline(self):
        self.assertEqual(health.evaluate(status(clips_per_min=1.0), baseline=None), [])

    def test_stress_failures_above_two_percent(self):
        alarms = health.evaluate(status(stress_fail_pct=2.01), baseline=50.0)
        self.assertTrue(any("stress" in a for a in alarms))
        self.assertEqual(health.evaluate(status(stress_fail_pct=2.0), baseline=50.0), [])

    def test_wer_mean_above_five_percent(self):
        alarms = health.evaluate(status(wer_mean=0.0501), baseline=50.0)
        self.assertTrue(any("WER" in a for a in alarms))
        self.assertEqual(health.evaluate(status(wer_mean=0.05), baseline=50.0), [])

    def test_disk_and_temperature(self):
        self.assertTrue(any("disk" in a for a in
                            health.evaluate(status(disk_free_gb=9.99), baseline=50.0)))
        self.assertTrue(any("temperature" in a for a in
                            health.evaluate(status(gpu_temp_c=86), baseline=50.0)))
        self.assertEqual(health.evaluate(status(gpu_temp_c=85), baseline=50.0), [])

    def test_all_alarms_at_once(self):
        bad = status(clips_per_min=1, stress_fail_pct=9, wer_mean=0.2,
                     disk_free_gb=1, gpu_temp_c=99, last_clip_age_s=5000)
        alarms = health.evaluate(bad, baseline=50.0)
        self.assertEqual(len(alarms), 6, alarms)


class BaselineTests(unittest.TestCase):
    def test_first_ten_minute_average(self):
        history = [{"t": 0, "clips_done": 0}, {"t": 300, "clips_done": 100},
                   {"t": 600, "clips_done": 200}, {"t": 900, "clips_done": 500}]
        self.assertAlmostEqual(health.baseline_rate(history), 20.0)

    def test_baseline_needs_two_samples(self):
        self.assertIsNone(health.baseline_rate([{"t": 0, "clips_done": 0}]))


class BuildStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        os.makedirs(os.path.join(self.dir, "audio", "f_neutral"))
        self.now = 1_800_000_000

    def tearDown(self):
        self.tmp.cleanup()

    def write_fixture(self):
        audio = os.path.join(self.dir, "audio/f_neutral/a.flac")
        with open(audio, "wb") as handle:
            handle.write(b"x")
        os.utime(audio, (self.now - 42, self.now - 42))
        rows = [
            {"id": "a", "audio": "audio/f_neutral/a.flac", "seconds": 2.0,
             "voice": "f_neutral", "wer": 0.0, "stress_checked": 10, "stress_errors": 1},
            {"id": "b", "audio": "audio/f_neutral/b.flac", "seconds": 4.0,
             "voice": "f_neutral", "wer": 0.10, "stress_checked": 10, "stress_errors": 0},
        ]
        with open(os.path.join(self.dir, "manifest.jsonl"), "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        with open(os.path.join(self.dir, "review.jsonl"), "w") as handle:
            handle.write(json.dumps({"id": "c", "reject_reasons": ["wer"]}) + "\n")

    def test_status_fields(self):
        self.write_fixture()
        history = [{"t": self.now - 600, "clips_done": 0}]
        result = health.build_status(
            self.dir, 1000, self.now, gpu=(73, 11.5, 24.0), free_gb=77.0,
            history=history,
        )
        self.assertEqual(result["clips_done"], 2)
        self.assertEqual(result["dropped"], 1)
        self.assertAlmostEqual(result["wer_mean"], 0.05)
        self.assertAlmostEqual(result["wer_p95"], 0.10)
        self.assertAlmostEqual(result["stress_fail_pct"], 5.0)
        self.assertEqual(result["gpu_temp_c"], 73)
        self.assertEqual(result["vram_used_gb"], 11.5)
        self.assertEqual(result["disk_free_gb"], 77.0)
        self.assertEqual(result["last_clip_age_s"], 42.0)
        self.assertAlmostEqual(result["clips_per_min"], 0.2)
        self.assertAlmostEqual(result["eta_hours"], round(998 / 0.2 / 60, 2))

    def test_first_sample_has_zero_rate_and_no_eta(self):
        self.write_fixture()
        result = health.build_status(self.dir, 1000, self.now, gpu=(None, None, None),
                                     free_gb=50.0, history=[])
        self.assertEqual(result["clips_per_min"], 0.0)
        self.assertIsNone(result["eta_hours"])

    def test_missing_audio_gives_null_age(self):
        with open(os.path.join(self.dir, "manifest.jsonl"), "w") as handle:
            handle.write(json.dumps({"id": "a", "audio": "audio/f_neutral/a.flac",
                                     "seconds": 1}) + "\n")
        result = health.build_status(self.dir, 10, self.now, gpu=(None, None, None),
                                     free_gb=50.0)
        self.assertIsNone(result["last_clip_age_s"])
        self.assertIsNone(result["wer_p95"])
        self.assertIsNone(result["stress_fail_pct"])


if __name__ == "__main__":
    unittest.main()
