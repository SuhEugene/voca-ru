"""Status rendering from the real status.json shape, plus widget size limits."""

from __future__ import annotations

import json
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "tg"))

import tgcore  # noqa: E402

# Field-for-field what remote/lib/health.py build_status() writes.
STATUS = {
    "updated_at": "2026-11-24T12:00:00Z",
    "run_started_at": "2026-11-24T09:00:00Z",
    "elapsed_min": 180.0,
    "clips_done": 7552,
    "clips_expected": 16000,
    "progress_pct": 47.2,
    "clips_per_min": 39.2,
    "eta_hours": 3.59,
    "wer_mean": 0.0124,
    "wer_p95": 0.031,
    "dropped": 12,
    "stress_fail_pct": 0.21,
    "gpu_temp_c": 71,
    "vram_used_gb": 21.3,
    "disk_free_gb": 38.4,
    "last_clip_age_s": 42.0,
}

# scripts/gen/build_corpus.py run-stats.json, trimmed to what the bot reads.
STATS = {
    "clips_generated": 7552,
    "clips_accepted": 7540,
    "clips_dropped": 12,
    "clips_per_min": 39.2,
    "wer_mean": 0.0124,
    "wer_max": 0.31,
    "stress_failed_clips": 3,
}

POD = {"id": "abc", "cost": 0.34,
       "ssh": "ssh x@ssh.runpod.io", "created": "2026-11-24T09:00:00Z"}

NOW = tgcore.parse_iso("2026-11-24T12:00:00Z")


class StatusTextTests(unittest.TestCase):
    def render(self, **overrides):
        status = dict(STATUS)
        status.update(overrides)
        return tgcore.format_status(status, stats=STATS, pod=POD, now=NOW)

    def test_phone_lines(self):
        text = self.render()
        for needle in ("47.2%", "7,552 / 16,000", "39.2/min", "WER 1.24%",
                       "p95 3.10%", "stress 0.21%", "dropped 12",
                       "71\u00b0C", "21.3 GB", "disk 38 GB"):
            self.assertIn(needle, text)

    def test_local_time_header(self):
        text = self.render()
        self.assertIn(time.strftime("%H:%M", time.localtime(NOW)), text)

    def test_spend_is_rate_times_hours_since_created(self):
        text = self.render()
        self.assertIn("$0.34/h", text)
        self.assertIn("= $1.02", text)  # 3 h x 0.34

    def test_eta_from_rate(self):
        self.assertIn("ETA 3h35m", self.render())
        self.assertIn("last clip 42s", self.render())

    def test_stale_status_is_red(self):
        text = self.render(updated_at="2026-11-24T11:00:00Z")
        self.assertTrue(text.startswith("\U0001f534"))
        self.assertIn("stale status", text)

    def test_healthy_status_is_green(self):
        self.assertTrue(self.render().startswith("\U0001f7e2"))

    def test_missing_gpu_and_wer_do_not_crash(self):
        text = self.render(gpu_temp_c=None, vram_used_gb=None, wer_mean=None,
                           wer_p95=None, stress_fail_pct=None)
        self.assertIn("47.2%", text)
        self.assertNotIn("None", text)

    def test_empty_status_before_the_first_sample(self):
        text = tgcore.format_status({}, now=NOW)
        self.assertIn("0 clips", text)
        self.assertIn("last clip \u2014", text)

    def test_pod_missing_omits_spend(self):
        text = tgcore.format_status(STATUS, stats=STATS, pod={}, now=NOW)
        self.assertNotIn("$0.34/h", text)

    def test_dropped_comes_from_status_not_stats(self):
        text = self.render(dropped=12)
        self.assertIn("dropped 12", text)


class WidgetTests(unittest.TestCase):
    def test_public_data_is_small_json(self):
        payload = tgcore.widget_public_data(STATUS, stats=STATS, pod=POD, now=NOW)
        self.assertLessEqual(len(payload.encode("utf-8")), tgcore.WIDGET_STATE_LIMIT)
        data = json.loads(payload)
        for key in ("t", "p", "d", "n", "r", "e", "w", "w9", "s", "dr", "g", "df", "c"):
            self.assertIn(key, data)
        self.assertEqual(data["d"], 7552)
        self.assertEqual(data["n"], 16000)

    def test_public_data_stays_small_with_extreme_numbers(self):
        status = dict(STATUS, clips_done=10**9, clips_expected=10**9,
                      progress_pct=99.99, wer_mean=0.9999, wer_p95=0.9999,
                      stress_fail_pct=99.99, dropped=10**9,
                      vram_used_gb=1234.56, vram_total_gb=1234.56,
                      disk_free_gb=1234.56)
        payload = tgcore.widget_public_data(status, stats=STATS, pod=POD, now=NOW)
        self.assertLessEqual(len(payload.encode("utf-8")), tgcore.WIDGET_STATE_LIMIT)

    def test_html_within_both_documented_limits(self):
        html = tgcore.widget_html()
        self.assertLessEqual(len(html.encode("utf-8")), tgcore.WIDGET_HTML_LIMIT)
        self.assertLessEqual(tgcore.compress_bound(html), 4096)

    def test_html_reads_the_shared_state(self):
        html = tgcore.widget_html()
        self.assertIn("TelegramWidget", html)
        self.assertIn("publicData", html)
        self.assertIn("watchPublicData", html)


if __name__ == "__main__":
    unittest.main()
