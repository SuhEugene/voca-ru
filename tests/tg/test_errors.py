"""Error detection over log lines in the shape the remote pipeline writes."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "tg"))

import tgcore  # noqa: E402

# Lines produced by remote/generate.sh, remote/lib/health.py and build_corpus.py
HEALTHY = [
    "=== generate 2026-11-24T02:14:05Z ===",
    "voices=f_neutral,m_neutral limit=0 split=all batch_size=8",
    "already on disk: 0 accepted, 0 dropped (skipped)",
    "2026-11-24T02:30:01Z clips=120/16000 rate=11.4/min eta=23.2h "
    "wer=0.0067 dropped=0 stress=0.0% temp=71C vram=13.63GB disk=38GB age=42.0s",
    "[gen] batch 3/8, 8 sentences, 0 retries",
    "DONE generate.sh: corpus in /workspace/data/corpus-ru",
]

ERRORS = [
    ("Traceback (most recent call last):", "traceback"),
    ("  File \"/workspace/scripts/gen/build_corpus.py\", line 412, in main", None),
    ("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", "oom"),
    ("ALARM: no new clip for 21.0 min (limit 20)", "alarm"),
    ("ALARM: disk free 8.2 GB < 10.0 GB", "alarm"),
    ("FAIL generate.sh: pipeline exited 1 (rerun to resume)", "fail"),
    ("ERROR: root: adapter checkpoint is missing", "error"),
    ("subprocess.CalledProcessError: Command 'ffmpeg' returned non-zero exit status 1",
     None),
    ("make: *** [all] Error 2", None),
    ("Killed", "killed"),
    ("[12345.6] OutOfMemoryError: CUDA out of memory", "oom"),
    ("OSError: [Errno 28] No space left on device", "disk"),
]


class DetectionTests(unittest.TestCase):
    def test_healthy_lines_are_silent(self):
        for line in HEALTHY:
            self.assertIsNone(tgcore.detect_error(line), line)

    def test_error_lines(self):
        for line, expected in ERRORS:
            if expected is None:
                continue
            self.assertEqual(tgcore.detect_error(line), expected, line)

    def test_traceback_outranks_error_on_its_own_line(self):
        self.assertEqual(tgcore.detect_error("Traceback (most recent call last):"),
                         "traceback")

    def test_lowercase_prose_is_not_an_alert(self):
        for line in ("0 errors, 0 failures", "wer improved; no error this batch",
                     "INFO: 12 clips dropped by review"):
            self.assertIsNone(tgcore.detect_error(line), line)

    def test_signature_ignores_timestamps_and_numbers(self):
        a = tgcore.error_signature(
            "2026-11-24T02:30:01Z FAIL generate.sh: pipeline exited 1")
        b = tgcore.error_signature(
            "2026-11-24T03:44:59Z FAIL generate.sh: pipeline exited 1")
        self.assertEqual(a, b)
        self.assertNotEqual(a, tgcore.error_signature(
            "FAIL publish.sh: pipeline exited 1"))


if __name__ == "__main__":
    unittest.main()
