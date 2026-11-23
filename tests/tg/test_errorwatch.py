"""Log tailing: offsets, rotation, dedupe, context."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "tg"))

import errorwatch  # noqa: E402


class FakeNotifier:
    enabled = True

    def __init__(self):
        self.sent = []

    def send(self, text, level, **extra):
        self.sent.append((text, level, extra))
        return True


class Clock:
    def __init__(self):
        self.value = 1_000_000.0

    def __call__(self):
        return self.value


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.tmp.name, "run.log")
        self.state = os.path.join(self.tmp.name, "watch-state.json")
        self.clock = Clock()

    def tearDown(self):
        self.tmp.cleanup()

    def append(self, *lines):
        with open(self.log, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def make(self):
        watcher = errorwatch.ErrorWatcher([self.log], state_path=self.state,
                                          dedupe_s=600, now=self.clock)
        watcher.notifier = FakeNotifier()
        return watcher

    def test_offsets_survive_a_restart(self):
        self.append("clips=1 ok", "FAIL generate.sh: pipeline exited 1")
        watcher = self.make()
        watcher.poll()
        self.assertEqual(len(watcher.notifier.sent), 1)

        restarted = self.make()
        restarted.poll()
        self.assertEqual(restarted.notifier.sent, [])  # not replayed

        self.append("ERROR: another problem")
        restarted.poll()
        self.assertEqual(len(restarted.notifier.sent), 1)

    def test_rotation_reads_the_new_file_from_the_start(self):
        self.append("FAIL old run")
        watcher = self.make()
        watcher.poll()
        with open(self.log, "w", encoding="utf-8") as handle:
            handle.write("ALARM: no new clip for 21.0 min (limit 20)\n")
        watcher.poll()
        self.assertEqual(len(watcher.notifier.sent), 2)
        self.assertEqual(watcher.notifier.sent[1][1], "WARN")

    def test_partial_line_is_not_consumed(self):
        with open(self.log, "w", encoding="utf-8") as handle:
            handle.write("FAIL generate.sh: pipeline exit")
        watcher = self.make()
        watcher.poll()
        self.assertEqual(watcher.notifier.sent, [])
        self.append("ed 1")
        watcher.poll()
        self.assertEqual(len(watcher.notifier.sent), 1)

    def test_repeats_are_coalesced(self):
        for _ in range(3):
            self.append("ERROR: root: adapter checkpoint is missing")
        watcher = self.make()
        watcher.poll()
        self.assertEqual(len(watcher.notifier.sent), 1)

    def test_dedupe_expires(self):
        self.append("ERROR: same failure")
        watcher = self.make()
        watcher.poll()
        self.clock.value += 601
        self.append("ERROR: same failure")
        watcher.poll()
        self.assertEqual(len(watcher.notifier.sent), 2)

    def test_alert_carries_at_most_15_lines_of_context(self):
        self.append(*[f"context line {i}" for i in range(30)])
        self.append("FAIL generate.sh: pipeline exited 1")
        watcher = self.make()
        watcher.poll()
        summary, level, extra = watcher.notifier.sent[0]
        self.assertEqual(level, "ERROR")
        self.assertIn("FAIL generate.sh", summary)
        context = extra["context"].splitlines()
        self.assertEqual(len(context), 15)
        self.assertEqual(context[-1], "FAIL generate.sh: pipeline exited 1")
        self.assertTrue(extra["host"])

    def test_healthy_lines_send_nothing(self):
        self.append("2026-11-24T02:30:01Z clips=120/16000 rate=11.4/min "
                    "wer=0.0067 dropped=0 stress=0.0% temp=71C")
        watcher = self.make()
        watcher.poll()
        self.assertEqual(watcher.notifier.sent, [])

    def test_missing_log_is_waited_for(self):
        watcher = self.make()
        watcher.poll()
        self.append("FAIL after the log appeared")
        watcher.poll()
        self.assertEqual(len(watcher.notifier.sent), 1)


if __name__ == "__main__":
    unittest.main()
