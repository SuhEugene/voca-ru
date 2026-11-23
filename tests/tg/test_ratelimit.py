"""Rate limiting and duplicate coalescing, with a fake clock and a temp state."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "tg"))

import tgcore  # noqa: E402


class Clock:
    def __init__(self, start=1_000_000.0):
        self.value = start

    def __call__(self):
        return self.value


class RateLimiterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.clock = Clock()
        self.path = os.path.join(self.tmp.name, "notify-state.json")

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, max_per_min=3, dedupe_s=300):
        return tgcore.RateLimiter(self.path, max_per_min=max_per_min,
                                  dedupe_s=dedupe_s, now=self.clock)

    def test_signature_is_whitespace_insensitive(self):
        limiter = self.make()
        self.assertTrue(limiter.check("GPU  hot\n").send)
        decision = limiter.check("GPU hot")
        self.assertFalse(decision.send)
        self.assertEqual(decision.reason, "duplicate")

    def test_distinct_messages_are_sent(self):
        limiter = self.make()
        for index in range(3):
            self.assertTrue(limiter.check(f"message {index}").send)

    def test_fourth_message_in_a_minute_is_throttled(self):
        limiter = self.make(max_per_min=3)
        for index in range(3):
            limiter.check(f"m{index}")
        decision = limiter.check("m3")
        self.assertFalse(decision.send)
        self.assertEqual(decision.reason, "throttled")
        self.assertEqual(decision.suppressed, 1)

    def test_suppressed_count_is_reported_on_the_next_send(self):
        limiter = self.make(max_per_min=2)
        limiter.check("a")
        limiter.check("b")
        limiter.check("c")
        limiter.check("d")
        self.clock.value += 61
        decision = limiter.check("e")
        self.assertTrue(decision.send)
        self.assertEqual(decision.suppressed, 2)

    def test_window_slides(self):
        limiter = self.make(max_per_min=1)
        self.assertTrue(limiter.check("a").send)
        self.assertFalse(limiter.check("b").send)
        self.clock.value += 61
        self.assertTrue(limiter.check("c").send)

    def test_duplicate_window_expires(self):
        limiter = self.make(dedupe_s=30)
        self.assertTrue(limiter.check("same").send)
        self.assertFalse(limiter.check("same").send)
        self.clock.value += 31
        self.assertTrue(limiter.check("same").send)

    def test_state_survives_a_new_instance(self):
        limiter = self.make(max_per_min=1)
        limiter.check("a")
        again = self.make(max_per_min=1)
        self.assertFalse(again.check("b").send)


if __name__ == "__main__":
    unittest.main()
