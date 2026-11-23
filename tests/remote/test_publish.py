"""Publish gate: a corpus with expected drops must still count as complete."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "lib"))

import publish  # noqa: E402


class CompletenessTests(unittest.TestCase):
    def test_drops_count_towards_completion(self):
        self.assertTrue(publish.is_complete({"clips": 15500, "dropped": 500}, 16000))

    def test_truncated_corpus_is_incomplete(self):
        self.assertFalse(publish.is_complete({"clips": 9000, "dropped": 100}, 16000))

    def test_unknown_expected_does_not_block(self):
        self.assertTrue(publish.is_complete({"clips": 1, "dropped": 0}, 0))


if __name__ == "__main__":
    unittest.main()
