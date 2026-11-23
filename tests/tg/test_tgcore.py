"""Text rules: HTML escaping, the 4096-character cut, alert layout."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "tg"))

import tgcore  # noqa: E402


class EscapeTests(unittest.TestCase):
    def test_three_entities(self):
        self.assertEqual(tgcore.escape_html("<b>&</b>"), "&lt;b&gt;&amp;&lt;/b&gt;")

    def test_ampersand_is_not_double_escaped_twice(self):
        self.assertEqual(tgcore.escape_html("a &amp; b"), "a &amp;amp; b")

    def test_plain_text_untouched(self):
        text = "\u0421\u0435\u0433\u043e\u0434\u043d\u044f 71\u00b0C $0.34/h"
        self.assertEqual(tgcore.escape_html(text), text)


class TruncateTests(unittest.TestCase):
    def test_short_text_untouched(self):
        self.assertEqual(tgcore.truncate("hello", 10), "hello")

    def test_exactly_at_limit(self):
        text = "x" * 4096
        self.assertEqual(tgcore.truncate(text), text)

    def test_long_text_cut_to_limit(self):
        text = "\u044f" * 5000  # Cyrillic: two UTF-8 bytes per character
        result = tgcore.truncate(text)
        self.assertEqual(len(result), tgcore.MSG_LIMIT)
        self.assertTrue(result.endswith("\u2026"))

    def test_custom_limit(self):
        self.assertEqual(len(tgcore.truncate("abcdef", 4)), 4)


class AlertTests(unittest.TestCase):
    def test_layout_has_level_title_host(self):
        body = tgcore.format_alert("boom", "ERROR", title="run.log", host="gpu-1")
        self.assertIn("\u274c <b>ERROR</b>", body)
        self.assertIn("run.log", body)
        self.assertIn("@ gpu-1", body)
        self.assertIn("boom", body)

    def test_markup_is_escaped(self):
        body = tgcore.format_alert("<script>alert(1)</script>", "WARN")
        self.assertNotIn("<script>", body)
        self.assertIn("&lt;script&gt;", body)

    def test_context_is_a_pre_block(self):
        body = tgcore.format_alert("x", "ERROR", context="line1\nline2")
        self.assertIn("<pre>line1\nline2</pre>", body)

    def test_huge_text_stays_within_limit(self):
        body = tgcore.format_alert("y" * 9000, "ERROR")
        self.assertLessEqual(len(body), tgcore.MSG_LIMIT)

    def test_unknown_level_falls_back_to_info(self):
        body = tgcore.format_alert("x", "NOPE")
        self.assertIn("<b>NOPE</b>", body)


if __name__ == "__main__":
    unittest.main()
