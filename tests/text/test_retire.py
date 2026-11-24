"""Retiring sentences that produced no usable audio (F8).

``scripts/gen/retire.py`` writes data/scripts-ru/retired.jsonl; ``load_rows`` in
scripts/gen/build_corpus.py must then never hand that sentence to the teacher.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gen"))

import build_corpus  # noqa: E402
import retire  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


GOOD = "Сегодня хорошая погода."
BAD = "Плохая фраза не прошла проверку."


class RetireTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.corpus = self.tmp / "corpus-raw"
        self.corpus.mkdir()
        write_jsonl(self.corpus / "manifest.jsonl", [
            {"id": "aaa_f", "text_id": "aaa", "voice": "f_neutral", "text": GOOD},
            {"id": "aaa_m", "text_id": "aaa", "voice": "m_neutral", "text": GOOD},
        ])
        write_jsonl(self.corpus / "review.jsonl", [
            {"id": "bbb_f", "text_id": "bbb", "voice": "f_neutral", "text": BAD,
             "reject_reasons": ["wer 0.31 > 0.15", "stress error"]},
            {"id": "bbb_m", "text_id": "bbb", "voice": "m_neutral", "text": BAD,
             "reject_reasons": ["duration ratio"]},
        ])
        self.out = self.tmp / "retired.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_retired_contents_and_summary(self):
        summary = retire.retire(str(self.corpus), str(self.out))
        self.assertEqual(read_jsonl(self.out), [{
            "id": "bbb", "text": BAD,
            "voices_rejected": ["f_neutral", "m_neutral"],
            "reasons": ["duration ratio", "stress error", "wer 0.31 > 0.15"],
        }])
        self.assertEqual(summary, {"retired": 1, "partial": 0, "healthy": 1, "voices": 2})

    def test_partial_voice_is_not_retired(self):
        write_jsonl(self.corpus / "manifest.jsonl", [
            {"id": "aaa_f", "text_id": "aaa", "voice": "f_neutral", "text": GOOD},
        ])
        summary = retire.retire(str(self.corpus), str(self.out))
        self.assertEqual([row["id"] for row in read_jsonl(self.out)], ["bbb"])
        self.assertEqual(summary["partial"], 1)

    def test_load_rows_skips_retired_only_when_file_exists(self):
        sources = self.tmp / "batch.jsonl"
        write_jsonl(sources, [
            {"id": "aaa", "text": GOOD, "split": "train"},
            {"id": "bbb", "text": BAD, "split": "train"},
        ])
        old = build_corpus.RETIRED
        build_corpus.RETIRED = str(self.out)
        try:
            self.assertEqual([r["id"] for r in build_corpus.load_rows([str(sources)])],
                             ["aaa", "bbb"])
            retire.retire(str(self.corpus), str(self.out))
            self.assertEqual([r["id"] for r in build_corpus.load_rows([str(sources)])],
                             ["aaa"])
        finally:
            build_corpus.RETIRED = old


if __name__ == "__main__":
    unittest.main()
