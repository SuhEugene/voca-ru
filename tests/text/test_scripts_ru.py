"""Contract tests for the frozen Russian script corpus (F2, docs/contracts.md §5-6)."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "scripts-ru"
BATCH1 = DATA / "batch1.jsonl"
BATCH2 = DATA / "batch2.jsonl"
SPLITS = DATA / "SPLITS.sha256"

FIELDS = ["id", "text", "split", "category", "origin", "source_id",
          "source_license", "source_url", "word_count", "char_count", "batch"]
ALLOWED = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
              "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
              " .,;:!?«»„“”\"()—–-")

NUMBER_WORDS = (
    "один|одна|одно|два|две|три|четыре|пять|шесть|семь|восемь|девять|"
    "десять|одиннадцать|двенадцать|тринадцать|четырнадцать|пятнадцать|"
    "шестнадцать|семнадцать|восемнадцать|девятнадцать|двадцать|тридцать|"
    "сорок|пятьдесят|шестьдесят|семьдесят|восемьдесят|девяносто|"
    "сто|двести|триста|четыреста|пятьсот|шестьсот|семьсот|восемьсот|"
    "девятьсот|тысяча|тысячи|тысяч"
)
MONTHS = ("январ|феврал|март|апрел|мая|июн|июл|август|сентябр|октябр|ноябр|декабр")
NAMES = ("Иван|Мари|Олег|Анн|Пётр|Петр|Софь|Дмитри|Елен|Серге|Ольг|Алексе|"
         "Татьян|Михаил|Ирин|Никола|Наталь|Павел|Павл|Ксени|Андре|Вер[аыуе]|"
         "Максим|Юли|Борис|Нин|Артём|Артем|Дарь")


def norm_key(text: str) -> str:
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[^а-я\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ScriptsRuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch1 = load(BATCH1)
        cls.batch2 = load(BATCH2)
        cls.rows = cls.batch1 + cls.batch2

    def test_counts_and_splits(self):
        self.assertEqual(len(self.batch1), 4500)
        self.assertEqual(len(self.batch2), 3500)
        splits = {}
        for row in self.batch1:
            splits[row["split"]] = splits.get(row["split"], 0) + 1
        self.assertEqual(splits, {"train": 3870, "val": 315, "test": 315})
        self.assertTrue(all(row["split"] == "train" for row in self.batch2))

    def test_fields(self):
        for row in self.rows:
            self.assertEqual(list(row), FIELDS)
            self.assertIsInstance(row["id"], str)
            self.assertEqual(len(row["id"]), 16)
            self.assertIsInstance(row["text"], str)
            for key in ("category", "origin", "source_id", "source_license",
                        "source_url"):
                self.assertTrue(row[key], key)
            self.assertIsInstance(row["word_count"], int)
            self.assertIsInstance(row["char_count"], int)
            self.assertIn(row["batch"], (1, 2))

    def test_filter_rules(self):
        for row in self.rows:
            text = row["text"]
            words = text.split()
            self.assertGreaterEqual(len(words), 5, text)
            self.assertLessEqual(len(words), 30, text)
            self.assertTrue(text[-1] in ".!?", text)
            self.assertEqual(row["word_count"], len(words), text)
            self.assertEqual(row["char_count"], len(text), text)
            self.assertTrue(text[0].isupper(), text)
            for ch in text:
                self.assertIn(ch, ALLOWED, f"{ch!r} in {text!r}")
            self.assertFalse(re.search(r"[A-Za-z0-9]", text), text)
            self.assertNotIn("..", text)
            self.assertEqual(text.count("«"), text.count("»"), text)

    def test_ids_and_duplicates(self):
        ids = [row["id"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        keys = [norm_key(row["text"]) for row in self.rows]
        self.assertEqual(len(keys), len(set(keys)))
        for row in self.rows:
            self.assertEqual(row["id"], hashlib.sha256(norm_key(row["text"]).encode()).hexdigest()[:16])

    def test_no_sentence_in_two_splits(self):
        splits = {}
        for row in self.rows:
            splits.setdefault(norm_key(row["text"]), set()).add(row["split"])
        for key, seen in splits.items():
            self.assertEqual(len(seen), 1, f"{key}: {seen}")
        self.assertTrue(all(row["split"] == "train" for row in self.batch2))

    def test_origins_and_licences(self):
        by_origin = {}
        for row in self.rows:
            by_origin[row["origin"]] = by_origin.get(row["origin"], 0) + 1
        self.assertEqual(by_origin["common_voice_sentence_collector"], 3000)
        self.assertEqual(by_origin["wikisource_pd"], 900)
        self.assertEqual(by_origin["project_llm"], 1500)
        self.assertEqual(by_origin["project_templates"], 2600)
        for row in self.rows:
            if row["origin"] in ("common_voice_sentence_collector", "project_llm",
                                 "project_templates"):
                self.assertEqual(row["source_license"], "cc0-1.0")
            if row["origin"] == "wikisource_pd":
                self.assertEqual(row["source_license"], "public-domain")

    def test_hard_cases_present(self):
        texts = [row["text"] for row in self.rows]
        numbers = sum(1 for t in texts if re.search(rf"\b({NUMBER_WORDS})\b", t, re.I))
        dates_times = sum(1 for t in texts
                          if re.search(MONTHS, t, re.I) or re.search(r"\bчас", t, re.I))
        money = sum(1 for t in texts if re.search(r"рубл|копе|доллар|евро", t, re.I))
        questions = sum(1 for t in texts if t.endswith("?"))
        names = sum(1 for t in texts if re.search(rf"\b({NAMES})", t))
        self.assertGreaterEqual(numbers, 400)
        self.assertGreaterEqual(dates_times, 200)
        self.assertGreaterEqual(money, 150)
        self.assertGreaterEqual(questions, 200)
        self.assertGreaterEqual(names, 100)

    def test_frozen_split_manifest(self):
        lines = SPLITS.read_text(encoding="utf-8").splitlines()
        hashes = {}
        for line in lines:
            if line.startswith("#") or line.startswith("batch") and "=" in line:
                continue
            name, value = line.split()
            hashes[name] = value.split(":", 1)[1]
        self.assertEqual(hashes["batch1.jsonl"],
                         hashlib.sha256(BATCH1.read_bytes()).hexdigest())
        self.assertEqual(hashes["batch2.jsonl"],
                         hashlib.sha256(BATCH2.read_bytes()).hexdigest())
        manifest = "\n".join(f"{row['id']} {row['split']}"
                             for row in sorted(self.rows, key=lambda r: r["id"]))
        self.assertEqual(hashes["split-manifest"],
                         hashlib.sha256(manifest.encode()).hexdigest())


if __name__ == "__main__":
    unittest.main()
