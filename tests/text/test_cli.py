import json
import os
import subprocess
import sys
import tempfile
import unittest

import _support


class CliTest(unittest.TestCase):
    def test_adds_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "in.jsonl")
            target = os.path.join(folder, "out.jsonl")
            with open(source, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "a", "text": "Сегодня 23 января."}) + "\n")
                handle.write(json.dumps({"id": "b", "text": "Привет, как дела?"}) + "\n")
            env = dict(os.environ, PYTHONPATH=str(_support.TRAINING))
            result = subprocess.run(
                [sys.executable, "-m", "voca_ru.text", "--in", source,
                 "--out", target, "--no-ruaccent"],
                cwd=str(_support.ROOT), env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(target, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(len(rows), 2)
            for row in rows:
                for key in ("norm_text", "stressed_text", "phonemes", "tokens", "oov"):
                    self.assertIn(key, row)
                self.assertIsInstance(row["tokens"], list)
                self.assertTrue(all(isinstance(token, int) for token in row["tokens"]))
                self.assertTrue(all(0 <= token < 76 for token in row["tokens"]))
                self.assertEqual(row["norm_text"], row["stressed_text"])
                self.assertIn("sentences=2", result.stderr)


if __name__ == "__main__":
    unittest.main()
