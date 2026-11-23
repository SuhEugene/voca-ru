"""Shell entrypoints: syntax, help, and the generate.sh dry-run contract."""

from __future__ import annotations

import os
import subprocess
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPTS = ["bootstrap.sh", "generate.sh", "watch.sh", "publish.sh", "pack.sh"]


def run(args, **kwargs):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, **kwargs)


class ShellSyntaxTests(unittest.TestCase):
    def test_bash_syntax(self):
        for name in SCRIPTS:
            path = os.path.join(REPO, "remote", name)
            result = run(["bash", "-n", path])
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")

    def test_set_euo_pipefail(self):
        for name in SCRIPTS:
            with open(os.path.join(REPO, "remote", name), encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn("set -euo pipefail", text, name)
            self.assertNotIn("read -p", text, f"{name} must not prompt interactively")

    def test_help_exits_zero(self):
        for name in ("generate.sh", "watch.sh", "publish.sh"):
            result = run(["bash", os.path.join(REPO, "remote", name), "--help"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.strip(), name)

    def test_unknown_argument_fails(self):
        result = run(["bash", os.path.join(REPO, "remote/generate.sh"), "--nope"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown argument", result.stderr)


class GenerateDryRunTests(unittest.TestCase):
    def test_dry_run_prints_the_full_command(self):
        result = run(["bash", os.path.join(REPO, "remote/generate.sh"),
                      "--dry-run", "--limit", "7", "--voices", "f_neutral",
                      "--batch-size", "8"])
        self.assertEqual(result.returncode, 0, result.stderr)
        command = result.stdout
        self.assertIn("build_corpus.py", command)
        self.assertIn("--limit 7", command)
        self.assertIn("--voices f_neutral", command)
        self.assertIn("--batch-size 8", command)
        self.assertIn("--require-lora", command)
        self.assertIn("--split all", command)
        self.assertIn("--asr-device cuda", command)

    def test_dry_run_defaults_are_the_full_run(self):
        result = run(["bash", os.path.join(REPO, "remote/generate.sh"), "--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--batch-size 8", result.stdout)
        self.assertIn("--voices f_neutral", result.stdout)
        self.assertIn("m_neutral", result.stdout)
        self.assertNotIn("--limit", result.stdout)


if __name__ == "__main__":
    unittest.main()
