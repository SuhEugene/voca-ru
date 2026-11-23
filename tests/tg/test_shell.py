"""Shell entry points: syntax, the disabled path, and --ping skipping."""

from __future__ import annotations

import os
import subprocess
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
TG = os.path.join(REPO, "remote", "tg")
SCRIPTS = ["notify.sh", "watch-errors.sh", "hook.sh"]


def clean_env(tmp):
    """No chat id, no token, no inherited Telegram settings."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "HOME": tmp,
           "TG_TOKEN": "",
           "TG_TOKEN_FILE": os.path.join(tmp, "missing-token"),
           "TG_STATE": os.path.join(tmp, "state.json"),
           "TG_STATE_DIR": os.path.join(tmp, "state")}
    return env


def run(args, env, cwd=REPO):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=60)


class SyntaxTests(unittest.TestCase):
    def test_bash_syntax(self):
        for name in SCRIPTS:
            result = subprocess.run(["bash", "-n", os.path.join(TG, name)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")

    def test_set_euo_pipefail(self):
        for name in SCRIPTS:
            with open(os.path.join(TG, name), encoding="utf-8") as handle:
                self.assertIn("set -euo pipefail", handle.read(), name)

    def test_python_syntax(self):
        for name in ("tgcore.py", "notify.py", "errorwatch.py", "status-bot.py"):
            path = os.path.join(TG, name)
            result = subprocess.run(["python3", "-m", "py_compile", path],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")


class DisabledTests(unittest.TestCase):
    def test_notify_silently_disabled(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result = run(["bash", os.path.join(TG, "notify.sh"), "hello", "ERROR"],
                         clean_env(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, "")

    def test_notify_with_chat_but_no_token_fails_softly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            env = clean_env(tmp)
            env["TG_CHAT_ID"] = "123"
            result = run(["bash", os.path.join(TG, "notify.sh"), "hello", "ERROR"], env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("not sent", result.stderr)

    def test_ping_skips_without_a_saved_chat(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            env = clean_env(tmp)
            env["TG_CHAT_ID"] = "123"
            result = run(["bash", os.path.join(TG, "notify.sh"), "--ping"], env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("skipped", result.stderr)

    def test_watcher_disabled(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result = run(["bash", os.path.join(TG, "watch-errors.sh"), "--once"], 
                         clean_env(tmp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("disabled", result.stderr)

    def test_pipeline_scripts_work_without_telegram(self):
        result = run(["bash", "remote/generate.sh", "--dry-run", "--limit", "2"],
                     clean_env(os.path.dirname(os.path.abspath(__file__))))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("build_corpus.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
