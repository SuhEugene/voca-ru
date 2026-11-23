"""Bot commands and widget lifecycle with a fake client; no network."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "tg"))

import tgcore  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "status_bot", os.path.join(REPO, "remote", "tg", "status-bot.py"))
status_bot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status_bot)

STATUS = {
    "updated_at": "2026-11-24T12:00:00Z",
    "run_started_at": "2026-11-24T09:00:00Z",
    "clips_done": 7552, "clips_expected": 16000, "progress_pct": 47.2,
    "clips_per_min": 39.2, "wer_mean": 0.0124, "wer_p95": 0.031,
    "dropped": 12, "stress_fail_pct": 0.21, "gpu_temp_c": 71,
    "vram_used_gb": 21.3, "disk_free_gb": 38.4, "last_clip_age_s": 42.0,
}


class Clock:
    def __init__(self):
        self.value = 1_760_000_000.0

    def __call__(self):
        return self.value


class FakeClient:
    def __init__(self):
        self.calls = []
        self.widget_supported = True
        self.edit_error = None
        self.next_id = 100

    def send_widget(self, chat_id, html, **extra):
        self.calls.append(("sendWidget", chat_id, extra))
        if not self.widget_supported:
            raise tgcore.TelegramError("sendWidget", 400, "Bad Request: no widgets")
        return {"message_id": 7, "widget": {"id": "777"}}

    def edit_shared_widget_data(self, widget_id, public_data, **extra):
        self.calls.append(("editSharedWidgetData", widget_id, extra))
        if self.edit_error:
            raise self.edit_error
        return True

    def send_message(self, chat_id, text, **extra):
        self.calls.append(("sendMessage", chat_id, text))
        self.next_id += 1
        return {"message_id": self.next_id}

    def edit_message_text(self, chat_id, message_id, text, **extra):
        self.calls.append(("editMessageText", chat_id, message_id))
        if self.edit_error:
            raise self.edit_error
        return True


class BotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = os.path.join(self.tmp.name, "repo")
        corpus = os.path.join(root, "data", "corpus-ru")
        os.makedirs(corpus)
        with open(os.path.join(corpus, "status.json"), "w", encoding="utf-8") as handle:
            json.dump(STATUS, handle)
        with open(os.path.join(corpus, "run-stats.json"), "w", encoding="utf-8") as handle:
            json.dump({"clips_per_min": 39.2, "wer_mean": 0.0124}, handle)
        os.makedirs(os.path.join(root, ".cache"))
        with open(os.path.join(root, ".cache", "runpod-pod.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"cost": 0.34, "created": "2026-11-24T09:00:00Z"}, handle)
        self.root = root
        self.state_path = os.path.join(self.tmp.name, "state.json")
        self.env = {"VOCA_ROOT": root, "TG_STATE_DIR": self.tmp.name,
                    "TG_STATE": self.state_path}
        self.clock = Clock()
        self.client = FakeClient()

    def tearDown(self):
        self.tmp.cleanup()

    def bot(self):
        return status_bot.Bot(env=self.env, client=self.client,
                              state=tgcore.BotState(self.state_path),
                              now=self.clock)

    def message(self, text, chat_id=5):
        return {"update_id": 1, "message": {"chat": {"id": chat_id}, "text": text}}

    def test_chatid_stores_the_chat(self):
        bot = self.bot()
        bot.handle(self.message("/chatid"))
        self.assertEqual(bot.state.chat_id, 5)
        kind, chat_id, text = self.client.calls[0]
        self.assertEqual(kind, "sendMessage")
        self.assertIn("chat id: 5", text)

    def test_status_creates_a_shared_widget_and_sets_cache_time(self):
        bot = self.bot()
        bot.handle(self.message("/status"))
        kinds = [call[0] for call in self.client.calls]
        self.assertEqual(kinds, ["sendWidget", "editSharedWidgetData"])
        self.assertTrue(self.client.calls[0][2]["is_shared"])
        self.assertEqual(self.client.calls[0][2]["public_data"][:1], "{")
        self.assertEqual(self.client.calls[1][2]["cache_time"], 60)
        record = bot.state.widget(5)
        self.assertEqual(record["kind"], "widget")
        self.assertEqual(record["widget_id"], "777")
        self.assertTrue(record["active"])

    def test_widget_id_survives_a_restart(self):
        self.bot().handle(self.message("/status"))
        reloaded = tgcore.BotState(self.state_path)
        self.assertEqual(reloaded.widget(5)["widget_id"], "777")
        bot = self.bot()
        bot.handle(self.message("/status"))
        kinds = [call[0] for call in self.client.calls]
        self.assertEqual(kinds.count("sendWidget"), 1)  # reused, not re-created

    def test_stop_disables_refresh(self):
        bot = self.bot()
        bot.handle(self.message("/status"))
        bot.handle(self.message("/stop"))
        record = bot.state.widget(5)
        self.assertFalse(record["active"])
        self.clock.value += 120
        before = len(self.client.calls)
        bot.refresh_due()
        self.assertEqual(len(self.client.calls), before)

    def test_refresh_waits_sixty_seconds(self):
        bot = self.bot()
        bot.handle(self.message("/status"))
        sent = len(self.client.calls)
        bot.refresh_due()
        self.assertEqual(len(self.client.calls), sent)  # fresh
        self.clock.value += 61
        bot.refresh_due()
        self.assertEqual(self.client.calls[-1][0], "editSharedWidgetData")

    def test_widget_unavailable_falls_back_to_a_message(self):
        self.client.widget_supported = False
        bot = self.bot()
        bot.handle(self.message("/status"))
        kinds = [call[0] for call in self.client.calls]
        self.assertEqual(kinds, ["sendWidget", "sendMessage"])
        self.assertEqual(bot.state.widget(5)["kind"], "message")

    def test_message_fallback_edits_the_same_message(self):
        self.client.widget_supported = False
        bot = self.bot()
        bot.handle(self.message("/status"))
        message_id = bot.state.widget(5)["message_id"]
        self.clock.value += 61
        bot.refresh_due()
        kind, chat_id, edited_id = self.client.calls[-1]
        self.assertEqual((kind, chat_id, edited_id),
                         ("editMessageText", 5, message_id))

    def test_not_modified_is_not_an_error(self):
        self.client.widget_supported = False
        bot = self.bot()
        bot.handle(self.message("/status"))
        self.client.edit_error = tgcore.TelegramError(
            "editMessageText", 400, "Bad Request: message is not modified")
        self.clock.value += 61
        count = len(self.client.calls)
        bot.refresh_due()
        self.assertEqual(len(self.client.calls), count + 1)  # attempted once, no resend
        self.assertIn("editMessageText", [call[0] for call in self.client.calls])
        self.assertEqual(bot.state.widget(5)["kind"], "message")

    def test_deleted_widget_is_resent(self):
        bot = self.bot()
        bot.handle(self.message("/status"))
        sent = self.client.calls[0][2]["public_data"]
        self.assertTrue(sent)
        self.client.edit_error = tgcore.TelegramError(
            "editSharedWidgetData", 400, "Bad Request: message to edit not found")
        self.clock.value += 61
        bot.refresh_due()
        kinds = [call[0] for call in self.client.calls]
        self.assertEqual(kinds.count("sendWidget"), 2)  # a fresh widget was sent

    def test_unknown_command_shows_help(self):
        bot = self.bot()
        bot.handle(self.message("/nope"))
        self.assertIn("/status", self.client.calls[-1][2])

    def test_render_cli_reads_the_corpus(self):
        text = status_bot.status_text(self.env)
        self.assertIn("47.2%", text)
        self.assertIn("$0.34/h", text)


if __name__ == "__main__":
    unittest.main()
