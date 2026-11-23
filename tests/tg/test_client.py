"""Telegram client behaviour with a fake transport: 429 backoff and masking."""

from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "tg"))

import tgcore  # noqa: E402

TOKEN = "123456:AAF-not-a-real-token"


def reply(payload, status=200):
    return status, json.dumps(payload)


class Recorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, body, timeout):
        self.calls.append((url, json.loads(body.decode("utf-8"))))
        if not self.responses:
            raise AssertionError("unexpected extra request")
        return self.responses.pop(0)


class Clock:
    def __init__(self):
        self.slept = []

    def __call__(self, seconds):
        self.slept.append(seconds)


class ClientTests(unittest.TestCase):
    def make(self, responses):
        self.transport = Recorder(responses)
        self.clock = Clock()
        return tgcore.TelegramClient(TOKEN, transport=self.transport, sleep=self.clock)

    def test_success_returns_result_and_posts_json(self):
        client = self.make([reply({"ok": True, "result": {"message_id": 3}})])
        result = client.send_message(1, "hello")
        self.assertEqual(result["message_id"], 3)
        url, params = self.transport.calls[0]
        self.assertTrue(url.endswith("/sendMessage"))
        self.assertEqual(params, {"chat_id": 1, "text": "hello",
                                  "disable_web_page_preview": True})

    def test_429_honours_retry_after(self):
        client = self.make([
            reply({"ok": False, "error_code": 429, "description": "Too Many Requests",
                   "parameters": {"retry_after": 7}}, status=429),
            reply({"ok": True, "result": {"message_id": 4}}),
        ])
        self.assertEqual(client.send_message(1, "x")["message_id"], 4)
        self.assertEqual(self.clock.slept, [7])

    def test_429_is_capped(self):
        client = self.make([
            reply({"ok": False, "error_code": 429, "description": "Too Many Requests",
                   "parameters": {"retry_after": 500}}, status=429),
            reply({"ok": True, "result": True}),
        ])
        client.call("sendMessage", {})
        self.assertEqual(self.clock.slept, [30])

    def test_429_gives_up_after_attempts(self):
        responses = [reply({"ok": False, "error_code": 429, "description": "slow down",
                            "parameters": {"retry_after": 1}}, status=429)] * 3
        client = self.make(responses)
        with self.assertRaises(tgcore.TelegramError) as ctx:
            client.call("sendMessage", {})
        self.assertEqual(ctx.exception.error_code, 429)
        self.assertEqual(len(self.clock.slept), 2)

    def test_bad_request_is_not_retried(self):
        client = self.make([reply({"ok": False, "error_code": 400,
                                   "description": "Bad Request: chat not found"},
                                  status=400)])
        with self.assertRaises(tgcore.TelegramError) as ctx:
            client.call("sendMessage", {})
        self.assertIn("chat not found", str(ctx.exception))
        self.assertEqual(len(self.transport.calls), 1)

    def test_transport_failure_becomes_telegram_error(self):
        def broken(url, body, timeout):
            raise OSError("network unreachable")

        client = tgcore.TelegramClient(TOKEN, transport=broken)
        with self.assertRaises(tgcore.TelegramError):
            client.call("sendMessage", {})

    def test_token_never_appears_in_errors(self):
        client = self.make([reply({"ok": False, "error_code": 401,
                                   "description": "Unauthorized"}, status=401)])
        with self.assertRaises(tgcore.TelegramError) as ctx:
            client.call("getMe", {})
        self.assertNotIn(TOKEN, str(ctx.exception))

    def test_widget_methods_use_documented_parameters(self):
        client = self.make([
            reply({"ok": True, "result": {"message_id": 9,
                                          "widget": {"id": "777"}}}),
            reply({"ok": True, "result": True}),
        ])
        message = client.send_widget(1, "<b>x</b>", is_shared=True,
                                     public_data="{}", width=0, height=0)
        self.assertEqual(message["widget"]["id"], "777")
        client.edit_shared_widget_data("777", {"a": 1}, cache_time=60)
        method = self.transport.calls[1][0].rsplit("/", 1)[-1]
        self.assertEqual(method, "editSharedWidgetData")
        self.assertEqual(self.transport.calls[1][1]["widget_id"], "777")
        self.assertEqual(self.transport.calls[1][1]["cache_time"], 60)
        self.assertNotIn("html", self.transport.calls[1][1])


class MaskTests(unittest.TestCase):
    def test_token_replaced(self):
        self.assertNotIn(TOKEN, tgcore.mask_secret(f"url bot{TOKEN}/sendMessage", TOKEN))

    def test_token_shaped_text_replaced_without_the_secret(self):
        self.assertEqual(tgcore.mask_secret("bot123:secret/sendMessage"),
                         "bot<token>/sendMessage")

    def test_log_error_writes_the_masked_line(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tg-errors.log")
            tgcore.log_error(path, f"HTTP 429 for bot{TOKEN}", TOKEN)
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertNotIn(TOKEN, text)
            self.assertIn("HTTP 429", text)


if __name__ == "__main__":
    unittest.main()
