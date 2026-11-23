#!/usr/bin/env python3
"""Long-polling Telegram bot: /status posts a live widget (or an edited message).

Commands: /start, /chatid, /status, /stop, /help.

The widget follows the Bot API's shared-widget flow (sendWidget with
is_shared=True, then editSharedWidgetData every 60 s with cache_time=60).  When
the account's apps or the API do not offer widgets, it falls back to one
message per chat edited with editMessageText every 60 s.  One widget per chat,
ids persisted in remote/tg/state.json, 429 retry_after honoured by tgcore.

Standard library only; reads data/corpus-ru/{status.json,run-stats.json} and
.cache/runpod-pod.json read-only.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tgcore  # noqa: E402

HELP = (
    "corpus bot\n"
    "/status \u2014 live widget (refreshes every 60 s)\n"
    "/stop \u2014 stop refreshing it\n"
    "/chatid \u2014 show this chat id\n"
)


def load_snapshot(env):
    folder = tgcore.data_dir(env)
    status = tgcore.load_json(os.path.join(folder, "status.json"), {}) or {}
    stats = tgcore.load_json(os.path.join(folder, "run-stats.json"), {}) or {}
    pod_path = env.get("TG_POD_JSON") or os.path.join(
        env.get("VOCA_ROOT") or tgcore.REPO, ".cache", "runpod-pod.json")
    pod = tgcore.load_json(pod_path, {}) or {}
    return status, stats, pod


def status_text(env, now=None):
    status, stats, pod = load_snapshot(env)
    return tgcore.format_status(status, stats=stats, pod=pod, now=now)


def status_payload(env, now=None):
    status, stats, pod = load_snapshot(env)
    return tgcore.widget_public_data(status, stats=stats, pod=pod, now=now)


class Bot:
    def __init__(self, *, env=None, client=None, state=None, now=time.time,
                 sleep=time.sleep):
        self.env = env or os.environ
        self.now = now
        self.sleep = sleep
        self.state = state or tgcore.BotState()
        self.token = tgcore.read_token(self.env)
        self.client = client or (
            tgcore.TelegramClient(self.token, base_url=self.env.get("TG_API_BASE")
                                  or "https://api.telegram.org")
            if self.token else None)
        self.error_log = os.path.join(tgcore.state_dir(self.env), "tg-errors.log")
        self.host = socket.gethostname()

    # -- messaging ---------------------------------------------------------
    def reply(self, chat_id, text):
        try:
            self.client.send_message(chat_id, text)
        except tgcore.TelegramError as exc:
            tgcore.log_error(self.error_log, str(exc), self.token)

    def _fallback(self, chat_id):
        text = status_text(self.env, self.now())
        message = self.client.send_message(chat_id, text)
        return {"kind": "message", "message_id": message.get("message_id"),
                "widget_id": None, "active": True,
                "created_at": self.now(), "refreshed_at": self.now()}

    def create_widget(self, chat_id):
        """Send a new widget, or a plain message if widgets are unavailable."""
        payload = status_payload(self.env, self.now())
        caption = "corpus \u00b7 " + time.strftime("%H:%M", time.localtime(self.now()))
        try:
            message = self.client.send_widget(
                chat_id, tgcore.widget_html(), caption=caption, is_shared=True,
                width=0, height=0, public_data=payload)
        except tgcore.TelegramError as exc:
            tgcore.log_error(self.error_log, f"widget unavailable, using message: {exc}",
                             self.token)
            return self._fallback(chat_id)

        widget = message.get("widget") or {}
        widget_id = widget.get("id") or message.get("widget_id")
        if not widget_id:
            return self._fallback(chat_id)
        try:
            self.client.edit_shared_widget_data(
                widget_id, payload, cache_time=tgcore.WIDGET_CACHE_TIME)
        except tgcore.TelegramError as exc:
            tgcore.log_error(self.error_log, f"cache_time not set: {exc}", self.token)
        return {"kind": "widget", "widget_id": widget_id,
                "message_id": message.get("message_id"), "active": True,
                "created_at": self.now(), "refreshed_at": self.now()}

    def refresh(self, chat_id, record):
        """One refresh; returns the (possibly replaced) record."""
        record = dict(record or {})
        payload = status_payload(self.env, self.now())
        try:
            if record.get("kind") == "widget" and record.get("widget_id"):
                self.client.edit_shared_widget_data(
                    record["widget_id"], payload, cache_time=tgcore.WIDGET_CACHE_TIME)
            elif record.get("message_id"):
                self.client.edit_message_text(chat_id, record["message_id"],
                                              status_text(self.env, self.now()))
            else:
                return self.create_widget(chat_id)
        except tgcore.TelegramError as exc:
            text = str(exc).lower()
            if "not modified" in text:
                pass
            elif "not found" in text or "message to edit" in text or exc.error_code == 400:
                tgcore.log_error(self.error_log, f"widget lost, resending: {exc}",
                                 self.token)
                return self.create_widget(chat_id)
            else:
                tgcore.log_error(self.error_log, str(exc), self.token)
                raise
        record["refreshed_at"] = self.now()
        return record

    # -- commands ----------------------------------------------------------
    def handle(self, update):
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        text = (message.get("text") or "").strip()
        if chat_id is None or not text.startswith("/"):
            return
        command = text.split()[0].split("@")[0].lower()

        if command in ("/start", "/chatid"):
            self.state.set_chat_id(chat_id)
            self.reply(chat_id, f"chat id: {chat_id}\n\n{HELP}")
        elif command == "/status":
            self.state.set_chat_id(chat_id)
            record = self.state.widget(chat_id)
            try:
                record = self.refresh(chat_id, record) if record and record.get("active") \
                    else self.create_widget(chat_id)
            except tgcore.TelegramError as exc:
                tgcore.log_error(self.error_log, str(exc), self.token)
                return
            self.state.set_widget(chat_id, record)
        elif command == "/stop":
            record = self.state.widget(chat_id) or {}
            record["active"] = False
            self.state.set_widget(chat_id, record)
            self.reply(chat_id, "status widget stopped")
        else:
            self.reply(chat_id, HELP)

    # -- loop --------------------------------------------------------------
    def refresh_due(self):
        changed = False
        for key, record in list((self.state.data.get("widgets") or {}).items()):
            if not record or not record.get("active"):
                continue
            if self.now() - float(record.get("refreshed_at") or 0) < tgcore.REFRESH_SECONDS:
                continue
            chat_id = int(key) if str(key).lstrip("-").isdigit() else key
            try:
                self.state.set_widget(key, self.refresh(chat_id, record))
            except tgcore.TelegramError:
                pass  # already logged; the next tick retries
            changed = True
        return changed

    def run(self):
        if not self.client:
            print("[bot] disabled: no token", flush=True)
            return 0
        print(f"[bot] started on {self.host}; {len(self.state.data.get('widgets') or {})} "
              f"saved widget(s)", flush=True)
        try:
            self.client.call("deleteWebhook", {})  # getUpdates and webhooks are exclusive
        except tgcore.TelegramError as exc:
            tgcore.log_error(self.error_log, str(exc), self.token)
        backoff = 1.0
        while True:
            try:
                self.refresh_due()
                updates = self.client.get_updates(self.state.offset, timeout=8) or []
                for update in updates:
                    self.handle(update)
                    self.state.advance(update.get("update_id"))
                backoff = 1.0
            except tgcore.TelegramError as exc:
                tgcore.log_error(self.error_log, str(exc), self.token)
                print(f"[bot] {exc}", flush=True)
                self.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--render", action="store_true",
                        help="print the status text once and exit (no network)")
    parser.add_argument("--state", default=None, help="path to state.json")
    args = parser.parse_args(argv)

    if args.render:
        print(status_text(os.environ))
        return 0

    state = tgcore.BotState(args.state) if args.state else tgcore.BotState()
    bot = Bot(state=state)
    try:
        return bot.run()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
