#!/usr/bin/env python3
"""One-shot Telegram alert used by remote/tg/notify.sh.

    notify.py "text" [ERROR|WARN|INFO]
    notify.py --ping

Always exits 0 for delivery problems: a Telegram outage must never break the
pipeline.  With TG_CHAT_ID (and a token) unset it is silently disabled.
"""

from __future__ import annotations

import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tgcore  # noqa: E402


def usage():
    return __doc__.strip()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    env = os.environ

    if "--help" in argv or "-h" in argv:
        print(usage())
        return 0

    if "--ping" in argv:
        # Explicitly allowed one-off connectivity check.  Only when the bot has
        # already saved a chat id from /start; never invents a recipient.
        chat_id = tgcore.BotState().chat_id
        token = tgcore.read_token(env)
        if not chat_id or not token:
            print("[tg] ping skipped: no chat id in remote/tg/state.json "
                  "(send /start to the bot first)", file=sys.stderr)
            return 0
        client = tgcore.TelegramClient(token)
        try:
            client.send_message(chat_id, "connected \u2705 " + socket.gethostname())
        except tgcore.TelegramError as exc:
            tgcore.log_error(os.path.join(tgcore.state_dir(env), "tg-errors.log"),
                             str(exc), token)
            print(f"[tg] ping failed: {exc}", file=sys.stderr)
        return 0

    level = "INFO"
    words = [arg for arg in argv if not arg.startswith("--")]
    if len(words) >= 2 and words[-1].upper() in tgcore.LEVEL_ICON:
        level = words.pop().upper()
    text = " ".join(words).strip()
    if not text:
        print(usage(), file=sys.stderr)
        return 2

    notifier = tgcore.Notifier(env=env)
    if not notifier.enabled:
        if notifier.chat_id:
            print("[tg] chat id set but no token; alert not sent", file=sys.stderr)
        return 0  # silently disabled unless a chat was configured without a token

    # The shell already prefixes hostname/level; keep the message exactly as asked.
    if not notifier.send(text, level):
        print("[tg] alert not sent (duplicate, throttled, or Telegram unreachable)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
