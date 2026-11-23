#!/usr/bin/env bash
# One-shot Telegram alert:  bash remote/tg/notify.sh "text" [ERROR|WARN|INFO]
#
# Silently disabled (exit 0) when TG_CHAT_ID is unset.  Any delivery problem is
# logged to remote/tg/state/tg-errors.log and still exits 0: alerts must never
# break the pipeline.  The token is read from $TG_TOKEN or
# ~/.secrets/telegram-bot-token and is never printed.
#
#   bash remote/tg/notify.sh --ping      # one connectivity message, if a chat id exists
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

if [ -z "${TG_CHAT_ID:-}" ] && [ ! -f "$HERE/state.json" ]; then
  exit 0
fi

PY="${TG_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$ROOT/.venv-remote/bin/python" ]; then
    PY="$ROOT/.venv-remote/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  fi
fi
if [ -z "$PY" ]; then
  printf '[tg] python3 not found; alert dropped\n' >&2
  exit 0
fi

# notify.py itself exits 0 on delivery failure; this guard covers a crash too.
"$PY" "$HERE/notify.py" "$@" || printf '[tg] notify.py failed (ignored)\n' >&2
exit 0
