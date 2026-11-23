#!/usr/bin/env bash
# Tail run/health logs and alert Telegram on new errors.
#
#   bash remote/tg/watch-errors.sh                       # run.log + health.log
#   bash remote/tg/watch-errors.sh /path/run.log         # one file
#   TG_INTERVAL=5 bash remote/tg/watch-errors.sh --once   # one pass (for tests)
#
# No-op when neither TG_CHAT_ID nor remote/tg/state.json provides a chat id.
# Offsets are persisted, so restarts do not replay old alerts and rotation is
# handled by errorwatch.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CORPUS="${TG_CORPUS_DIR:-$ROOT/data/corpus-ru}"
EXTRA=()

if [ "$#" -gt 0 ] && [ "$1" = "--once" ]; then
  EXTRA+=(--once)
  shift
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
  printf '[tg-watch] python3 not found; disabled\n' >&2
  exit 0
fi

if [ -z "${TG_CHAT_ID:-}" ] && [ ! -f "$HERE/state.json" ]; then
  printf '[tg-watch] disabled: no TG_CHAT_ID and no state.json\n' >&2
  exit 0
fi

if [ "$#" -eq 0 ]; then
  set -- "$CORPUS/run.log" "$CORPUS/health.log"
fi

exec "$PY" "$HERE/errorwatch.py" \
  --interval "${TG_INTERVAL:-2}" \
  --dedupe "${TG_ERROR_DEDUPE_SECONDS:-600}" \
  "${EXTRA[@]}" "$@"
