#!/usr/bin/env bash
# Optional Telegram hook, sourced by remote/generate.sh and remote/bootstrap.sh.
# Every entry point is a no-op unless TG_CHAT_ID is set (and python3 exists), so
# the pipeline behaves exactly as before when Telegram is not configured.
#
# shellcheck shell=bash

set -euo pipefail

TG_HOOK_ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

tg_hook_enabled() {
  [ -n "${TG_CHAT_ID:-}" ] || return 1
  command -v python3 >/dev/null 2>&1 || return 1
  [ -f "$TG_HOOK_ROOT/remote/tg/errorwatch.py" ] || return 1
  return 0
}

tg_hook_python() {
  if [ -x "$TG_HOOK_ROOT/.venv-remote/bin/python" ]; then
    printf '%s\n' "$TG_HOOK_ROOT/.venv-remote/bin/python"
  else
    command -v python3
  fi
}

# tg_hook_install LOG...  — watch the given files, alert on new errors.
tg_hook_install() {
  tg_hook_enabled || return 0
  TG_WATCH_PID=""
  mkdir -p "$TG_HOOK_ROOT/.cache/logs"
  nohup "$(tg_hook_python)" "$TG_HOOK_ROOT/remote/tg/errorwatch.py" "$@" \
    >>"$TG_HOOK_ROOT/.cache/logs/tg-watch.log" 2>&1 &
  TG_WATCH_PID=$!
  trap 'tg_hook_exit' EXIT
}

# tg_hook_capture LOG — tee this script's output to LOG and watch it.
tg_hook_capture() {
  local log="$1"
  mkdir -p "$(dirname "$log")"
  exec > >(tee -a "$log") 2>&1
  tg_hook_install "$log"
}

# EXIT trap: stop the watcher and report a non-zero exit once.
tg_hook_exit() {
  local code=$?
  trap - EXIT
  if [ -n "${TG_WATCH_PID:-}" ]; then
    kill "$TG_WATCH_PID" 2>/dev/null || true
  fi
  if [ "$code" -ne 0 ] && [ -n "${TG_CHAT_ID:-}" ]; then
    bash "$TG_HOOK_ROOT/remote/tg/notify.sh" \
      "$(basename "$0") exited $code on $(hostname); see the logs" ERROR || true
  fi
  exit "$code"
}
