#!/usr/bin/env bash
# Health check for a running corpus generation. Read-only towards the corpus.
# Writes data/corpus-ru/status.json and appends to health.log every 5 minutes.
# Exits 2 on the first alarm unless --keep-going is given.
#
#   bash remote/watch.sh                 # loop forever, 5 min interval
#   bash remote/watch.sh --once          # one sample, non-zero on alarm
#   bash remote/watch.sh --interval 60 --keep-going
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INTERVAL="${WATCH_INTERVAL:-300}"
CORPUS="$ROOT/data/corpus-ru"
VOICES="${VOCA_VOICES:-f_neutral,m_neutral}"
EXTRA=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --once)       EXTRA+=(--once); shift ;;
    --keep-going) EXTRA+=(--keep-going); shift ;;
    --interval)   INTERVAL="${2:?--interval needs seconds}"; shift 2 ;;
    --corpus)     CORPUS="${2:?--corpus needs a path}"; shift 2 ;;
    --voices)     VOICES="${2:?--voices needs a list}"; shift 2 ;;
    -h|--help)    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [ ! -f "$ROOT/remote/bundle.env" ]; then
  printf 'FAIL remote/bundle.env is missing; run: bash remote/bootstrap.sh\n' >&2
  exit 1
fi
# shellcheck disable=SC1091
. "$ROOT/remote/bundle.env"

mkdir -p "$CORPUS"
printf '[watch] corpus=%s interval=%ss voices=%s\n' "$CORPUS" "$INTERVAL" "$VOICES"
exec "$ROOT/.venv-remote/bin/python" "$ROOT/remote/lib/health.py" \
  --corpus-dir "$CORPUS" \
  --scripts-dir "$ROOT/data/scripts-ru" \
  --voices "$VOICES" \
  --interval "$INTERVAL" \
  "${EXTRA[@]}"
