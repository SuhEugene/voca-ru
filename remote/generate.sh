#!/usr/bin/env bash
# Generate the full Russian corpus (8 000 sentences x 2 voices = 16 000 clips).
# Resumable: clips already in manifest.jsonl or review.jsonl are skipped.
# Safe to kill: per-clip output is flushed as it is accepted.
#
#   bash remote/generate.sh                       # full run
#   bash remote/generate.sh --limit 8             # smoke: 8 sentences x 2 voices
#   bash remote/generate.sh --limit 40 --voices f_neutral
#   bash remote/generate.sh --dry-run             # print the exact command
#
# --limit counts SENTENCES (each is read by every selected voice).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LIMIT=0
VOICES="f_neutral,m_neutral"
BATCH="${VOCA_BATCH_SIZE:-8}"
SPLIT="all"
DRY=0

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --limit)      LIMIT="${2:?--limit needs a number}"; shift 2 ;;
    --voices)     VOICES="${2:?--voices needs a list}"; shift 2 ;;
    --batch-size) BATCH="${2:?--batch-size needs a number}"; shift 2 ;;
    --split)      SPLIT="${2:?--split needs a value}"; shift 2 ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

OUT="$ROOT/data/corpus-ru"
DONE_ACCEPTED=$( [ -f "$OUT/manifest.jsonl" ] && wc -l <"$OUT/manifest.jsonl" | tr -d ' ' || printf '0' )
DONE_DROPPED=$( [ -f "$OUT/review.jsonl" ] && wc -l <"$OUT/review.jsonl" | tr -d ' ' || printf '0' )

CMD=(
  "$ROOT/.venv-remote/bin/python" "$ROOT/scripts/gen/build_corpus.py"
  --out "$OUT"
  --sources batch1.jsonl batch2.jsonl
  --voices "$VOICES"
  --split "$SPLIT"
  --batch-size "$BATCH"
  --require-lora
  --asr-device "${VOCA_ASR_DEVICE:-cuda}"
  --asr-compute-type float16
)
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  CMD+=(--limit "$LIMIT")
fi

if [ "$DRY" -eq 1 ]; then
  printf 'cd %q && ' "$ROOT"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

if [ ! -f "$ROOT/remote/bundle.env" ]; then
  printf 'FAIL remote/bundle.env is missing; run: bash remote/bootstrap.sh\n' >&2
  exit 1
fi
# shellcheck disable=SC1091
. "$ROOT/remote/bundle.env"

if [ ! -x "$ROOT/.venv-remote/bin/python" ]; then
  printf 'FAIL .venv-remote is missing; run: bash remote/bootstrap.sh\n' >&2
  exit 1
fi

for voice in ${VOICES//,/ }; do
  if [ ! -f "$ROOT/assets/voices/$voice.wav" ] || [ ! -f "$ROOT/assets/voices/$voice.txt" ]; then
    printf 'FAIL voice reference missing: assets/voices/%s.{wav,txt}\n' "$voice" >&2
    exit 1
  fi
done

mkdir -p "$OUT"

{
  printf '\n=== generate %s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'voices=%s limit=%s split=%s batch_size=%s\n' "$VOICES" "$LIMIT" "$SPLIT" "$BATCH"
  printf 'already on disk: %s accepted, %s dropped (skipped)\n' "$DONE_ACCEPTED" "$DONE_DROPPED"
  printf 'command: '
  printf '%q ' "${CMD[@]}"
  printf '\n'
} | tee -a "$OUT/run.log"

set -o pipefail
"${CMD[@]}" 2>&1 | tee -a "$OUT/run.log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  printf 'FAIL generate.sh: pipeline exited %d (rerun to resume)\n' "$status" | tee -a "$OUT/run.log" >&2
  exit "$status"
fi
printf 'DONE generate.sh: corpus in %s\n' "$OUT" | tee -a "$OUT/run.log"
