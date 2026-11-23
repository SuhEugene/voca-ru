#!/usr/bin/env bash
# Verify the finished corpus, write the dataset card, upload it to Hugging Face.
# Refuses to upload if verification fails or HF_TOKEN is unset.
#
#   bash remote/publish.sh --check          # verify + print statistics
#   bash remote/publish.sh --dry-run        # verify + write data/corpus-ru/README.md
#   HF_TOKEN=hf_... bash remote/publish.sh  # upload
#   HF_REPO=me/other-corpus bash remote/publish.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CORPUS="${CORPUS:-$ROOT/data/corpus-ru}"
REPO="${HF_REPO:-SuhEugene/voca-ru-corpus}"
ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --check)         ARGS+=(--check); shift ;;
    --dry-run)       ARGS+=(--dry-run); shift ;;
    --allow-partial) ARGS+=(--allow-partial); shift ;;
    --public)        ARGS+=(--public); shift ;;
    --corpus)        CORPUS="${2:?--corpus needs a path}"; shift 2 ;;
    --repo)          REPO="${2:?--repo needs owner/name}"; shift 2 ;;
    -h|--help)       sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [ ! -f "$ROOT/remote/bundle.env" ]; then
  printf 'FAIL remote/bundle.env is missing; run: bash remote/bootstrap.sh\n' >&2
  exit 1
fi
# shellcheck disable=SC1091
. "$ROOT/remote/bundle.env"

exec "$ROOT/.venv-remote/bin/python" "$ROOT/remote/lib/publish.py" \
  --root "$ROOT" --corpus "$CORPUS" --repo "$REPO" "${ARGS[@]}"
