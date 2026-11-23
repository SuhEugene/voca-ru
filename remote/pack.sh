#!/usr/bin/env bash
# Build out/remote-bundle.tar.zst (run locally, from anywhere in the repo).
# Contains the scripts, the text frontend, the scripts corpus, the voice
# references and the configs — never models, audio caches or .cache/.
#
#   bash remote/pack.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GIT_COMMON="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
SHARED="$ROOT"
if [ -n "$GIT_COMMON" ]; then
  SHARED="$(dirname "$GIT_COMMON")"
fi
STAGE="$ROOT/out/remote-bundle"
OUT="$ROOT/out/remote-bundle.tar.zst"

step() { printf '\n== %s\n' "$*"; }

step "1/5 staging $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE"/{remote/lib,scripts/gen,training/voca_ru,data/scripts-ru,assets/voices,configs}

cp "$ROOT"/remote/*.sh "$ROOT"/remote/*.md "$ROOT"/remote/assets.lock \
   "$ROOT"/remote/requirements-remote.txt "$STAGE/remote/"
cp "$ROOT"/remote/lib/*.py "$STAGE/remote/lib/"
cp "$ROOT"/remote/bundle-README.md "$STAGE/README.md"
cp "$ROOT"/scripts/gen/build_corpus.py "$ROOT"/scripts/gen/screen.py \
   "$ROOT"/scripts/gen/README.md "$STAGE/scripts/gen/"
cp -R "$ROOT"/training/voca_ru/. "$STAGE/training/voca_ru/"
find "$STAGE/training" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$ROOT"/data/scripts-ru/batch1.jsonl "$ROOT"/data/scripts-ru/batch2.jsonl \
   "$ROOT"/data/scripts-ru/SPLITS.sha256 "$ROOT"/data/scripts-ru/sources.md \
   "$STAGE/data/scripts-ru/"
cp "$ROOT"/configs/*.yaml "$STAGE/configs/"
cp "$ROOT"/assets/voices/README.md "$ROOT"/assets/voices/f_neutral.txt \
   "$ROOT"/assets/voices/m_neutral.txt "$STAGE/assets/voices/"

for voice in f_neutral m_neutral; do
  if [ -f "$ROOT/assets/voices/$voice.wav" ]; then
    cp "$ROOT/assets/voices/$voice.wav" "$STAGE/assets/voices/"
  elif [ -f "$SHARED/out/g1/voices/$voice.wav" ]; then
    # the wavs are git-ignored and live at the main repo root (shared worktrees)
    cp "$SHARED/out/g1/voices/$voice.wav" "$STAGE/assets/voices/"
  elif [ -f "$ROOT/out/g1/voices/$voice.wav" ]; then
    cp "$ROOT/out/g1/voices/$voice.wav" "$STAGE/assets/voices/"
  else
    printf 'FAIL %s.wav missing (assets/voices/ and %s/out/g1/voices/)\n' "$voice" "$SHARED" >&2
    printf '     recreate: .cache/venvs/r4/bin/python scripts/gen/teacher.py design\n' >&2
    exit 1
  fi
done

step "2/5 checking the staging file list"
python3 "$ROOT/remote/lib/bundle_files.py" --check-stage "$STAGE"

step "3/5 packing"
if ! command -v zstd >/dev/null 2>&1; then
  printf 'FAIL zstd not found; install it (brew install zstd)\n' >&2
  exit 1
fi
mkdir -p "$ROOT/out"
COPYFILE_DISABLE=1 tar -C "$STAGE" -cf - . | zstd -12 -T0 -q -f -o "$OUT"

step "4/5 size"
SIZE_MB=$(du -m "$OUT" | awk '{print $1}')
printf 'bundle: %s (%s MB, %s files)\n' "$OUT" "$SIZE_MB" "$(find "$STAGE" -type f | wc -l | tr -d ' ')"
if [ "$SIZE_MB" -gt 50 ]; then
  printf 'WARN bundle is larger than 50 MB; check for stray models or audio\n' >&2
fi

step "5/5 sha256"
python3 - "$OUT" <<'PY'
import hashlib, sys
path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
print(digest.hexdigest(), path)
PY

printf '\nPACK PASS — upload %s to the GPU box, then follow README.md inside.\n' "$OUT"
