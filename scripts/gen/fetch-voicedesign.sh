#!/bin/sh
# Fetch Qwen3-TTS-12Hz-1.7B-VoiceDesign into the project cache.
# The speech tokenizer is byte-identical to the Base checkpoint and is copied,
# not downloaded. model.safetensors goes through the parallel range fetcher.
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
# Cache is shared by every worktree, so it lives at the main repo root even
# when this script runs inside one. cache-env.sh would point at the worktree.
SHARED=$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")
DST="$SHARED/.cache/models/qwen3-tts-1.7b-voicedesign"
BASE="$SHARED/.cache/models/qwen3-tts-1.7b"
REPO=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
mkdir -p "$DST/speech_tokenizer"

for f in config.json generation_config.json merges.txt preprocessor_config.json \
         tokenizer_config.json vocab.json; do
  [ -s "$DST/$f" ] || curl -fsSL "https://huggingface.co/$REPO/resolve/main/$f" -o "$DST/$f"
done
for f in config.json configuration.json preprocessor_config.json; do
  [ -s "$DST/speech_tokenizer/$f" ] || cp "$BASE/speech_tokenizer/$f" "$DST/speech_tokenizer/$f"
done
[ -s "$DST/speech_tokenizer/model.safetensors" ] || \
  cp "$BASE/speech_tokenizer/model.safetensors" "$DST/speech_tokenizer/model.safetensors"

if [ ! -s "$DST/model.safetensors" ] || \
   [ "$(shasum -a 256 "$DST/model.safetensors" | cut -d' ' -f1)" != \
     391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522 ]; then
  "$ROOT/scripts/fetch-hf-file.sh" \
    "https://huggingface.co/$REPO/resolve/main/model.safetensors" \
    "$DST/model.safetensors" \
    391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522 16
fi
echo "voice-design cache ready: $DST"
