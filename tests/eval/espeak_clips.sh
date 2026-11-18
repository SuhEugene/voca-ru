#!/bin/sh
# Build three 24 kHz espeak-ng clips + manifest for the end-to-end eval run.
# Usage: tests/eval/espeak_clips.sh [outdir]
set -eu
out=${1:-.cache/eval/espeak}
mkdir -p "$out/audio"
say() { # id category text
  espeak-ng -v ru -s 150 -w "$out/$1.raw.wav" "$3"
  ffmpeg -y -loglevel error -i "$out/$1.raw.wav" -ar 24000 -ac 1 "$out/audio/$1.wav"
  rm "$out/$1.raw.wav"
  printf '{"id":"%s","text":"%s","audio":"audio/%s.wav","category":"%s"}\n' \
    "$1" "$3" "$1" "$2" >> "$out/manifest.jsonl"
}
: > "$out/manifest.jsonl"
say date date "Сегодня 23 января."
say conversation conversation "Кот сидит на окне и смотрит во двор."
say question question "Сколько стоит эта книга?"
echo "$out/manifest.jsonl"
