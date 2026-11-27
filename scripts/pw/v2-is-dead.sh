#!/bin/bash
# Watchdog for the second-corpus run: 0 while the pipeline is alive or producing,
# 100 when it died. Mirrors is_dead.sh but for the v2 stage names.
#   v2-is-dead.sh [max_idle_minutes]
set -u
idle_min=${1:-20}
marker=/tmp/pw/v2-dead.last

snapshot() {
    cd /workspace/voca || exit 1
    local raw accept feats ann step
    raw=$([ -f data/corpus-v2/raw.jsonl ] && wc -l < data/corpus-v2/raw.jsonl || echo 0)
    accept=$([ -f data/corpus-v2/manifest.jsonl ] && wc -l < data/corpus-v2/manifest.jsonl || echo 0)
    feats=$(find data/ru-corpus-v2/prepared -name '*.pt' 2>/dev/null | wc -l)
    ann=$([ -f data/ru-corpus-v2/annotations/train.jsonl ] && wc -l < data/ru-corpus-v2/annotations/train.jsonl || echo 0)
    step=$(grep -ao '"step": *[0-9]*' /workspace/train.log 2>/dev/null | tail -1 | grep -o '[0-9]*$')
    echo "$raw $accept $feats $ann ${step:-0}"
}

state=$(snapshot)
training=0; pgrep -f "[t]raining/train.py" >/dev/null || training=1
working=0; pgrep -f "[v]2-corpus.sh|[v]2-train.sh|[s]plit_corpus.py|[p]repare.py|[a]nnotate.py" >/dev/null || working=1

if [ "$training" -eq 0 ] || [ "$working" -eq 0 ]; then
    echo "$state $(date +%s)" > "$marker"
    echo "0 v2-dead"
    exit 0
fi

if [ -f /workspace/v2-complete ]; then
    echo "0 v2-dead"      # finished on purpose
    exit 0
fi

prev=$(cat "$marker" 2>/dev/null || echo "none 0")
if [ "${prev% * * * * *}" != "$state" ]; then
    echo "$state $(date +%s)" > "$marker"
    echo "0 v2-dead"
    exit 0
fi

idle=$(( ($(date +%s) - ${prev##* }) / 60 ))
if [ "$idle" -ge "$idle_min" ]; then echo "100 v2-dead"; else echo "0 v2-dead"; fi
