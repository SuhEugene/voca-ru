#!/bin/bash
# Watchdog for the fine-tune: prints 0 while training is alive or progressing,
# 100 when it died, so pw fires a notification.
#   is_dead.sh [max_idle_minutes]
# 100 means: no training process AND no supervisor chain, or a live chain that
# has not advanced the step within the idle window.
set -u
idle_min=${1:-15}
log=/workspace/train.log
marker=/tmp/pw/is_dead.last

step=$(grep -ao '"step": *[0-9]*' "$log" 2>/dev/null | tail -1 | grep -o '[0-9]*$')
step=${step:-0}

if grep -aq "STAGE_DONE stage2" "$log" 2>/dev/null; then
    echo "0 is_dead"          # finished on purpose, not dead
    exit 0
fi

training=0; pgrep -f "[t]raining/train.py" >/dev/null || training=1
chain=0;    pgrep -f "[g]an-finetune.sh stage" >/dev/null || chain=1

state=$(cat "$marker" 2>/dev/null || echo "none 0")
prev_step=${state%% *}
if [ "$step" != "$prev_step" ]; then
    echo "$step $(date +%s)" > "$marker"
    echo "0 is_dead"
    exit 0
fi

idle=$(( ($(date +%s) - ${state##* }) / 60 ))
# a live trainer that preloads features for minutes is not a stall
if [ "$training" -eq 0 ] && [ "$idle" -lt "$idle_min" ]; then
    echo "0 is_dead"; exit 0
fi
if [ "$chain" -eq 0 ] && [ "$idle" -lt "$idle_min" ]; then
    echo "0 is_dead"; exit 0
fi
echo "100 is_dead"
