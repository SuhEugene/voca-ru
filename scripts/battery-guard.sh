#!/bin/sh
# Battery watchdog for long unattended runs.
# Logs level every INTERVAL seconds, warns once at WARN%, pauses heavy jobs at PAUSE%
# (SIGSTOP, resumable), resumes when charging again above RESUME%.
#
#   scripts/battery-guard.sh            # run in the background
#   scripts/battery-guard.sh --status   # print one line and exit
set -u
INTERVAL=${INTERVAL:-300}
WARN=${WARN:-25}
PAUSE=${PAUSE:-15}
RESUME=${RESUME:-30}
ROOT=/Users/suheugene/projects/22-tiniest-voca-ru
LOG=$ROOT/.cache/logs/battery.log
JOBS='build_corpus.py|teacher.py|train.py|hf download|fetch-hf-file.sh'
mkdir -p "$ROOT/.cache/logs"

read_batt() {
  pmset -g batt | awk '/InternalBattery/ {
    p = 0; ac = 0
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^[0-9]+%;?$/) { gsub(";", "", $i); gsub("%", "", $i); p = $i + 0 }
    }
    if ($0 ~ /AC Power|AC attached/) ac = 1
    print p, ac
  }'
}

notify() {
  osascript -e "display notification \"$1\" with title \"voca-ru: батарея\"" >/dev/null 2>&1
}

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

state=$(read_batt | awk '{print ($2 == 1) ? "running" : "paused"}')
last_warn=-100

if [ "${1:-}" = "--status" ]; then
  read_batt | while read -r pct ac; do
    echo "battery ${pct}% ac=${ac} threshold warn=${WARN} pause=${PAUSE}"
  done
  exit 0
fi

log "guard started interval=${INTERVAL}s warn=${WARN}% pause=${PAUSE}%"
while :; do
  read -r pct ac <<EOF
$(read_batt)
EOF
  [ -n "${pct:-}" ] || { sleep "$INTERVAL"; continue; }
  log "level ${pct}% ac=${ac} state=${state}"

  if [ "$ac" = "1" ]; then
    if [ "$state" = "paused" ] && [ "$pct" -ge "$RESUME" ]; then
      pkill -CONT -f "$JOBS" 2>/dev/null && log "resumed jobs at ${pct}%"
      notify "Заряд ${pct}%, работа продолжена"
      state=running
    fi
  else
    if [ "$state" = "running" ] && [ "$pct" -le "$PAUSE" ]; then
      pkill -STOP -f "$JOBS" 2>/dev/null && log "paused jobs at ${pct}% (discharging)"
      notify "Заряд ${pct}% — работа приостановлена"
      state=paused
    fi
    if [ "$pct" -le "$WARN" ] && [ $((last_warn + 10)) -le "$pct" -o "$pct" -lt "$last_warn" ]; then
      notify "Заряд ${pct}% — подключите питание"
      last_warn=$pct
    fi
  fi
  sleep "$INTERVAL"
done
