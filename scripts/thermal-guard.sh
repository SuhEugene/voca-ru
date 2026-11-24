#!/bin/bash
# Keep a long job under a CPU temperature ceiling by pausing/resuming it.
#
#   thermal-guard.sh MAX_TEMP COOL_TEMP command [args...]
#
# The job runs in its own process group, so SIGSTOP hits the whole tree.
# A paused job holds its memory and resumes exactly where it stopped; nothing
# is lost. Temperature comes from `macmon pipe` (no sudo needed on Apple Silicon).
set -uo pipefail

MAX=${1:?usage: thermal-guard.sh MAX_TEMP COOL_TEMP command...}
COOL=${2:?usage: thermal-guard.sh MAX_TEMP COOL_TEMP command...}
shift 2

LOG=${THERMAL_LOG:-/Users/suheugene/projects/22-tiniest-voca-ru/.cache/logs/thermal.log}
mkdir -p "$(dirname "$LOG")"

temp() {
    macmon pipe -s 1 2>/dev/null | head -1 |
        /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["temp"]["cpu_temp_avg"])' 2>/dev/null
}

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

# macOS has no setsid(1); job control gives each background job its own
# process group, so `kill -STOP -$PID` reaches the whole tree.
set -m
"$@" &
PID=$!
set +m
trap 'kill -CONT -$PID 2>/dev/null; kill -TERM -$PID 2>/dev/null; exit 130' INT TERM
log "start pid=$PID max=${MAX}C cool=${COOL}C cmd=$*"

paused=0
while kill -0 "$PID" 2>/dev/null; do
    T=$(temp)
    if [ -n "$T" ]; then
        if [ "$paused" -eq 0 ] && awk -v t="$T" -v m="$MAX" 'BEGIN{exit !(t>=m)}'; then
            kill -STOP -"$PID" 2>/dev/null && { paused=1; log "pause ${T}C"; }
        elif [ "$paused" -eq 1 ] && awk -v t="$T" -v c="$COOL" 'BEGIN{exit !(t<=c)}'; then
            kill -CONT -"$PID" 2>/dev/null && { paused=0; log "resume ${T}C"; }
        fi
    fi
    sleep 5
done
[ "$paused" -eq 1 ] && { kill -CONT -"$PID" 2>/dev/null; log "resume on exit ${T}C"; }
wait "$PID"
RC=$?
log "exit rc=$RC"
exit "$RC"
