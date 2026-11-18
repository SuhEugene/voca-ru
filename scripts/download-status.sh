#!/bin/sh
# Progress, speed and ETA of a .cache download made of range parts.
# Usage: download-status.sh [parts-dir] [total-bytes] [sample-seconds]
# Defaults track the Qwen3-TTS-1.7B teacher download.
set -u
D=${1:-.cache/models/qwen3-tts-1.7b/parts-model.safetensors}
TOTAL=${2:-3857413744}
SAMPLE=${3:-5}

sum() { find "$D" -name 'part.*' -exec stat -f%z {} + 2>/dev/null | awk '{s+=$1} END{print s+0}'; }
human() { awk -v n="$1" 'BEGIN{split("B KB MB GB TB",u," ");i=1;while(n>=1024&&i<5){n/=1024;i++}printf (i==1?"%d %s":"%.1f %s"),n,u[i]}'; }

a=$(sum); sleep "$SAMPLE"; b=$(sum)
rate=$(( (b - a) / SAMPLE ))
left=$(( TOTAL - b )); [ "$left" -lt 0 ] && left=0

pct=$(awk -v b="$b" -v t="$TOTAL" 'BEGIN{printf "%.1f", (t?100*b/t:0)}')
bar=$(awk -v p="$pct" 'BEGIN{n=int(p/2.5); for(i=0;i<50;i++) printf (i<n?"#":"-")}')
if [ "$rate" -gt 0 ]; then
  eta=$(awk -v l="$left" -v r="$rate" 'BEGIN{s=int(l/r); printf "%dh%02dm%02ds", s/3600, (s%3600)/60, s%60}')
else
  eta="stalled"
fi

printf '%s\n' "[$bar] $pct%"
printf '%s / %s   %s/s   left %s   ETA %s\n' "$(human "$b")" "$(human "$TOTAL")" "$(human "$rate")" "$(human "$left")" "$eta"
