#!/bin/bash
# Grab an RTX 4090 on Runpod the moment capacity appears.
#
#   scripts/runpod-grab.sh            # poll every 10 s, up to 6 h
#   POLL=20 MAX_HOURS=1 scripts/runpod-grab.sh
#
# Exits 0 and writes .cache/runpod-pod.json on success, 1 on a fatal API error
# (bad key, no funds, price above cap), 2 after the time limit.
set -uo pipefail

KEY="${RUNPOD_API_KEY:-$(cat "$HOME/.secrets/runpod-api-key" 2>/dev/null)}"
[ -n "$KEY" ] || { echo "no Runpod API key (set RUNPOD_API_KEY or ~/.secrets/runpod-api-key)" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/.cache/logs/runpod-grab.log"
OUT="$ROOT/.cache/runpod-pod.json"
TMP="$ROOT/.cache/logs/runpod-create.json"
POLL="${POLL:-10}"
MAX_HOURS="${MAX_HOURS:-6}"
MAX_PRICE="${MAX_PRICE:-0.45}"
API="https://api.runpod.io/v2"
mkdir -p "$(dirname "$LOG")"

BODY='{"name":"voca-ru-corpus","cloud":"COMMUNITY","gpu":{"id":"NVIDIA GeForce RTX 4090","count":1},"image":"runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404","disk":50,"ports":["22/tcp"],"startSsh":true}'

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
attempt=0
log "start poll=${POLL}s max=${MAX_HOURS}h pid=$$"

while [ "$(date +%s)" -lt "$deadline" ]; do
  attempt=$((attempt + 1))
  code=$(curl -s -o "$TMP" -w '%{http_code}' -m 30 \
    -X POST -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "$BODY" "$API/pods")

  case "$code" in
    200|201)
      id=$(jq -r '.id // empty' "$TMP")
      cost=$(jq -r '.cost // 0' "$TMP")
      ssh=$(jq -r '.ssh.proxy.command // empty' "$TMP")
      if [ -z "$id" ]; then
        log "create answered $code without a pod id: $(head -c 300 "$TMP")"
        continue
      fi
      if awk -v c="$cost" -v m="$MAX_PRICE" 'BEGIN { exit !(c <= m) }'; then
        jq -n --arg id "$id" --argjson cost "$cost" --arg ssh "$ssh" \
              --arg at "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
          '{id:$id, cost:$cost, ssh:$ssh, created:$at}' > "$OUT"
        log "GRABBED id=$id cost=\$$cost/h ssh: $ssh"
        exit 0
      fi
      log "REFUSED: cost \$$cost/h exceeds cap \$$MAX_PRICE — deleting pod $id"
      curl -s -o /dev/null -X DELETE -H "Authorization: Bearer $KEY" "$API/pods/$id"
      exit 1
      ;;
    400)
      detail=$(jq -r '.detail // empty' "$TMP" 2>/dev/null)
      if [[ "$detail" == *"no longer any instances available"* ]]; then
        [ $((attempt % 6)) -eq 1 ] && log "no capacity (attempt $attempt)"
      else
        log "400: $detail"
        case "$detail" in *"balance"*) exit 1 ;; esac
      fi
      ;;
    401|403)
      log "auth failed ($code) — check the API key"
      exit 1
      ;;
    *)
      log "unexpected HTTP $code: $(head -c 200 "$TMP")"
      ;;
  esac

  sleep "$POLL"
done

log "time limit ${MAX_HOURS}h reached without finding a 4090"
exit 2
