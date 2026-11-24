#!/bin/bash
# Grab an RTX 4090 and prove CUDA works before doing anything else.
#
#   scripts/runpod-grab.sh            # poll every 10 s, up to 6 h
#   DISK_GB=50 POLL=20 scripts/runpod-grab.sh
#
# Exits 0 and writes .cache/runpod-pod.json when a pod is up AND
# `cuInit(0) == 0` inside it, 1 on a fatal API error or a CUDA failure,
# 2 after the time limit.
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
DISK_GB="${DISK_GB:-50}"
IMAGE="${IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}"
API="https://api.runpod.io/v2"
mkdir -p "$(dirname "$LOG")"

# nvidia-caps device nodes are missing on some hosts, which makes cuInit fail
# with 999; asking the runtime for all driver capabilities makes it create them.
BODY=$(jq -nc --arg img "$IMAGE" --argjson disk "$DISK_GB" '{
  name: "voca-ru-corpus", cloud: "COMMUNITY",
  gpu: {id: "NVIDIA GeForce RTX 4090", count: 1},
  image: $img, disk: $disk, ports: ["22/tcp"], startSsh: true,
  env: {NVIDIA_DRIVER_CAPABILITIES: "all"}
}')

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

cuda_ok() {  # $1 = pod id
  local id="$1" raw rc
  for _ in $(seq 1 12); do
    raw=$(curl -s -m 30 -X POST -H "Authorization: Bearer $KEY" \
      -H 'Content-Type: application/json' \
      --data-binary '{"command":["python3","-c","import ctypes;l=ctypes.CDLL(\"libcuda.so.1\");print(l.cuInit(0))"],"timeoutSeconds":25}' \
      "$API/pods/$id/exec" 2>/dev/null)
    rc=$(printf '%s' "$raw" | jq -r '.stdout // empty' 2>/dev/null | tr -d '[:space:]')
    [ "$rc" = "0" ] && return 0
    sleep 10
  done
  return 1
}

deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
attempt=0
log "start poll=${POLL}s max=${MAX_HOURS}h disk=${DISK_GB}G pid=$$"

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
      [ -n "$id" ] || { log "create $code without id: $(head -c 200 "$TMP")"; continue; }
      if ! awk -v c="$cost" -v m="$MAX_PRICE" 'BEGIN { exit !(c <= m) }'; then
        log "REFUSED: \$$cost/h over cap \$$MAX_PRICE — deleting $id"
        curl -s -o /dev/null -X DELETE -H "Authorization: Bearer $KEY" "$API/pods/$id"
        exit 1
      fi
      log "created id=$id \$$cost/h — waiting for RUNNING and CUDA"
      sleep 20
      if cuda_ok "$id"; then
        jq -n --arg id "$id" --argjson cost "$cost" --arg ssh "$ssh" \
              --arg at "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
          '{id:$id, cost:$cost, ssh:$ssh, created:$at, cuda:"ok"}' > "$OUT"
        log "READY id=$id cuda=ok ssh: $ssh"
        exit 0
      fi
      log "CUDA FAILED on $id — keeping the pod for diagnosis, stopping"
      jq -n --arg id "$id" --argjson cost "$cost" --arg ssh "$ssh" \
            --arg at "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
        '{id:$id, cost:$cost, ssh:$ssh, created:$at, cuda:"failed"}' > "$OUT"
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
    401|403) log "auth failed ($code)"; exit 1 ;;
    *) log "unexpected HTTP $code: $(head -c 200 "$TMP")" ;;
  esac
  sleep "$POLL"
done

log "time limit ${MAX_HOURS}h reached without a working 4090"
exit 2
