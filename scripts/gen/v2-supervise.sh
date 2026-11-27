#!/bin/bash
# Keeps training alive on a rented host that sometimes kills the process, then
# evaluates and exports. Every restart resumes from runs/v2/last.pt, so at most
# one save interval is lost.
#
#   setsid nohup bash scripts/gen/v2-supervise.sh > /workspace/supervise.log 2>&1 &
set -u
cd /workspace/voca || exit 1
TARGET=${1:-104000}
PY=${PY:-python}
log() { echo "[supervise] $(date -u +%H:%M:%S) $*"; }

step_now() {
    $PY - <<'EOF' 2>/dev/null || echo 0
import torch
state = torch.load("/workspace/voca/runs/v2/last.pt", map_location="cpu", weights_only=False)
print(state.get("step", 0))
EOF
}

for attempt in $(seq 1 30); do
    reached=$(step_now)
    [ "${reached:-0}" -ge "$TARGET" ] && break
    log "attempt $attempt from step $reached"
    bash scripts/gen/v2-train.sh train "$TARGET" >> /workspace/train.log 2>&1
    log "attempt $attempt exited $? at step $(step_now)"
done

reached=$(step_now)
log "training reached $reached of $TARGET"
bash scripts/gen/v2-finish.sh all >> /workspace/finish.log 2>&1 || { log "finish failed"; exit 1; }
touch /workspace/v2-complete
log "complete"
