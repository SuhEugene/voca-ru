#!/bin/bash
# Unattended chain for the v2 run: waits for generation, screens, builds inputs,
# prepares, annotates, trains to 104 000 steps with restarts, then evaluates and
# exports. Touching /workspace/v2-complete is the only success signal.
#
#   setsid nohup bash scripts/v2-chain.sh > /workspace/chain.log 2>&1 &
set -u
cd /workspace/voca || exit 1
log() { echo "[chain] $(date -u +%H:%M:%S) $*"; }
step_now() {
    python - <<'EOF' 2>/dev/null || echo 0
import torch
state = torch.load("/workspace/voca/runs/v2/last.pt", map_location="cpu", weights_only=False)
print(state.get("step", 0))
EOF
}

log "waiting for generation"
while ! grep -q 'GEN_DONE=0' /workspace/gen.log 2>/dev/null; do
    sleep 30
    grep -q 'GEN_DONE=' /workspace/gen.log 2>/dev/null && { log "generation failed"; exit 1; }
done
log "generation done: $(wc -l < data/corpus-v2/raw.jsonl) rows"

log "homographs: regenerating the clips whose reading the context changes"
bash scripts/gen/v2-corpus.sh fix >> /workspace/fix.log 2>&1 || { log "fix failed"; exit 1; }
log "after fix: female raw $(wc -l < data/corpus-v2/raw.jsonl), male raw $(wc -l < data/corpus-v2-male/raw.jsonl 2>/dev/null || echo 0)"

log "screening"
bash scripts/gen/v2-corpus.sh screen >> /workspace/screen.log 2>&1 || { log "screen failed"; exit 1; }
log "screened: accepted $(wc -l < data/corpus-v2/manifest.jsonl), rejected $(wc -l < data/corpus-v2/review.jsonl)"

log "inputs"
bash scripts/v2-train.sh inputs >> /workspace/prepare.log 2>&1 || { log "inputs failed"; exit 1; }

log "prepare"
bash scripts/v2-train.sh prepare 12 >> /workspace/prepare.log 2>&1 || { log "prepare failed"; exit 1; }

log "annotate"
bash scripts/v2-train.sh annotate >> /workspace/annotate.log 2>&1 || { log "annotate failed"; exit 1; }
log "annotated: train $(wc -l < data/ru-corpus-v2/annotations/train.jsonl)"

# the rental must stop before the credit runs out at 13:52 UTC
bash scripts/v2-train.sh budget 9.5 >> /workspace/train.log 2>&1

for attempt in 1 2 3 4 5 6; do
    log "training attempt $attempt from step $(step_now)"
    bash scripts/v2-train.sh train 104000 >> /workspace/train.log 2>&1
    reached=$(step_now)
    log "attempt $attempt ended at step $reached"
    [ "${reached:-0}" -ge 104000 ] && break
done

reached=$(step_now)
if [ "${reached:-0}" -lt 104000 ]; then
    log "training stopped below target at $reached; keeping the checkpoint"
fi

log "evaluating and exporting"
bash scripts/gen/v2-finish.sh all >> /workspace/finish.log 2>&1 || { log "finish failed"; exit 1; }

touch /workspace/v2-complete
log "complete"
