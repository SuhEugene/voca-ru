#!/bin/bash
# Inputs, acoustic features, alignment and training for the second corpus.
# The male clips are reused from the released corpus; the female ones are the new
# voice. Everything is resumable; the trainer stops itself at the budget deadline.
#
#   bash v2-train.sh inputs|prepare|annotate|train|budget [args]
set -euo pipefail
cd /workspace/voca

export VOCA_CACHE=/workspace/voca/.cache
export VOCA_MODELS=$VOCA_CACHE/models
export VOCA_ASR_DEVICE=cuda
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT=data/corpus-v2
WORK=data/ru-corpus-v2
PY=${PY:-python}

inputs() {
    $PY scripts/gen/v2_inputs.py --female "$OUT" --male /workspace/v1/corpus \
        --out "$WORK/inputs" --female-voice f_storyteller \
        --male-extra data/corpus-v2-male \
        --female-audio "$OUT" --male-audio /workspace/v1/corpus
}

prepare() {
    OUT="$WORK" bash scripts/prepare-parallel.sh "${1:-12}"
}

annotate() {
    $PY training/annotate.py --features "$WORK/prepared" --output "$WORK/annotations" \
        --splits train validation test --device cuda --resume
}

train() {
    local steps=${1:-104000}
    local resume=""
    [ -f runs/v2/last.pt ] && resume="--resume"
    $PY training/train.py --train "$WORK/annotations/train.jsonl" \
        --validation "$WORK/annotations/validation.jsonl" \
        --output runs/v2 --config configs/base.yaml --device cuda \
        --steps "$steps" --deadline-file /workspace/budget.json \
        --save-every 1000 --sample-every 2000 --eval-every 10000 \
        --log-every 10 $resume "${@:2}"
}

budget() {
    # $1 = hours from now that the rental may run; leaves the margin the caller asks for
    $PY - "$1" <<'EOF'
import json, sys, time
hours = float(sys.argv[1])
json.dump({"deadline": time.time() + hours * 3600}, open("/workspace/budget.json", "w"))
print("deadline in", hours, "h ->", open("/workspace/budget.json").read())
EOF
}

case "${1:-}" in
inputs)   inputs ;;
prepare)  prepare "${2:-12}" ;;
annotate) annotate ;;
train)    shift; train "$@" ;;
budget)   budget "${2:?hours}" ;;
*) echo "usage: $0 inputs|prepare|annotate|train|budget" >&2; exit 2 ;;
esac
