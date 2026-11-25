#!/bin/bash
# GAN fine-tune of the trained Russian model on a rented GPU.
# Runs on the pod, inside tmux: download corpus -> extract audio -> prepare ->
# continue from the 104k checkpoint with a stronger discriminator.
set -euo pipefail
cd /workspace/voca

CORPUS=/workspace/corpus
OUT=data/ru-corpus
PY=${PY:-python}
WORKERS=${WORKERS:-12}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_ENABLE_HF_TRANSFER=1

step=${1:-all}

corpus() {
    $PY - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download("SuhEugene/voca-ru-corpus", repo_type="dataset",
                         allow_patterns=["data/*.parquet"], local_dir="/workspace/parquet")
print("parquet at", path)
EOF
    $PY extract.py --parquet /workspace/parquet --out "$CORPUS"
}

prepare() {
    $PY - <<'EOF'
import json, pathlib
root = pathlib.Path("/workspace/voca/data/ru-corpus/inputs")
prefix = "/Users/suheugene/projects/22-tiniest-voca-ru/.cache/models/corpus-ru/corpus-final/"
for path in root.glob("*.jsonl"):
    if path.name.startswith("._"):   # macOS AppleDouble droppings from tar/rsync
        continue
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    for row in rows:
        row["audio"] = row["audio"].replace(prefix, "/workspace/corpus/")
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(path.name, len(rows), "rows rewritten")
EOF
    bash prepare-parallel.sh "$WORKERS"
    $PY - <<'EOF'
import json, pathlib
path = pathlib.Path("/workspace/voca/data/ru-corpus/annotations/annotations.json")
meta = json.loads(path.read_text())
meta["acoustic_root"] = "/workspace/voca/data/ru-corpus/prepared"
path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print("acoustic_root ->", meta["acoustic_root"])
EOF
}

train() {
    local target=${1:-134000}
    local run=runs/ft
    # train.py caps each invocation at --max-seconds (default 3600) and exits at
    # a step boundary, so reaching the target takes several resumes.
    while :; do
        local done_step=0
        [ -f "$run/last.pt" ] && done_step=$($PY - "$run/last.pt" <<'PYEOF'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["step"])
PYEOF
)
        [ -z "$done_step" ] && done_step=104000
        if [ "$done_step" -ge "$target" ]; then
            echo "[train] reached step $done_step (target $target)"
            break
        fi
        local extra=(--init-from runs/gan/last.pt --new-data)
        [ -f "$run/last.pt" ] && extra=(--resume "$run/last.pt")
        echo "[train] step $done_step -> $target"
        $PY training/train.py --train "$OUT/annotations/train.jsonl" \
            --validation "$OUT/annotations/validation.jsonl" \
            --output "$run" --config configs/base.yaml --device cuda \
            --steps "$target" "${extra[@]}" --max-seconds 7200 \
            --gan-start 0 --gan-ratio 1.1 --feature-matching 45 --gan-ramp 1000 \
            --save-every 1000 --sample-every 2000 --eval-every 10000 \
            --log-every 10 || return 1
    done
}

stage() {
    local target=$1 name=$2
    train "$target" || return 1
    # the trainer caps each call at --max-seconds and can exit early; only a
    # checkpoint that actually reached the target counts as a finished stage
    local reached
    reached=$($PY - runs/ft/last.pt <<'PYEOF'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["step"])
PYEOF
)
    if [ "$reached" -lt "$target" ]; then
        echo "STAGE_SHORT $name: step $reached < $target" >> /workspace/train.log
        return 1
    fi
    mkdir -p "/workspace/artifacts/$name"
    cp -a runs/ft/inference.pt runs/ft/last.pt runs/ft/metrics.jsonl runs/ft/config.json \
          runs/ft/recipe.json "/workspace/artifacts/$name/" 2>/dev/null || true
    cp -a runs/ft/samples "/workspace/artifacts/$name/" 2>/dev/null || true
    touch "/workspace/artifacts/$name/DONE"
    echo "STAGE_DONE $name step $reached" >> /workspace/train.log
}

case "$step" in
all)     corpus; prepare; stage 134000 stage1; stage 164000 stage2 ;;
corpus)  corpus ;;
prepare) prepare ;;
train)   train 134000 ;;
stage1)  stage 134000 stage1 ;;
stage2)  stage 164000 stage2 ;;
esac
