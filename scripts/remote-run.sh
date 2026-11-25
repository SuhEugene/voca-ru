#!/bin/bash
# Remote training run on a rented GPU. Idempotent: prepare resumes, train
# resumes from last.pt.
#
#   bash /workspace/voca/run.sh prepare     # rewrite paths + acoustic features
#   bash /workspace/voca/run.sh train       # 104k steps
set -euo pipefail
cd /workspace/voca

CORPUS=/workspace/corpus
OUT=data/ru-corpus
PY=${PY:-python}
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

case "${1:-all}" in
prepare)
    # every input row carries the Mac's absolute audio path; point it at the pod
    $PY - <<'EOF'
import json, pathlib
root = pathlib.Path("/workspace/voca/data/ru-corpus/inputs")
prefix = "/Users/suheugene/projects/22-tiniest-voca-ru/.cache/models/corpus-ru/corpus-final/"
for path in root.glob("*.jsonl"):
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    for row in rows:
        row["audio"] = row["audio"].replace(prefix, "/workspace/corpus/")
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(path.name, len(rows), "rows rewritten")
EOF
    for split in train validation test; do
        echo "[prep] $split $(date -u +%FT%TZ)"
        $PY training/prepare.py --inputs "$OUT/inputs/$split.jsonl" --output "$OUT/prepared" \
            --split "$split" --resume
    done
    # the annotation layer points at the acoustic root it was built against
    $PY - <<'EOF'
import json, pathlib
path = pathlib.Path("/workspace/voca/data/ru-corpus/annotations/annotations.json")
meta = json.loads(path.read_text())
meta["acoustic_root"] = "/workspace/voca/data/ru-corpus/prepared"
path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
print("acoustic_root ->", meta["acoustic_root"])
EOF
    ;;
train)
    echo "[train] start $(date -u +%FT%TZ)"
    RESUME=()
    [ -f runs/ru/last.pt ] && RESUME=(--resume runs/ru/last.pt)
    $PY training/train.py --train "$OUT/annotations/train.jsonl" \
        --validation "$OUT/annotations/validation.jsonl" \
        --output runs/ru --config configs/base.yaml --device cuda \
        --steps 104000 --save-every 1000 --sample-every 2000 --eval-every 2000 \
        "${RESUME[@]}"
    ;;
all)
    bash "$0" prepare
    bash "$0" train
    ;;
esac
