#!/bin/bash
# One line per stage for the pw fields: percent 0-98 while a stage runs, 100 only
# when its gate passes. A dead stage stays at its last percentage; is_dead.sh
# reports the death.
#   v2-status.sh setup|generate|screen|prepare|annotate|train
set -u
cd /workspace/voca || exit 1
V2=data/corpus-v2
WORK=data/ru-corpus-v2
stage=${1:-train}
PY=${PY:-python}

count() { [ -d "$1" ] && find "$1" -maxdepth 1 -name "$2" | wc -l || echo 0; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

case "$stage" in
setup)
    models=$(du -sm /workspace/voca/.cache/models 2>/dev/null | cut -f1); models=${models:-0}
    male=$(count /workspace/v1/corpus/audio/m_neutral '*.flac')
    if [ "$models" -ge 8000 ] && [ "$male" -ge 9595 ]; then echo "100 setup"; exit 0; fi
    awk -v m="$models" -v a="$male" 'BEGIN{ p=(m/8000*70)+(a/9595*30); if(p>98)p=98; if(p<0)p=0; printf "%.1f setup\n", p }'
    ;;
generate)
    raw=$(lines "$V2/raw.jsonl")
    if [ "$raw" -ge 19822 ]; then echo "100 generate"; exit 0; fi
    awk -v r="$raw" 'BEGIN{ p=r*98/19822; if(p>98)p=98; printf "%.1f generate\n", p }'
    ;;
screen)
    raw=$(( $(lines "$V2/raw.jsonl") + $(lines "data/corpus-v2-male/raw.jsonl") ))
    done_=$(( $(lines "$V2/manifest.jsonl") + $(lines "$V2/review.jsonl") + $(lines "data/corpus-v2-male/manifest.jsonl") + $(lines "data/corpus-v2-male/review.jsonl") ))
    if [ "$raw" -gt 0 ] && [ "$done_" -ge "$raw" ]; then echo "100 screen"; exit 0; fi
    awk -v d="$done_" -v r="$raw" 'BEGIN{ if(r==0){printf "0.0 screen\n"; exit} p=d*98/r; if(p>98)p=98; printf "%.1f screen\n", p }'
    ;;
prepare)
    rows=0; feats=0
    for split in train validation test; do
        rows=$((rows + $(lines "$WORK/inputs/$split.jsonl")))
        feats=$((feats + $(find "$WORK/prepared/$split" -name '*.pt' 2>/dev/null | wc -l)))
    done
    if [ "$rows" -gt 0 ] && [ "$feats" -eq "$rows" ]; then echo "100 prepare"; exit 0; fi
    awk -v f="$feats" -v r="$rows" 'BEGIN{ if(r==0){printf "0.0 prepare\n"; exit} p=f*98/r; if(p>98)p=98; printf "%.1f prepare\n", p }'
    ;;
annotate)
    acc=$(( $(lines "$WORK/annotations/train.jsonl") + $(lines "$WORK/annotations/validation.jsonl") + $(lines "$WORK/annotations/test.jsonl") ))
    rej=$(( $(lines "$WORK/annotations/train.rejected.jsonl") + $(lines "$WORK/annotations/validation.rejected.jsonl") + $(lines "$WORK/annotations/test.rejected.jsonl") ))
    total=0
    for split in train validation test; do total=$((total + $(lines "$WORK/prepared/$split.jsonl"))); done
    if [ "$total" -gt 0 ] && [ $((acc + rej)) -ge "$total" ]; then echo "100 annotate"; exit 0; fi
    awk -v d="$((acc + rej))" -v t="$total" 'BEGIN{ if(t==0){printf "0.0 annotate\n"; exit} p=d*98/t; if(p>98)p=98; printf "%.1f annotate\n", p }'
    ;;
train)
    log=/workspace/train.log
    step=$(grep -ao '"step": *[0-9]*' "$log" 2>/dev/null | tail -1 | grep -o '[0-9]*$'); step=${step:-0}
    if [ "$step" -ge 104000 ]; then echo "100 train"; exit 0; fi
    awk -v s="$step" 'BEGIN{ p=s*98/104000; if(p>98)p=98; printf "%.1f train\n", p }'
    ;;
esac
