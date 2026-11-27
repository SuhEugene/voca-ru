#!/bin/bash
# Stage gates for the second corpus and the retrain. Each subcommand prints one
# line of machine-readable numbers and PASS/FAIL, so a check is one ssh call and
# a failure is visible without reading a log.
#
#   bash v2-gates.sh status
#   bash v2-gates.sh setup | generate | screen | prepare | annotate | train
set -uo pipefail
cd /workspace/voca || exit 1

V2=data/corpus-v2
WORK=data/ru-corpus-v2
EXPECTED_CLIPS=19822
PY=${PY:-python}

count() { [ -d "$1" ] && find "$1" -maxdepth 1 -name "$2" | wc -l || echo 0; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

setup() {
    local models male ok=1
    models=$(du -sm /workspace/voca/.cache/models 2>/dev/null | cut -f1)
    male=$(count /workspace/v1/corpus/audio/m_neutral '*.flac')
    [ "${models:-0}" -ge 8000 ] || ok=0
    [ "$male" -ge 9595 ] || ok=0
    echo "setup models_mb=${models:-0} male_audio=${male} expect_male=9595 status=$([ $ok = 1 ] && echo PASS || echo FAIL)"
}

generate() {
    local raw lost ok=1
    raw=$(lines "$V2/raw.jsonl")
    lost=$(lines "$V2/raw.jsonl.lost" 2>/dev/null || echo 0)
    [ "$raw" -ge "$EXPECTED_CLIPS" ] || ok=0
    echo "generate raw=${raw} expect=${EXPECTED_CLIPS} lost=${lost} status=$([ $ok = 1 ] && echo PASS || echo FAIL)"
}

screen() {
    local raw accepted rejected ok=1 male_raw male_acc male_rej
    male_raw=$(lines "data/corpus-v2-male/raw.jsonl")
    male_acc=$(lines "data/corpus-v2-male/manifest.jsonl")
    male_rej=$(lines "data/corpus-v2-male/review.jsonl")
    raw=$(( $(lines "$V2/raw.jsonl") + male_raw ))
    accepted=$(( $(lines "$V2/manifest.jsonl") + male_acc ))
    rejected=$(( $(lines "$V2/review.jsonl") + male_rej ))
    [ "$raw" -gt 0 ] || ok=0
    [ $((accepted + rejected)) -ge "$raw" ] || ok=0
    echo "screen raw=${raw} accepted=${accepted} rejected=${rejected} male=${male_acc} status=$([ $ok = 1 ] && echo PASS || echo FAIL)"
}

prepare() {
    local rows=0 features=0 expected ok=1
    for split in train validation test; do
        rows=$((rows + $(lines "$WORK/inputs/$split.jsonl")))
        features=$((features + $(find "$WORK/prepared/$split" -name '*.pt' 2>/dev/null | wc -l)))
    done
    expected=${rows:-0}
    [ "$expected" -gt 0 ] || ok=0
    [ "${features:-0}" -eq "$expected" ] || ok=0
    echo "prepare inputs=${expected} features=${features:-0} status=$([ $ok = 1 ] && echo PASS || echo FAIL)"
}

annotate() {
    local rows accept reject ok=1 input=0
    rows=$(( $(lines "$WORK/annotations/train.jsonl") + $(lines "$WORK/annotations/validation.jsonl") + $(lines "$WORK/annotations/test.jsonl") ))
    reject=$(( $(lines "$WORK/annotations/train.rejected.jsonl") + $(lines "$WORK/annotations/validation.rejected.jsonl") + $(lines "$WORK/annotations/test.rejected.jsonl") ))
    accept=$rows
    input=$(lines "$WORK/prepared/train.jsonl")
    [ "$accept" -gt 0 ] || ok=0
    echo "annotate input_train=${input} accepted=${accept} rejected=${reject} status=$([ $ok = 1 ] && echo PASS || echo FAIL)"
}

train() {
    local log=/workspace/train.log step loss sps ok=1
    step=$(grep -ao '"step": *[0-9]*' "$log" 2>/dev/null | tail -1 | grep -o '[0-9]*$')
    step=${step:-0}
    loss=$(grep -ao '"loss": *[0-9.]*' "$log" 2>/dev/null | tail -1 | grep -o '[0-9.]*$')
    sps=$(grep -ao '"steps_per_second": *[0-9.]*' "$log" 2>/dev/null | tail -1 | grep -o '[0-9.]*$')
    [ "$step" -gt 0 ] || ok=0
    echo "train step=${step} target=104000 loss=${loss:-na} sps=${sps:-na} status=$([ $ok = 1 ] && echo PASS || echo FAIL)"
}

status() {
    echo "setup:    $(setup)"
    echo "generate: $(generate)"
    echo "screen:   $(screen)"
    echo "prepare:  $(prepare)"
    echo "annotate: $(annotate)"
    echo "train:    $(train)"
}

case "${1:-status}" in
setup) setup ;;
generate) generate ;;
screen) screen ;;

prepare) prepare ;;
annotate) annotate ;;
train) train ;;
status) status ;;
*) echo "usage: $0 [status|setup|generate|screen|prepare|annotate|train]" >&2; exit 2 ;;
esac
