#!/bin/bash
# Last stage of the v2 run on the rented host: test-split WER, ONNX and GGUF
# exports, one tarball to pull home, sha256 of everything, and the credit check.
#
#   bash v2-finish.sh
set -euo pipefail
cd /workspace/voca

export VOCA_CACHE=/workspace/voca/.cache
export VOCA_MODELS=$VOCA_CACHE/models
export HF_HUB_ENABLE_HF_TRANSFER=1
WORK=data/ru-corpus-v2
ART=/workspace/artifacts-v2
PY=${PY:-python}
mkdir -p "$ART"

eval_wer() {
    $PY scripts/wer_test.py runs/v2/inference.pt 40 "$WORK" 2>&1 | tee "$ART/wer.txt"
}

export_onnx() {
    $PY scripts/export_onnx.py runs/v2/inference.pt "$ART/voca-ru-v2.onnx"
}

export_gguf() {
    $PY scripts/export_gguf.py runs/v2/inference.pt "$ART/voca-ru-v2-js.gguf" \
        --stress crates/voca-ru-text/src/data/stress_dict.txt --lexicon crates/voca-ru-text/src/data/g2p_lexicon.txt
}

pack() {
    cp runs/v2/inference.pt runs/v2/metrics.jsonl runs/v2/config.json "$ART/"
    cp runs/v2/eval-*.json "$ART/" 2>/dev/null || true
    [ -d runs/v2/samples ] && cp -r runs/v2/samples "$ART/"
    # directories (samples) cannot be hashed; a file list keeps the step honest
    ( cd "$ART" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\n' \
        | sort | xargs sha256sum > SHA256SUMS )
    tar -C "$ART" -cf /workspace/voca-ru-v2.tar .
    sha256sum /workspace/voca-ru-v2.tar
}

case "${1:-all}" in
wer)    eval_wer ;;
onnx)   export_onnx ;;
gguf)   export_gguf ;;
pack)   pack ;;
all)    eval_wer; export_onnx; export_gguf; pack ;;
*) echo "usage: $0 [wer|onnx|gguf|pack|all]" >&2; exit 2 ;;
esac
