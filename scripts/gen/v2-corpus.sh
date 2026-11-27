#!/bin/bash
# Second corpus on a rented GPU: a new female voice (s3_lower_register, shipped as
# assets/voices/f_storyteller.*) over all 10 500 sentences, screened, prepared and
# annotated. The male clips of the released corpus are reused as they are.
#
#   bash v2-corpus.sh models    # download the teacher, judges and dictionary
#   bash v2-corpus.sh male      # released corpus: manifest + audio
#   bash v2-corpus.sh generate  # 10 500 female clips (wave 1)
#   bash v2-corpus.sh screen    # Whisper WER + meter2 stress + duration checks
set -euo pipefail
cd /workspace/voca

export VOCA_CACHE=/workspace/voca/.cache
export VOCA_MODELS=$VOCA_CACHE/models
export VOCA_TTS_DEVICE=cuda
export VOCA_ASR_DEVICE=cuda
export HF_HUB_ENABLE_HF_TRANSFER=1
OUT=data/corpus-v2
PY=${PY:-python}

models() {
    $PY - <<'EOF'
import os
from huggingface_hub import snapshot_download

root = "/workspace/voca/.cache/models"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
jobs = [
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", None, "qwen3-tts-1.7b"),
    ("siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf", None, "qwen3-tts-ru-stress"),
    ("deepdml/faster-whisper-large-v3-turbo-ct2", None, "whisper-large-v3-turbo"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-russian", None, "wav2vec2-xlsr-ru"),
    ("ruaccent/accentuator", None, "ruaccent"),
]
from concurrent.futures import ThreadPoolExecutor
def fetch(job):
    repo, allowed, dest = job
    path = snapshot_download(repo, local_dir=os.path.join(root, dest),
                             allow_patterns=allowed)
    print("ready", dest, path, flush=True)
with ThreadPoolExecutor(5) as pool:
    for _ in pool.map(fetch, jobs):
        pass
EOF
}

male() {
    $PY - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download("SuhEugene/voca-ru-corpus", repo_type="dataset",
                         allow_patterns=["data/*.parquet", "manifest.jsonl"],
                         local_dir="/workspace/v1")
print("dataset at", path)
EOF
    mkdir -p /workspace/v1/corpus
    $PY scripts/extract-parquet-audio.py --parquet /workspace/v1 --out /workspace/v1/corpus
}

generate() {
    # both voices, every split: the male clips are regenerated too, so the whole
    # corpus carries the context-resolved stress
    $PY scripts/gen/split_corpus.py --generate-only --voices f_storyteller,m_neutral \
        --sources batch1.jsonl batch2.jsonl batch3.jsonl \
        --ref-dir assets/voices --out "$OUT" --batch-size "${BATCH:-24}" --wave 1 \
        --require-lora --split all
}

screen() {
    $PY scripts/gen/split_corpus.py --screen-only --voices f_storyteller,m_neutral --out "$OUT" \
        --max-wer 0.15 --asr-device cuda --asr-compute-type float16 \
        --stress-device cuda --screen-batch 16
}

case "${1:-}" in
models)   models ;;
male)     male ;;
generate) generate ;;
screen)   screen ;;
*) echo "usage: $0 models|male|generate|fix|screen" >&2; exit 2 ;;
esac
