#!/usr/bin/env bash
# Prepare a rented CUDA box for the corpus generation stage.
# Idempotent: rerunning it verifies and fills in only what is missing.
#
#   HF_TOKEN is NOT needed here (all assets are public); it is needed for publish.sh.
#
#   bash remote/bootstrap.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="python3.12"
VENV="$ROOT/.venv-remote"
FAILS=0
declare -a CHECKS=()

pass() { CHECKS+=("PASS  $*"); printf 'PASS  %s\n' "$*"; }
fail() { CHECKS+=("FAIL  $*"); FAILS=$((FAILS + 1)); printf 'FAIL  %s\n' "$*" >&2; }
warn() { CHECKS+=("WARN  $*"); printf 'WARN  %s\n' "$*" >&2; }
step() { printf '\n== %s\n' "$*"; }

summary() {
  printf '\n==================== SUMMARY ====================\n'
  local line
  for line in "${CHECKS[@]}"; do printf '%s\n' "$line"; done
  printf '=================================================\n'
  if [ "$FAILS" -ne 0 ]; then
    printf 'BOOTSTRAP FAIL — %d check(s) failed; fix them and rerun.\n' "$FAILS"
    exit 1
  fi
  printf 'BOOTSTRAP PASS — ready for: bash remote/generate.sh\n'
}

# ---------------------------------------------------------------- 1. preflight
step "1/8 host preflight"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_CSV="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)"
  if [ -z "$GPU_CSV" ]; then
    GPU_CSV="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || true)"
    GPU_CSV="${GPU_CSV},unknown"
  fi
  IFS=',' read -r GPU_NAME GPU_MEM_MIB GPU_CAP <<<"$(printf '%s' "$GPU_CSV" | tr -d ' ')"
  pass "NVIDIA GPU present: ${GPU_NAME} (${GPU_MEM_MIB} MiB)"

  if [[ "${GPU_CAP}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    if awk -v c="$GPU_CAP" 'BEGIN { exit !(c + 0 >= 8.0) }'; then
      pass "compute capability ${GPU_CAP} >= 8.0 (bf16 capable)"
    else
      fail "compute capability ${GPU_CAP} < 8.0; bf16 generation needs Ampere or newer"
    fi
  else
    warn "compute capability not reported by this driver; torch will confirm after install"
  fi

  if [ "${GPU_MEM_MIB:-0}" -ge 16384 ] 2>/dev/null; then
    pass "VRAM ${GPU_MEM_MIB} MiB >= 16384 MiB"
    if [ "${GPU_MEM_MIB:-0}" -lt 24576 ]; then
      warn "VRAM below 24 GiB: batch 16 can OOM on long sentences; use --batch-size 8"
    fi
  else
    fail "VRAM ${GPU_MEM_MIB:-?} MiB < 16384 MiB"
  fi

  CUDA_VER="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n1)"
  if [[ "$CUDA_VER" == 12.* || "$CUDA_VER" == 13.* ]]; then
    pass "CUDA driver API ${CUDA_VER} (12.x or newer; cu12 wheels run on it)"
  elif [ -n "$CUDA_VER" ]; then
    fail "CUDA driver API ${CUDA_VER}; needs 12.x or 13.x (cu12 wheels)"
  else
    fail "cannot read the CUDA version from nvidia-smi"
  fi
else
  GPU_NAME="unknown"
  fail "nvidia-smi not found: no NVIDIA GPU / driver on this host"
fi

if [ -r /proc/meminfo ]; then
  RAM_GB="$(awk '/^MemTotal:/ { print int($2 / 1024 / 1024) }' /proc/meminfo)"
  if [ "${RAM_GB:-0}" -ge 32 ]; then
    pass "RAM ${RAM_GB} GiB >= 32 GiB"
  else
    fail "RAM ${RAM_GB:-?} GiB < 32 GiB"
  fi
else
  fail "/proc/meminfo not readable (not Linux?)"
fi

DISK_GB="$(df -Pk "$ROOT" | awk 'NR == 2 { print int($4 / 1024 / 1024) }')"
if [ "${DISK_GB:-0}" -ge 100 ]; then
  pass "free disk ${DISK_GB} GiB >= 100 GiB"
else
  fail "free disk ${DISK_GB:-?} GiB < 100 GiB"
fi

if command -v "$PYTHON" >/dev/null 2>&1 \
   && "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  pass "$PYTHON $("$PYTHON" -c 'import platform; print(platform.python_version())')"
else
  fail "Python 3.12 not found as '$PYTHON' (apt install python3.12 python3.12-venv)"
fi

if command -v ffmpeg >/dev/null 2>&1; then
  pass "ffmpeg $(ffmpeg -version 2>/dev/null | head -n1 | awk '{print $3}') (Whisper audio decode)"
else
  fail "ffmpeg not found; screening needs it (apt install ffmpeg)"
fi

if [ "$FAILS" -ne 0 ]; then
  summary
fi

# ---------------------------------------------------------------- 2. venv
step "2/8 Python environment (.venv-remote)"
export HF_HOME="$ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$ROOT/.cache"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_LOCAL_DIR_USE_SYMLINKS=0
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"

REQ_SHA="$("$PYTHON" -c 'import hashlib; print(hashlib.sha256(open("remote/requirements-remote.txt","rb").read()).hexdigest())')"
STAMP="$VENV/.requirements.sha256"
DEPS_OK=0
if [ -x "$VENV/bin/python" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$REQ_SHA" ] \
   && "$VENV/bin/python" -c 'import torch, qwen_tts, peft, faster_whisper, ruaccent, espeakng_loader, soundfile, librosa, torchaudio' >/dev/null 2>&1; then
  DEPS_OK=1
fi
if [ "$DEPS_OK" -eq 1 ]; then
  pass "pinned dependencies already installed (requirements unchanged)"
else
  if [ ! -x "$VENV/bin/python" ]; then
    printf '[venv] creating %s\n' "$VENV"
    "$PYTHON" -m venv "$VENV"
  fi
  "$VENV/bin/python" -m pip install --quiet --upgrade pip wheel setuptools
  printf '[venv] installing remote/requirements-remote.txt (this downloads torch)\n'
  "$VENV/bin/python" -m pip install -r remote/requirements-remote.txt
  printf '%s\n' "$REQ_SHA" >"$STAMP"
  pass "pinned dependencies installed"
fi

if "$VENV/bin/python" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False (CPU-only wheel or no driver)")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
if major < 8:
    sys.exit(f"compute capability {major}.{minor} < 8.0")
if not torch.cuda.is_bf16_supported():
    sys.exit("bfloat16 is not supported on this GPU")
print(f"[cuda] torch {torch.__version__}, {name}, sm_{major}{minor}, bf16 ok, "
      f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
PY
then
  pass "torch sees CUDA and bfloat16"
else
  fail "torch cannot use CUDA/bfloat16; check the driver and the torch wheel"
fi

# ---------------------------------------------------------------- 3. assets
step "3/8 model assets (download + SHA-256)"
if "$VENV/bin/python" remote/lib/fetch_assets.py --root "$ROOT"; then
  pass "all pinned assets verified"
else
  fail "asset download/verification failed"
fi

# ---------------------------------------------------------------- 4. env file
step "4/8 bundle environment"
cat >"$ROOT/remote/bundle.env" <<EOF
# Generated by remote/bootstrap.sh for this checkout. Safe to delete; rerun bootstrap.
export VOCA_ROOT="$ROOT"
export VOCA_CACHE="$ROOT/.cache"
export VOCA_MODELS="$ROOT/.cache/models"
export VOCA_RUACCENT_WORKDIR="$ROOT/.cache/models/ruaccent"
export HF_HOME="$ROOT/.cache/huggingface"
export XDG_CACHE_HOME="$ROOT/.cache"
export HF_HUB_DISABLE_TELEMETRY=1
export VOCA_TTS_DEVICE="cuda:0"
export VOCA_ASR_DEVICE="cuda"
export PYTHONPATH="$ROOT/training"
export PATH="$ROOT/.venv-remote/bin:\$PATH"
EOF
# shellcheck disable=SC1091
. "$ROOT/remote/bundle.env"
pass "wrote remote/bundle.env"

# ---------------------------------------------------------------- 5. text check
step "5/8 text frontend (F3: RUAccent + eSpeak NG)"
if "$VENV/bin/python" - <<'PY'
import sys
sys.path.insert(0, "training")
from voca_ru.text import analyze

result = analyze("Сегодня двадцать третьего января мы уезжаем в отпуск.")
marks = result["stressed_text"].count("\u0301")
assert marks >= 5, f"only {marks} stress marks"
assert result["phonemes"], "no phonemes"
print(f"[text] {result['stressed_text']}")
print(f"[text] {marks} marks, {len(result['phonemes'])} phones")
PY
then
  pass "text frontend produces stress marks and phonemes"
else
  fail "text frontend failed (RUAccent dictionary / eSpeak NG)"
fi

# ---------------------------------------------------------------- 6. smoke
step "6/8 one-clip smoke through the real pipeline"
if "$VENV/bin/python" remote/lib/smoke.py --root "$ROOT"; then
  pass "smoke clip accepted (Base + LoRA, WER, meter2 stress)"
else
  fail "smoke clip failed"
fi

# ---------------------------------------------------------------- 7. disk again
step "7/8 disk after downloads"
DISK_GB="$(df -Pk "$ROOT" | awk 'NR == 2 { print int($4 / 1024 / 1024) }')"
if [ "${DISK_GB:-0}" -ge 100 ]; then
  pass "free disk ${DISK_GB} GiB still >= 100 GiB (assets + smoke used ~8 GiB)"
else
  warn "free disk ${DISK_GB} GiB after downloads; the corpus itself needs ~6 GiB"
fi

# ---------------------------------------------------------------- 8. summary
step "8/8 summary"
summary
