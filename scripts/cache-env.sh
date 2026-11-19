# Source this before any command that may download a model, wheel or toolchain.
# Everything lands in <repo>/.cache so parallel subagents and later sessions
# reuse it instead of re-downloading on a slow link.
#
#   . "$(git rev-parse --show-toplevel)/scripts/cache-env.sh"
#
ROOT=/Users/suheugene/projects/22-tiniest-voca-ru
export VOCA_CACHE="$ROOT/.cache"
export XDG_CACHE_HOME="$VOCA_CACHE"
export HF_HOME="$VOCA_CACHE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export UV_CACHE_DIR="$VOCA_CACHE/uv"
export UV_PYTHON_INSTALL_DIR="$VOCA_CACHE/python"
export PIP_CACHE_DIR="$VOCA_CACHE/pip"
export TORCH_HOME="$VOCA_CACHE/torch"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$VOCA_CACHE/models" "$VOCA_CACHE/venvs"
