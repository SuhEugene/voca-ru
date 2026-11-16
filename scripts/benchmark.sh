#!/usr/bin/env bash
# Speed comparison: Rust port vs the PyTorch reference vs the published ONNX model.
#
#   bash scripts/benchmark.sh            # writes out/bench/*.json and BENCHMARK.md
#
# Needs: weights/ (see README), .venv with the reference installed, and
# out/reference from scripts/dump_reference.py. Run on an idle machine.
set -euo pipefail
cd "$(dirname "$0")/.."
export DIZIONE_DATA=weights/phonemizer
PY=.venv/bin/python
CARGO="${CARGO:-$HOME/.cargo/bin/cargo}"
RUNS="${RUNS:-30}"
mkdir -p out/bench

"$CARGO" build --release -p tiny-voca 2>&1 | tail -1
"$CARGO" build --release -p tiny-voca --no-default-features --target-dir target/mm 2>&1 | tail -1

# 1. Model only: identical token inputs, median of $RUNS runs after warm-up.
for threads in 1 4; do
  ./target/release/bench weights/tiny-voca.safetensors out/reference --threads $threads --runs "$RUNS" --json > out/bench/rust-$threads.json
  ./target/mm/release/bench weights/tiny-voca.safetensors out/reference --threads $threads --runs "$RUNS" --json > out/bench/rustmm-$threads.json
  $PY scripts/bench_python.py pytorch $threads "$RUNS" > out/bench/pytorch-$threads.json
  $PY scripts/bench_python.py onnx $threads "$RUNS" > out/bench/onnx-$threads.json
done

# 2. Whole command, cold process: text in, WAV out.
$PY scripts/bench_cli.py > out/bench/cli.json

# 3. Phonemizer only.
$PY scripts/bench_phonemizer.py > out/bench/phonemizer.json

$PY scripts/bench_report.py > BENCHMARK.md
cat BENCHMARK.md
