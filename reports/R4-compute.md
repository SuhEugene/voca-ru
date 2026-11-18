# R4 — compute environment and budget

Subagent R4, branch `task/compute-env`, 2026-11-19.
Machine: Apple M5 Pro, 24 GB, macOS 27.0.1. Everything below was measured on
this box unless marked *estimate*.

---

## 0. Verdict

| Question | Answer |
|---|---|
| Does PyTorch work on the system Python 3.15.2? | **Yes** — `torch` 2.15.0 has a `cp315` arm64 wheel, MPS available. |
| Can Qwen3-TTS 1.7 B generate Russian here? | **Yes, on MPS in bfloat16.** 5.0× realtime at batch 16. |
| Which Python for the teacher? | **3.12**, because `onnxruntime` has no cp315 wheel (torch itself is fine on 3.15). |
| Can the 4.5 M model be trained here? | **Yes, 6–26 h for 104 k–300 k steps** on MPS. |
| Rust repo build/test? | **Green** with a throwaway rustup: 25 s build, 2.9 MB binary, 7/7 tests. |
| 19.1 h of teacher audio? | **≈ 4 h** on this laptop at batch 16. No rented GPU needed. |
| Biggest practical blocker | **Network**: PyPI ~0.12 MB/s single-stream, HF single-stream ~tens of KB/s → must use parallel ranged fetches. |

## 1. Machine and environment

| Property | Value |
|---|---|
| SoC / model | Apple M5 Pro, `Mac17,9` |
| CPU | 15 cores = 5 performance + 10 efficiency |
| GPU | 16 cores, Metal 4 |
| Memory | 24 GiB unified |
| OS | macOS 27.0.1, Darwin 27.0.0, arm64 |
| torch threads | 5 (the performance cores) |
| Swap | 2 GiB total, ~0.5 GiB used |
| Disk free | 151 GiB at start, **134 GiB** after the teacher download |

Load average was 3.5–6 the whole session (other agents on the same box, plus a
Fan Control helper burning half a core), so the throughput numbers are lower
bounds. MPS-vs-CPU ratios were stable across runs, which is what the
projections rely on.

## 2. What I measured, and what it settles

- **Torch on 3.15.2 works.** `torch 2.15.0 | mps True`. The rumour that 3.15
  is too new was wrong.
- **MPS is ~6.5× the CPU** on a 4.3 M transformer.
- **CPU falls off a cliff at batch 16 × 512** (450 ms → 5,848 ms/step). This
  reproduced in every run and both model sizes. Do not raise the CPU batch.
- **The MPS allocatable ceiling is 30.19 GiB**, not 24 GiB: a 10 GiB tensor
  allocates, 12 GiB raises `RuntimeError: MPS backend out of memory … max
  allowed: 30.19 GiB`. The extra 6 GB is swap; treat 24 GB as the real budget.
- **`cargo` is absent.** Built with a temp rustup: release build 25.3 s,
  `tiny-voca` binary 2.9 MB, `cargo test` 7 tests / 0 failures in 7.9 s.
- **Missing tools**: sox binary, espeak-ng binary, git-lfs, `hf`. ffmpeg 9.0.2
  is present and covers the audio pipeline (10 min resampled in 0.12 s,
  `loudnorm` over the same file in 8.0 s).
- **espeak-ng is obtainable without brew**: `uv pip install espeakng-loader`
  ships `libespeak-ng.dylib` + the data tree; 128 voices, Russian `ru` and
  `ru-lv` both present. Verified.
- **No credentials set**: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`,
  `WANDB_API_KEY` all unset. Nothing in the plan is gated, but there is no W&B
  logging. `~/.cache/huggingface` exists.

## 3. Training budget (measured, then derated)

4.3 M-param block, fp32, `fwd+bwd+AdamW`, batch 8 × 512 frames = 47.6 s of
audio per step. Measured **69.4 ms/step on MPS**, **450.8 ms/step on CPU**.
Raw matmul 4096³ fp32: CPU 1.67 TFLOP/s, MPS 5.77 TFLOP/s; the step is
~1.6 TFLOP/s of useful work ≈ 28 % of peak, a normal efficiency for small
tensors.

Derated 3× for a real loop (dataloader, feature extraction, DSP head,
checkpointing) and 4.5× once the GAN phase is included:

| Steps | MPS, no GAN | MPS, with GAN | CPU, no GAN |
|---:|---:|---:|---:|
| 100 k | 5.8 h | 8.6 h | 38 h |
| 104 k | 6.0 h | 9.0 h | 39 h |
| 300 k | 17.3 h | 26 h | 113 h |

Rented A10G (~$1.0/h) or 4090 (~$0.7/h), *estimate* at 3–5× MPS: 100 k steps
≈ 1.5–2.5 h ≈ **$1–2**; 300 k ≈ 5–7 h ≈ **$4–5**. At this model size a GPU
buys iteration speed, not savings.

Limits: ~10–20 M params trainable; batch 16 × 512 on MPS, 8 × 512 on CPU;
≤ ~100 h corpus on this disk; one GPU job at a time.

## 4. Qwen3-TTS 1.7 B teacher — the answer

**It runs on MPS, end to end, in bfloat16. No CPU fallback, no failing op.**

Setup:

```sh
. "$(git rev-parse --show-toplevel)/scripts/cache-env.sh"
uv venv --python 3.12 .cache/venvs/r4
uv pip install --python .cache/venvs/r4/bin/python torch==2.15.0 qwen-tts
```

Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (Apache-2.0, not gated) at
`.cache/models/qwen3-tts-1.7b`, 4.4 GB on disk (3.86 GB transformer +
0.68 GB 12 Hz tokenizer + tokenizer files).

| Measurement | Result |
|---|---|
| Load (`device_map="mps"`, `dtype="auto"`) | works, 1.9–2.3 s warm / 7.5 s cold, **bfloat16** |
| MPS memory after load | 4.76 GiB driver-allocated |
| Voice prompt (`x_vector_only_mode=True`) | 0.2–1.7 s, reused across batches |
| Batch 1, warm | 3.92 s audio / 6.53 s → 0.60× realtime |
| Batch 8 | 28.9 s / 8.74 s → 3.31× realtime, 6.1 GiB |
| **Batch 16** | **56.7 s / 11.3 s → 5.04× realtime, 11.2 GiB** |
| Batch 32 | 123.4 s / 20.1 s → 6.13× realtime, 21.8 GiB |
| 199-char sentence, batch 1 | 14.32 s audio, RTF 1.60, peak 0.53, RMS 0.079 |
| Output | 24 kHz mono float32 |

Non-fatal warnings only: `sox: command not found` (used solely by the 25 Hz
tokenizer path, which the 12 Hz model never touches) and `flash-attn is not
installed` (expected on Apple silicon; `eager` attention, already in the
numbers above).

### 4.1 Generation budget for 19.1 h (17,371 clips, 8,500 sentences × 2 voices)

| Batch | Speed | Wall time for 19.1 h | Memory |
|---:|---:|---:|---:|
| 8 | 3.31× | 5.8 h | 6.1 GiB |
| **16** | **5.04×** | **3.8 h** | 11.2 GiB |
| 32 | 6.13× | 3.1 h | 21.8 GiB |

Recommended: **batch 16, ~4 h**, ≈ 69 characters/s, ≈ 0.71 s per clip.
Batch 32 is 20 % faster but eats 21.8 of 24 GiB — not worth it.

Voice design adds minutes. Whisper screening is **not measured**; *estimate*
1.5–2.5 h for 19.1 h with faster-whisper large-v3 int8 on the 5 P-cores.

Rented GPU for generation (*estimate*): ~1 h, ~$1 on a 4090 — but it would
mean uploading/downloading 4.5 GB on a 2 MB/s link, so generating locally is
the better call.

Generation and training do not overlap usefully: same 16-core GPU, no
isolation. Serialise (4 h generation, then training).

## 5. Checkpoint integrity — read this

The `model.safetensors` I tested with does **not** match the Hugging Face LFS
sha256 (`38fc7fc51c5e…`); it hashes to `ac725cd734da…`. Three 1 MB ranges
sampled across the file (offset 0, 1.9 GB, and the final MB) are byte-identical
to the origin, so the file is not wholesale corrupt — but a full-file
verification did not pass, and a re-fetch with the project's
`scripts/fetch-hf-file.sh` was still running when I finished.

What that means: the *timings, memory figures and the fact that MPS works* are
sound and will not change for a corrected checkpoint. Audio quality from this
particular copy is unverified — listen to the samples before trusting the
corpus. Re-verify with:

```sh
shasum -a 256 .cache/models/qwen3-tts-1.7b/model.safetensors
# expected 38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c
```

and re-run one batch afterwards.

## 6. Artefacts and paths

| Path | Size | What |
|---|---|---|
| `.cache/models/qwen3-tts-1.7b` | 4.4 GB | teacher weights + 12 Hz tokenizer; `clone_ref.wav` (16 s reference) |
| `.cache/venvs/r4` | 1.3 GB | Python 3.12.16, torch 2.15.0, qwen-tts 0.1.1 |
| `out/r4/ru-sample-{1,2,3}.wav` | 0.6 MB | three short Russian sentences |
| `out/r4/ru-sample-long.wav` | 0.7 MB | a 199-character sentence, 14.3 s |

`out/` is git-ignored, as it must be (audio is never committed). I did not
create `out/r4/run_teacher.py`; it belongs to another agent and was left alone.

Cleanup: my temporary directories
(`/private/var/folders/…/opencode/{r4,qwen312,py315,mlxenv}`) are gone — they
were removed during the session and nothing of mine remains outside the repo.
The repo `.cache` grew by the teacher weights (4.4 GB) and one venv (1.3 GB);
disk free went from 151 GiB to 134 GiB.

Still on disk, not mine to delete: `parts-model.safetensors/` and an empty
`model.safetensors.new` inside the model directory — staging files of the
in-progress verified re-fetch.

## 7. Reproducing

```sh
. "$(git rev-parse --show-toplevel)/scripts/cache-env.sh"
uv venv --python 3.15 /tmp/bench && uv pip install --python /tmp/bench/bin/python torch
# training micro-benchmark: synthetic 4.3 M block, fwd+bwd+AdamW
#   bench_torch.py mps 256 8   → 69 ms/step at batch 8 × 512 frames
#   bench_torch.py cpu 256 8   → 451 ms/step
```

The benchmark script is quoted in `docs/compute.md` §10; it builds the model
inline so it does not depend on training code that does not exist yet.

## 8. Open issues

1. Full sha256 of the teacher checkpoint is unverified (§5) — re-fetch was in
   flight. This is the one thing to re-check before bulk generation.
2. Whisper screening throughput is an estimate, not a measurement.
3. `git-lfs` and the `hf` CLI are absent; parallel ranged `curl` covers the
   need but the shared `scripts/fetch-hf-file.sh` should be the default.
4. The 30.19 GiB MPS ceiling is above physical RAM, so a runaway batch will
   swap before it errors. Cap batch sizes rather than relying on the OOM.
