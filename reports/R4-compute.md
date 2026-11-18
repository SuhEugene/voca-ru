# R4 — compute environment and budget

Subagent R4, branch `task/compute-env`, 2026-11-19.
Machine: Apple M5 Pro, 24 GB, macOS 27.0.1.

Every number is **measured** unless it says *estimate*. Commands are in
`docs/compute.md` §10 or inline. Nothing was installed system-wide; all
toolchains and venvs live in `/private/var/folders/lr/…/T/opencode`.

---

## 0. Verdict

| Question | Answer |
|---|---|
| Does PyTorch exist for this machine's Python 3.15.2? | **Yes.** `torch` 2.15.0 publishes `cp315` arm64 wheels; installed and MPS verified. |
| Can Qwen3-TTS 1.7 B generate Russian here? | see §4 |
| Can the 4.5 M model be trained here? | **Yes, at ~6–26 h for 100 k–300 k steps.** MPS is ~6.5× the CPU. |
| Is the Rust repo buildable/testable? | **Yes**, with a throwaway rustup: build 25 s, 2.9 MB binary, 7/7 tests pass. |
| Biggest practical blocker? | **Network**, not compute: PyPI is ~0.12 MB/s and resets, HF ~2.2 MB/s. |

---

## 1. What I measured

Full table in `docs/compute.md`. The items that changed a decision:

- **Torch on 3.15 works** (`torch.backends.mps.is_available() == True`). The
  documented "3.15 is too new" worry was wrong — but see §4.1 for the
  ecosystem caveat that still forces 3.12 for the teacher stack.
- **MPS ≈ 6.5× CPU** on a 4.3 M transformer, and it scales to batch 16×512
  (62 k frames/s). CPU collapses at batch 16 (13× slower) — a real cliff, not
  noise; it reproduced in every run and both model sizes.
- **MPS allocatable ceiling is 30.19 GiB** (a 10 GiB tensor allocates; 12 GiB
  fails). So the 24 GB is soft — but the extra 6 GB is swap, and using it
  would wreck the teacher/training overlap.
- **`cargo` is not installed.** I used a temp `rustup`:
  `cargo build --release -p tiny-voca` = 25.3 s, binary 2.9 MB, `cargo test`
  7 tests green, 7.9 s.
- **Missing tools:** sox, espeak-ng, git-lfs, `hf`. ffmpeg 9.0.2 is present
  with flac/mp3/aac/loudnorm/silencedetect, so resampling and loudness work.
- **No credentials:** `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `WANDB_API_KEY`
  all unset. Everything in the plan is openly downloadable, but there is no
  W&B logging and no gated-repo access.
- **Disk:** 151 GiB free at the start (141 GiB by the end of my downloads).
  Enough, but with 84 % used there is no room for a duplicate corpus.
- **ffmpeg throughput:** 10 min of 24 kHz audio resampled to 22.05 kHz in
  0.12 s (~5000× realtime); single-pass `loudnorm` over the same 10 min in
  8.0 s (~75× realtime). Post-processing 19 h of teacher output ≈ 15 min.

## 2. Training budget

4.3 M-param block, `fwd+bwd+AdamW`, fp32, batch 8 × 512 frames = 47.6 s of
audio per step. Measured 69.4 ms/step on MPS, 450.8 ms/step on CPU.

Derated 3× for a real loop (dataloader, features, DSP head, checkpointing) and
4.5× for the GAN phase:

| Steps | MPS, no GAN | MPS, with GAN | CPU, no GAN |
|---:|---:|---:|---:|
| 100 k | 5.8 h | 8.6 h | 38 h |
| 104 k | 6.0 h | 9.0 h | 39 h |
| 300 k | 17.3 h | 26 h | 113 h |

Rented A10G (~$1.0/h) or 4090 (~$0.7/h), *estimate* assuming 3–5× MPS:
100 k steps ≈ 1.5–2.5 h ≈ **$1–2**; 300 k ≈ 5–7 h ≈ **$4–5**. The GPU is not
about money at this model size; it is about iteration speed.

Hard limits: ~10–20 M params trainable; batch 16×512 on MPS, 8×512 on CPU;
≤ ~100 h corpus on this disk; one GPU job at a time (two MPS jobs share the
16-core GPU and slow each other proportionally).

## 3. Micro-benchmark details

`bench_torch.py` (appendix) builds the model inline, so nothing depends on
training code that does not exist yet. Two shapes:

| Model | Device | batch × frames | ms/step | frames/s |
|---|---|---:|---:|---:|
| 4.30 M | MPS | 16 × 512 | 131.4 | 62,364 |
| 4.30 M | CPU | 8 × 512 | 450.8 | 9,086 |
| 4.30 M | CPU | 16 × 512 | 5,847.5 | 1,401 |
| 1.24 M | MPS | 16 × 512 | 51.5 | 159,079 |
| 1.24 M | CPU | 8 × 512 | 140.5 | 29,158 |

Raw matmul 4096³ fp32: CPU 1.67 TFLOP/s, MPS 5.77 TFLOP/s. The 62 k frames/s
step is ~1.6 TFLOP/s of useful work ≈ 28 % of that peak, which is a normal
efficiency for small tensors and makes the number believable.

Caveat: the machine was under load 4–6 (another agent) the whole time, so
these are lower bounds; and MPS-vs-CPU ratios were stable across runs, which
is what the projections use.

## 4. Qwen3-TTS 1.7 B teacher

### 4.1 Which Python

`qwen-tts` 0.1.1 pins `transformers==4.57.3` and imports `onnxruntime`,
`torchaudio`, `librosa`, `sox`, `soundfile`, `accelerate`.

- **onnxruntime has arm64 wheels only up to cp314** (cp311/312/313/314). No
  3.15 wheel exists. Even though its use is confined to the 25 Hz tokenizer
  path, `import` of the package happens at module scope.
- `torchaudio` 2.11.1 ships `py3-none-macosx_14_0_arm64` (pure), so 3.15 is
  fine there.
- `sox` (PyPI) has no 3.15 wheel; it is a thin wrapper over the `sox` binary,
  which is **not installed here**.

So: **plain PyTorch on 3.15.2 works, but the TTS stack does not.** Use 3.12
(qwen-tts' own recommendation) — `uv` already has `cpython-3.12.16` cached.
Recipe:

```sh
uv venv --python 3.12 .venv-teacher
uv pip install --python .venv-teacher/bin/python torch==2.15.0
uv pip install --python .venv-teacher/bin/python qwen-tts
```

If torch download stalls on files.pythonhosted, pull it from the PyTorch R2
mirror, which measured **2.07 MB/s** (17× faster) instead of ~0.12 MB/s:

```sh
curl -L -C - -o torch.whl \
  https://download-r2.pytorch.org/whl/cpu/torch-2.15.0-cp312-cp312-macosx_14_0_arm64.whl
uv pip install --python .venv-teacher/bin/python ./torch.whl
```

Verified: `torch 2.15.0  mps True` on 3.12.16. Save that mirror URL in the
runbook; it is the difference between 90 minutes and 90 seconds of downloads.

### 4.2 Result

*(filled in after the generation run — see below)*

## 5. Cost/benefit summary for the orchestrator

*(filled in with §4.2)*
