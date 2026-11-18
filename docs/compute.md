# Compute environment and budget — Apple M5 Pro, 24 GB

Measured by subagent R4 on 2026-11-19 inside
`worktrees/compute-env` (branch `task/compute-env`). Every number below is
either **measured** (command shown or reproducible from the appendix) or
labelled **estimate** with its assumption. Nothing here is a guess dressed up
as a measurement.

Headline answers:

1. **PyTorch works on this machine's Python 3.15.2.** `torch` 2.15.0 ships a
   `cp315` macOS-arm64 wheel, and MPS is available. Exact command in
   §3. There is no “newest Python that still has wheels” problem here.
2. **Rust is not installed**, but a throwaway `rustup` in the temp dir builds
   and tests the repo cleanly (`cargo build --release` 25 s, binary 2.9 MB,
   `cargo test` green).
3. **Training the 4.5 M model locally is viable** — the transformer stack runs
   at ~62 k frames/s on MPS (≈ 720× realtime before derating). Expect tens of
   hours, not hundreds.
4. **The real bottleneck for the synthetic-teacher plan is not FLOPs, it is
   network throughput** — see §2.
5. Qwen3-TTS teacher results: see §9.

---

## 1. Machine

| Property | Value | How |
|---|---|---|
| SoC | Apple M5 Pro (`Mac17,9`) | `sysctl hw.model` |
| CPU cores | 15 = 5 performance + 10 efficiency | `sysctl hw.perflevel{0,1}.physicalcpu` |
| GPU cores | 16 | `system_profiler SPDisplaysDataType` |
| Metal | Metal 4 | `system_profiler SPDisplaysDataType` |
| Unified memory | 24 GiB (25,769,803,776 B) | `sysctl hw.memsize` |
| OS | macOS 27.0.1 (Darwin 27.0.0), arm64 | `sw_vers`, `uname -m` |
| torch CPU threads | 5 (= performance cores) | `torch.get_num_threads()` |

Notes:

- **The box was not idle while measuring.** Load average was 4–6 throughout
  (another agent on this machine, plus a Fan Control helper burning ~50 % of
  one core). Throughput numbers in §8 are therefore *lower bounds*; a quiet
  machine should be equal or better. Relative MPS-vs-CPU ratios are stable
  across repeated runs, which is what the estimates rest on.
- Swap is on and lightly used: `vm.swapusage: total 2048.00M used 535.44M`
  (`sysctl vm.swapusage`).

## 2. Disk and network

Disk — `df -h /`:

```
Filesystem      Size   Used  Avail Capacity  Mounted on
/dev/disk3s1s1  926Gi   13Gi  151Gi     8%    /
```

151 GiB free, confirmed. The plan needs ~4.5 GB teacher weights, ~6 GB of
generated audio, plus features and checkpoints — call it 15–20 GB. That fits
comfortably; the disk is *not* tight yet, but at 84 % used on the data volume
there is no room for a second full copy of the corpus.

Writable locations: `/Users/suheugene`, `/tmp`, and the session temp dir
`/private/var/folders/lr/…/T/opencode` (all verified with `test -w`).

Network — HTTP status and measured bulk throughput (`curl -r 0-30000000`
against a large file, so a real transfer and not just a handshake):

| Host | Status | Bulk throughput |
|---|---:|---:|
| huggingface.co | 200 | **~2.2 MB/s** (2,247,597 B/s) |
| github.com | 200 | — |
| openslr.org | 200 | — |
| pypi.org (index) | 200 | — |
| files.pythonhosted.org (wheels) | 200 | **~0.12 MB/s, connection resets** |

**This is the hard constraint on the teacher plan.** PyPI wheel downloads
stall and reset (a 132 MB torch wheel needed repeated `curl -C -` resumes);
Hugging Face is ~18× faster. 4.5 GB of teacher weights ≈ 35 min at 2.2 MB/s;
the same 4.5 GB from PyPI would be ~10 h. Budget the downloads, and prefer
`hf`/`curl -C -` over a bare `pip install`.

Measured mitigations:

| Source | Throughput |
|---|---:|
| `files.pythonhosted.org` (PyPI default) | ~0.12 MB/s, resets |
| `download-r2.pytorch.org` (PyTorch mirror, same wheel, sha256-identical) | **2.07 MB/s** |
| `huggingface.co` | **2.25 MB/s** |

For torch specifically, use the R2 mirror — it is 17× faster and the wheel is
byte-identical (sha256 checked):

```sh
curl -L -C - -o torch.whl \
  https://download-r2.pytorch.org/whl/cpu/torch-2.15.0-cp312-cp312-macosx_14_0_arm64.whl
uv pip install ./torch.whl
```

Bandwidth is shared: concurrent downloads from this machine slow each other
roughly proportionally, and other agents on this box were downloading at the
same time.

Measured on the teacher checkpoint later in the session: **16 parallel HTTP
ranges sustained ~0.8 MB/s aggregate** (498 MB in 10 min), i.e. ~80 min for
3.86 GB. Single-stream `curl -C -` on the same file stalled altogether. So
parallel ranges help but do not make this link fast — the link itself is the
bottleneck, and the 17× PyPI-mirror win above was a different CDN, not a
different technique.

## 3. Python and PyTorch — the critical finding

System Python is Homebrew **3.15.2** (`/opt/homebrew/bin/python3`).

**PyTorch 2.15.0 publishes a `cp315` macOS-arm64 wheel**, so 3.15.2 is *not*
too new:

```
$ curl -s https://pypi.org/pypi/torch/json | jq -r '.urls[].filename' | grep arm64
torch-2.15.0-cp311-cp311-macosx_14_0_arm64.whl
torch-2.15.0-cp312-cp312-macosx_14_0_arm64.whl
torch-2.15.0-cp313-cp313-macosx_14_0_arm64.whl
torch-2.15.0-cp314-cp314-macosx_14_0_arm64.whl
torch-2.15.0-cp314-cp314t-macosx_14_0_arm64.whl
torch-2.15.0-cp315-cp315-macosx_14_0_arm64.whl
torch-2.15.0-cp315-cp315t-macosx_14_0_arm64.whl
```

Verified working — this command was run, and the result is pasted below it:

```sh
uv venv --python /opt/homebrew/bin/python3.15 /tmp/py315
uv pip install --python /tmp/py315/bin/python "torch==2.15.0" numpy
```

```
torch 2.15.0
numpy 2.5.5
python 3.15.2
mps built True
mps available True
threads 5
device count 1
```

So: **`uv venv --python 3.15 && uv pip install torch` is the working PyTorch
recipe on this machine.** No fallback to an older Python is required.
`uv` 0.13.9 can also provision 3.13.18 / 3.12.16 on demand
(`uv python list` shows them already downloaded under
`~/.local/share/uv/python/`) if a project pins an older interpreter.

Package availability at 3.15 (checked, not installed): `torchaudio`,
`safetensors`, `soundfile`, `transformers` have current releases; `mlx`
0.32.4 publishes a cp315 wheel. See §9 for what was actually installed.

## 4. Rust toolchain

`cargo` is **not** installed system-wide (`cargo: command not found`,
no `~/.cargo`, no `~/.rustup`). Installed a throwaway toolchain instead, which
is not system-wide and does not touch `~/`:

```sh
export RUSTUP_HOME=/private/var/folders/…/T/opencode/rustup
export CARGO_HOME=/private/var/folders/…/T/opencode/cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
     --no-modify-path --profile minimal --default-toolchain stable
```

rustc/cargo **1.100.0 (0ef1bc386 2026-11-09)**, `stable-aarch64-apple-darwin`.

| Check | Result |
|---|---|
| `cargo build --release -p tiny-voca` (cold, LTO=fat, codegen-units=1) | **25.3 s** wall, 67 s user (3 cores busy) |
| Binary `target/release/tiny-voca` | **2.9 MB** (2,975,488 B), stripped |
| `target/release/voca` | 2.9 MB |
| `target/release/bench` | 928 KB |
| `cargo test` | **7 tests, 0 failed** (4 dizione parity, 2 tiny-voca unit, 1 doc-test), 7.9 s |

Baseline repo health is green. Note the release profile is LTO-fat, so a cold
build is a one-off 25 s — incremental rebuilds are much faster.

To make this permanent for the project (outside a subagent's temp dir):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# or: brew install rustup-init && rustup-init -y
```

## 5. MLX

`mlx` is **not installed** (no `mlx` in any interpreter; `import mlx` fails).
A cp315 wheel exists on PyPI (`mlx` 0.32.4, plus `mlx-metal`), so it would
install; the install itself did not complete here because `files.pythonhosted`
timed out repeatedly (§2). **Therefore no MLX matmul number is reported — this
is an unmeasured item, not a zero.**

Position: MLX is a credible fallback for *inference* of the 1.7 B teacher, but
it is not a drop-in for a PyTorch training loop with custom autograd ops. If
the MPS path fails on a specific op (§9), MLX is the next thing to try, at the
cost of porting the model.

## 6. Audio and data tooling

| Tool | State |
|---|---|
| ffmpeg | **9.0.2** — has `flac`, `libmp3lame`, `aac`, `pcm_s16le`, `loudnorm`, `aresample`, `silencedetect` |
| sox | **missing** binary |
| espeak-ng | **missing** binary — but see below, a wheel replacement works |
| git-lfs | **missing** (plain `git` 2.56.1 is present) |
| `hf` CLI | **missing** (no `huggingface_hub[cli]`) |
| node / npm | v26.13.1 / 11.20.0 |
| brew | 7.0.14 |

Missing pieces and their workarounds:

- **espeak-ng**: `espeak-ng --voices=ru` cannot run (binary absent), but the
  library can be obtained without a system install:

  ```sh
  uv pip install espeakng-loader espeakng
  ```

  Verified working: `libespeak-ng.dylib` + the full `espeak-ng-data` tree
  (including `ru_dict`) land in site-packages, 128 voices are listed, and
  **Russian voices `ru` and `ru-lv` are both available**. Use `espeakng-loader`
  + `phonemizer-fork` for Russian G2P rather than `brew install espeak-ng`.
- `git-lfs` is needed only if a corpus is distributed via LFS pointers.
- `hf` is convenient but not required: `curl -L -C -` on
  `https://huggingface.co/<repo>/resolve/main/<file>` works and resumes.
- `sox` is not a blocker: `ffmpeg` covers format conversion, resampling,
  loudness and trimming. Measured: 10 min of 24 kHz mono resampled to
  22.05 kHz in **0.12 s** (~5000× realtime), single-pass `loudnorm` over the
  same file in **8.0 s** (~75× realtime).

## 7. Credentials (presence only)

| Variable | State |
|---|---|
| `HF_TOKEN` | unset |
| `HUGGING_FACE_HUB_TOKEN` | unset |
| `WANDB_API_KEY` | unset |

`~/.cache/huggingface` **exists** and has a populated `hub/` directory, so
something has pulled models before, but no token is configured. Every model in
this plan (`Qwen/Qwen3-TTS-12Hz-1.7B-Base` is `gated:false`, Apache-2.0) is
downloadable anonymously, so this is not a blocker — but expect anonymous
rate limits, and there is no Weights & Biases logging without a key.

## 8. Training throughput — micro-benchmark

No training code exists yet in this repo, so throughput is bounded with a
synthetic model shaped like `crates/tiny-voca/src/model.rs`: pre-norm
transformer block, depthwise conv branch, rotary/plain attention, GELU FFN,
no biases, AdamW, fp32, `fwd + bwd + optimizer.step()`. Script:
`bench_torch.py` (appendix). Two shapes: the 4.3 M configuration that matches
`voca`'s parameter count, and the smaller 1.24 M 4-layer/192-dim block.

`frames` = acoustic frames at 22.05 kHz, hop 256 → **86.13 frames per second
of audio**.

**MPS (fp32, 4.3 M params):**

| batch | frames | ms/step | utt/s | frames/s |
|---:|---:|---:|---:|---:|
| 4 | 256 | 20.9 | 191 | 48,971 |
| 8 | 512 | 69.4 | 115 | 59,056 |
| 16 | 512 | 131.4 | 122 | 62,364 |
| 8 | 1024 | 166.1 | 48 | 49,321 |

**CPU (5 perf cores, fp32, 4.3 M params):**

| batch | frames | ms/step | utt/s | frames/s |
|---:|---:|---:|---:|---:|
| 4 | 256 | 171.7 | 23 | 5,966 |
| 8 | 512 | 450.8 | 18 | 9,086 |
| 16 | 512 | **5,847.5** | 3 | 1,401 |
| 8 | 1024 | 739.1 | 11 | 11,084 |

**1.24 M params (4 layers, 192 dim), for reference:**

| device | batch | frames | ms/step | frames/s |
|---|---:|---:|---:|---:|
| MPS | 16 | 512 | 51.5 | 159,079 |
| MPS | 8 | 512 | 28.0 | 146,277 |
| CPU | 8 | 512 | 140.5 | 29,158 |
| CPU | 16 | 512 | 2,072.3 | 3,953 |

Raw matmul for calibration (4096³ fp32): **CPU 1.67 TFLOP/s, MPS 5.77 TFLOP/s**. MPS fp16 works (tensors compute in fp16). At 62 k frames/s the 4.3 M
model does ~1.6 TFLOP/s of useful work, i.e. ~28 % of the measured MPS
matmul peak — a believable efficiency for small tensors.

Three findings that change how you use the machine:

1. **MPS is ~6.5× the CPU** on this model, and it scales with batch.
2. **CPU has a cliff at batch 16 × 512** (450 ms → 5,848 ms, 13× worse). It
   reproduced on every run, in both model sizes. Stay at batch ≤ 8 × 512 on
   CPU; do not “just raise the batch”.
3. **MPS allocatable memory ceiling is 30.19 GiB**, not 24 GiB: a 10 GiB
   tensor allocation succeeds and driver-allocated memory reaches 30.0 GiB
   before `RuntimeError: MPS backend out of memory … max allowed: 30.19 GiB`.
   macOS over-commits into swap. Treat ~24 GB as the no-swapping budget and
   30 GB as the hard wall.

MPS-MPS concurrency: two processes both get a slice of the same 16-core GPU;
there is no `CUDA_VISIBLE_DEVICES`-style isolation, so a teacher job and a
training job will slow each other roughly proportionally.

### Projected training time

Assumption set (stated, so it can be argued with):

- 4.5 M params, VITS-style: transformer stacks + durational/DSP head, **plus
  a GAN discriminator phase** later, which adds ~1.5× per step.
- Effective audio per step at batch 8 × 512 frames = 4,096 frames ≈ **47.6 s
  of audio**.
- Measured 69.4 ms/step on MPS is *optimistic*: it excludes the data loader,
  feature extraction, the DSP/source-filter head, GAN losses and
  checkpointing. Apply a **3× derating** for a real loop → ~0.21 s/step, and
  **4.5× during the GAN phase** → ~0.31 s/step.

| Steps | MPS, adversarial-free (3× derate) | MPS, with GAN phase (4.5× derate) | CPU (6.5× slower) |
|---:|---:|---:|---:|
| 100 k | **5.8 h** | 8.6 h | 38 h |
| 104 k (project figure) | 6.0 h | 9.0 h | 39 h |
| 300 k | 17.3 h | 26 h | 113 h |

Rented GPU, same job (approximate, public on-demand list prices as of
2026-11): an A10G (24 GB) is ~$1.0/h, a 4090 (24 GB) ~$0.7/h on spot
marketplaces. A 4090 is realistically 3–5× MPS for this model, so
100 k steps ≈ 1.5–2.5 h ≈ **$1–2**, and 300 k ≈ 5–7 h ≈ **$4–5**. In other
words: **renting a GPU to save a 6–26 h local run costs single-digit dollars**
— the deciding factor should be *iteration speed*, not price, and the
uncertainty is large because nothing was measured on a rented GPU here.

Hard limits this machine imposes:

| Limit | Value | Why |
|---|---|---|
| Largest trainable model | ~10–20 M params | 4.3 M runs at 62 k frames/s; batch 16 × 512 already leaves headroom, but activations grow linearly with params × batch × frames |
| Practical batch (MPS) | 16 × 512 frames; more with checkpointing | 30 GiB hard ceiling, no activation offload |
| Practical batch (CPU) | 8 × 512 frames | cliff at 16 (§8) |
| Dataset | ≤ ~100 h usable | 151 GB free, raw 16-bit 22.05 kHz mono = 159 MB/h plus features |
| Parallel runs | 1 GPU job + ~1 CPU job | shared 16-core GPU; CPU job must stay off the 5 P-cores to be useful |

## 9. Qwen3-TTS 1.7 B teacher on this machine

**Verdict: it runs on MPS, end to end, in bfloat16 — no CPU fallback needed.**
19.1 h of Russian audio is roughly a 4-hour job on this laptop.

Setup that worked (Python **3.12**, not 3.15 — see §9.1):

```sh
. "$(git rev-parse --show-toplevel)/scripts/cache-env.sh"
uv venv --python 3.12 .cache/venvs/r4
uv pip install --python .cache/venvs/r4/bin/python torch==2.15.0 qwen-tts
```

Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, Apache-2.0, `gated:false`,
**4.4 GB on disk** (3.86 GB transformer + 0.68 GB 12 Hz tokenizer + tokenizer
files). Path: `.cache/models/qwen3-tts-1.7b`.

Loading and generation, measured on this machine:

| Step | Result |
|---|---|
| `from_pretrained(device_map="mps", dtype="auto")` | **works**, 1.9–2.3 s warm, 7.5 s cold |
| dtype chosen by `auto` | bfloat16 |
| MPS memory after load | **4.76 GiB** driver-allocated |
| `create_voice_clone_prompt(x_vector_only_mode=True)` | 0.2–1.7 s, reusable across calls |
| Batch 1, warm | 3.92 s audio in 6.53 s → **0.60× realtime** |
| Batch 8 | 28.9 s audio in 8.74 s → **3.31× realtime**, 6.1 GiB |
| Batch 16 | 56.7 s audio in 11.3 s → **5.04× realtime**, 11.2 GiB |
| Batch 32 | 123.4 s audio in 20.1 s → **6.13× realtime**, 21.8 GiB |
| 199-char sentence, batch 1 | 14.32 s audio, RTF 1.60, peak 0.53 / RMS 0.079 (speech-like levels) |
| Output | 24 kHz mono float32, numpy arrays |
| Failing ops | **none on MPS** |

Two warnings, neither fatal:

- `sox: command not found` — printed by the `sox` Python package at import.
  It is only used by the 25 Hz tokenizer path, which the 12 Hz model does not
  touch. Generation works.
- `flash-attn is not installed` — expected on Apple silicon; `eager`
  attention is used and the speeds above already include that.

Sample rate 24 kHz, so a 19.1 h corpus is ~6.6 GB as mono float32 WAV,
~3.3 GB as 16-bit PCM, ~1.6 GB as FLAC.

### 9.1 Why 3.12 and not 3.15

Torch itself is fine on 3.15.2 (§3), but **`onnxruntime` publishes arm64
wheels only up to cp314**, and `qwen-tts` imports it at module scope. The `sox`
PyPI package likewise has no 3.15 wheel. So the teacher stack is pinned to
3.12, which `uv` provisions with no extra download (already in
`.cache/python`).

Corollary worth knowing: plain torch experiments (the training loop, the
micro-benchmark) can stay on 3.15, and the teacher needs 3.12. Two venvs, not
one.

### 9.2 Generation budget for the corpus

Plan: 17,371 clips, 19.1 h of audio, ~8,500 sentences × 2 voices.
`voice_clone_prompt` is built once per voice, so every batch skips prompt
encoding; batch within one voice.

| Batch | Speed | 19.1 h of audio | Memory |
|---:|---:|---:|---:|
| 8 | 3.31× | 5.8 h | 6.1 GiB |
| **16** | **5.04×** | **3.8 h** | 11.2 GiB |
| 32 | 6.13× | 3.1 h | 21.8 GiB |

Recommended operating point: **batch 16, ~4 h**, leaving 13 GiB for the OS and
any concurrent job. Batch 32 is 20 % faster and consumes almost the whole
machine; not worth the risk on a 24 GB box.

Throughput per character (measured, batch 16): 776 input characters produced
56.7 s of audio in 11.3 s — **≈ 69 characters/s**, ≈ 0.71 s per clip.
At batch 32 it is ≈ 83 characters/s.

Voice design: a handful of clips, seconds. Add ~5 minutes total.

Whisper screening (reject bad clips, check the transcript matches) — **not
measured here.** *Estimate:* faster-whisper large-v3 in int8 on the 5
performance cores runs at roughly 8–12× realtime for this kind of short-clip
batch, so 19.1 h of audio is a **1.5–2.5 h** pass. Treat as an estimate; it
needs one measurement before it is scheduled.

Rented-GPU comparison for generation (*estimate*): a 4090 at ~$0.7/h would run
the 1.7 B teacher perhaps 3–4× faster than MPS, i.e. **~1 h, ~$1**. Generation
is not the reason to rent a GPU — 4 hours on the laptop is acceptable and
avoids a 4.5 GB upload/download round trip on a 2 MB/s link.

### 9.3 Can generation and training overlap?

Only badly. Both want the same 16-core GPU; MPS gives no isolation, so a
teacher batch and a training step interleave and each runs slower. Memory is
not the constraint (11 GiB + ~2 GiB), attention is. Serialise them: 4 h of
generation, then the training run. If overlap is unavoidable, expect roughly
the sum of both wall times, not the max.

## 10. Appendix — reproducing the benchmark

`bench_torch.py` builds the model inline (no repo code is imported), so the
measurement does not depend on training code that does not exist yet. Create
the env with §3, then:

```sh
python bench_torch.py mps 256 8    # device, channels, layers
python bench_torch.py cpu 256 8
python bench_torch.py mps 192 4
```

The MPS allocatable-memory ceiling was measured by allocating 2, 4, …, 12 GiB
tensors in turn and printing `torch.mps.driver_allocated_memory()`.
