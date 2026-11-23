# S1 — self-contained remote corpus-generation bundle

**Owner:** S1 · **Branch:** `task/s1-remote` · **Date:** 2026-11-24
**Scope:** make the D2 corpus stage runnable on a rented CUDA GPU start to
finish (`bootstrap → generate → watch → publish`), plus a local `pack.sh`.
Training is out of scope.

## TL;DR

- `remote/` ships five scripts, four small Python modules, a pinned asset lock,
  pinned requirements and a runbook. `bash remote/pack.sh` builds
  **`out/remote-bundle.tar.zst` (~1.0 MB, 63 files)** with no models, no audio
  caches, no `.cache/`. `pack.sh` prints the tarball's SHA-256 (timestamps make
  it differ between rebuilds; the content does not).
  with no models, no audio caches, no `.cache/`.
- **44 local tests pass** (`tests/remote/`), including a real `pack.sh` run and
  manifest/FLAC verification against the **real D2 pilot corpus**: 40 clips,
  107.73 s, WER mean 0.67 %, 184 stress words checked, 0 errors — identical to
  `reports/D2-corpus.md`.
- Asset pins were regenerated from the shared cache and cross-checked against
  the HF API. **D2's "corrupt xlsr" note was a hash-type confusion**: the
  file's SHA-256 is `d1cdb1a7…` (matches HF LFS oid and the file on disk);
  `5874e4e0…` is the *xetHash*, not a SHA-256. The xlsr checkpoint is intact.
- **Not verified on real CUDA**: device placement, VRAM at batch size, real
  throughput, GPU temp. See §7.

## 1. What was built

| Path | Job |
|---|---|
| `remote/bootstrap.sh` | idempotent preflight (GPU/compute cap/VRAM/RAM/disk/Python/CUDA/ffmpeg), `.venv-remote`, pinned deps, 6 assets + SHA-256 table, F3 text check, 1-clip smoke, PASS/FAIL summary |
| `remote/generate.sh` | full 16 000-clip run; resumable; `--limit`, `--voices`, `--batch-size`, `--split`, `--dry-run`; tees `run.log` |
| `remote/watch.sh` | 5-min health loop → `status.json`, `health.log`; alarms exactly as specified; exits 2 on alarm |
| `remote/publish.sh` | verify → dataset card → private HF repo + upload; refuses on failed verification / missing `HF_TOKEN` / incomplete corpus |
| `remote/pack.sh` | local bundle builder |
| `remote/lib/verify.py` | manifest↔FLAC verification + dataset card |
| `remote/lib/health.py` | pure status/alarm logic + watchdog CLI |
| `remote/lib/smoke.py` | 1-clip real-pipeline smoke and its validator |
| `remote/lib/fetch_assets.py` | grouped `hf download` + per-file SHA-256 table |
| `remote/lib/bundle_files.py` | canonical bundle file list (used by `pack.sh`) |
| `remote/assets.lock` | 33 pinned files: 29 HF + 4 bundled voice files |
| `remote/requirements-remote.txt` | torch 2.15.0, qwen-tts 0.1.1, peft 0.21.3, transformers 4.57.3, faster-whisper 1.2.1, ctranslate2 4.9.0, ruaccent 1.5.8.3, espeakng-loader 0.2.4, onnxruntime 1.32.0, huggingface_hub 0.36.2 … |
| `remote/README.md`, `remote/bundle-README.md` | runbook, hardware table, first-10-minutes list, troubleshooting |
| `tests/remote/` | `test_verify`, `test_health`, `test_bundle`, `test_assets`, `test_smoke`, `test_shell` |

### The six assets (`assets.lock`)

| # | HF repo | Files |
|---|---|---|
| 1 | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 11 (weights 3.86 GB + speech tokenizer 0.68 GB + tokenizer/config) |
| 2 | `siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf` | `lora/adapter.{json,safetensors}` (34.9 MB) |
| 3 | same repo | `meter/{head.pt,meter2.py}` |
| 4 | `jonatasgrosman/wav2vec2-large-xlsr-53-russian` | 5 (1.26 GB CTC encoder) |
| 5 | `Systran/faster-whisper-large-v3-turbo` | 5 (1.62 GB CTranslate2) |
| 6 | `ruaccent/accentuator` | `dictionary/*.json.gz` (21.9 MB) |

Total download ≈ 7.49 GB. The four voice files (`f_neutral`, `m_neutral`
`.wav`/`.txt`) are shipped in the tarball. Every file is size+SHA-256 verified
on every bootstrap; cached files that already match are not re-downloaded.

## 2. Changes outside `remote/`

Needed for CUDA, kept minimal, local defaults unchanged:

- `scripts/gen/build_corpus.py`
  - `VOCA_MODELS` (model root), `VOCA_TTS_DEVICE` (default `mps`),
    `VOCA_ASR_DEVICE` (default `mps`), `--split all`, `--require-lora`;
    `mps_gib()` → device-aware `device_gib()`; `run-stats.json` key
    `peak_mps_gib` → `peak_device_gib`.
  - `--require-lora` makes a failed LoRA load abort, per the task contract.
    Without the flag the D2 behaviour (loud banner, continue on Base) is kept.
- `training/voca_ru/eval/asr.py`: accept `device="cuda"` for CTranslate2.
- `.gitignore`: ignore `remote/bundle.env` (generated on the server).

These two are owned by D2/F5; the orchestrator may prefer to move
`--require-lora` into a remote-only wrapper. It is 4 lines and documented here.

## 3. Resumability and safety

- `generate.sh` delegates resume to `build_corpus.py`: accepted and dropped ids
  are re-read from `manifest.jsonl` + `review.jsonl`; batch seeds are derived
  from ids/voice/attempt, so a restart repeats a batch identically. A kill
  between the FLAC move and the manifest append leaves an orphan FLAC that the
  rerun overwrites; it never leaves a manifest row without audio.
- `watch.sh` opens the corpus read-only. It writes only `status.json`,
  `health.log`, `status-history.jsonl`.
- `publish.sh` verifies before writing the card and again before uploading;
  the HF repo is created private (`--public` opts out); `rejected/`, logs and
  status files are excluded from the upload.

## 4. Tests (run locally, no GPU)

```sh
.cache/venvs/r4/bin/python -m unittest discover -s tests/remote -v
# 44 tests, OK (~0.3 s)
```

Coverage:

- **Manifest verification**: valid corpus, missing file, orphan FLAC, duplicate
  id, duration mismatch, wrong sample rate, path traversal, dataset card.
- **Health thresholds**: each alarm at/around its boundary, all six at once,
  first-10-minute baseline, first sample rate = 0, status fields from real files.
- **Bundle**: every required file has a source; a real `pack.sh` run produces
  <50 MB, contains the expected files and no cache; staging passes
  `bundle_files.check_stage`.
- **Assets**: six sources, 33 well-formed entries, destinations match the paths
  the pipeline reads, local SHA-256 spot-check against the shared cache.
- **Smoke validator**: good clip passes; Base-without-LoRA, missing marks and
  empty manifest fail.
- **Publish gate**: drops count towards completion; a truncated corpus fails.
- **Shell**: `bash -n` on all five scripts, `set -euo pipefail`, no `read -p`,
  `--help` exits 0, unknown flag exits 2, `generate.sh --dry-run` resolves the
  full command (limit/voices/batch/require-lora/split).

Real-data check (read-only) against `worktrees/d2-corpus/data/corpus-ru`:

```sh
python remote/lib/publish.py --root . --corpus <d2-pilot> --check --allow-partial
# verification PASS; 40 clips, 0.03 h, WER 0.67 %, 184 stress words, 0 errors
```

`bootstrap.sh` on this Mac correctly stops at preflight with
`BOOTSTRAP FAIL` (no NVIDIA GPU, no `/proc/meminfo`, no `python3.12` on PATH).

## 5. Estimated server time and cost

Target: 8 000 sentences × 2 voices = 16 000 clips, ~19–20 h of audio.

| GPU | batch | expected wall | price band | cost |
|---|---:|---|---|---|
| RTX 4090 24 GB | 8 | 4–7 h | ~$0.35–0.7/h | ~$2–5 |
| L40S 48 GB | 16 | 3–5 h | ~$0.8–1.0/h | ~$3–5 |
| A100 40 GB | 16 | 2.5–5 h | ~$1.0–1.5/h | ~$3–7 |

Basis: G1 measured 2.43× realtime at batch 8 and 2.96–3.16× bucketed on M5;
D2's whole pilot (generation + Whisper + meter2) ran at 11.4 clips/min there.
A 4090 should beat MPS on generation but screening is sequential, so the range
is deliberately wide. Add ~10 min of downloads and ~2 min of smoke to the bill.

**Exact settings that move the price:** `--batch-size 8` default (16 on ≥40 GB),
length-bucketed batches (always on), bf16 eager attention, Whisper
`large-v3-turbo` float16 on CUDA, meter2 on CPU. Do the 40-clip smoke first
(~3 min) so a bad adapter/reference never burns hours.

## 6. Reproduce

```sh
cd worktrees/s1-remote
bash remote/pack.sh
.cache/venvs/r4/bin/python -m unittest discover -s tests/remote -v
# server:
#   tar -xf remote-bundle.tar.zst && cd remote-bundle
#   bash remote/bootstrap.sh
#   tmux new -s gen 'bash remote/generate.sh'
#   tmux new -s watch 'bash remote/watch.sh'
#   HF_TOKEN=... bash remote/publish.sh
```

## 7. What could only be verified on the server

1. `torch.cuda.is_available()`, compute capability ≥8, bf16 (bootstrap asserts).
2. `device_map="cuda:0"` through `qwen_tts` (the library documents it; not run).
3. VRAM at batch 8/16 with the longest bucket; OOM is not reproducible here.
4. Real throughput, `clips_per_min`/ETA in `watch.sh`, GPU temperature.
5. `hf download` of the six repos on Linux (network, Xet, gated-repo access;
   if any repo is gated, `HF_TOKEN` must also be exported for bootstrap).
6. The first real smoke clip: LoRA adherence, stress marks in audio, WER.
7. `hf` CLI upload flags (`--exclude`, `repo create --exist-ok`) against the
   live Hub.

## 8. Open issues / risks

1. **`torch==2.15.0` / `torchaudio==2.15.0` Linux CUDA wheel** is inferred from
   the macOS-verified pair. If the wheel is CPU-only, `bootstrap` fails loudly at
   the CUDA check; the fix is a PyPI index pin, not a silent degradation.
2. **Default batch 8** is chosen to survive 16–24 GB cards; G1/R4 suggest 16 is
   ~1.3–1.6× faster on ≥40 GB. `--batch-size 16` is documented.
3. **`ffmpeg` is a system dependency** of `FasterWhisperASR._decode_16k`; it is
   now a hard preflight check (`apt install ffmpeg`).
4. **D2's xlsr note** should be corrected in the D2 report: the file is intact;
   `d1cdb1a7…` is its SHA-256 and `5874e4e0…` its xetHash.
5. **`meter2` confidence** still limits stress auditing (same as D2 §8.5).
6. **`publish.sh` completeness gate** derives the expected clip count from the
   script files × the voices present in the manifest, so a deliberate one-voice
   run expects 8 000 clips and passes; a truncated two-voice run needs
   `--allow-partial`.
7. Local build artefacts: `out/remote-bundle.tar.zst` and `out/remote-bundle/`
   remain (git-ignored, 1 MB). The temporary `out/health-test/` copy was
   deleted.
