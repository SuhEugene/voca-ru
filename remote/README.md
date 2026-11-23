# Remote corpus generation (Qwen3-TTS teacher on a rented CUDA GPU)

This directory makes the **corpus generation stage** runnable on a rented Linux
GPU box with no manual steps beyond `HF_TOKEN` at publishing time. It generates
8 000 Russian sentences × 2 voices = **16 000 screened clips (~19–20 h)** for
training the Russian `voca` student. Model training is out of scope.

| Script | Runs where | Job |
|---|---|---|
| `bootstrap.sh` | server | preflight checks, `.venv-remote`, pinned deps, 6 assets + SHA-256, 1-clip smoke |
| `generate.sh` | server | full run (resumable, length-bucketed), `data/corpus-ru/` |
| `watch.sh` | server, 2nd tmux | `status.json` + `health.log` every 5 min, alarms |
| `publish.sh` | server (or local, same layout) | verify, dataset card, private HF upload |
| `pack.sh` | local | builds `out/remote-bundle.tar.zst` |
| `lib/` | both | verification, health thresholds, smoke check, asset fetch |

## Hardware

| | Minimum | Recommended | Why |
|---|---|---|---|
| GPU | RTX 4090 24 GB (16 GB floor) | L40S 48 GB / A100 40 GB | 1.7B bf16 + Whisper large-v3-turbo; batch 8 |
| Compute capability | 8.0 (Ampere) | 8.6+ | bf16 generation |
| System RAM | 32 GB | 64 GB | RUAccent dictionary + two models |
| Disk | 100 GB free | 150 GB | 7.5 GB assets, ~6 GB corpus, HF cache |
| OS / driver | Ubuntu 22.04/24.04, CUDA 12.x driver | same | torch 2.15.0 CUDA wheel |
| Python | 3.12 | 3.12 | pinned `venv` |

**Dollar-relevant settings:** teacher `bfloat16`, `attn_implementation=eager`,
batch size **8** with length bucketing (the safe default on 16–24 GB; use
`--batch-size 16` on a 40 GB+ card for more throughput), Whisper
`large-v3-turbo` **float16 on CUDA**, meter2 stress on CPU. Expect
**4–7 h on a 4090** (~$3–6 at typical on-demand prices), **2.5–5 h on an
A100 40 GB**. The GPU bills for bootstrap too: ~10 min of downloads and
~2 min of smoke.

## Runbook

On the rented box, as root or inside the provided container:

```sh
unzip/untar the bundle, cd remote-bundle
bash remote/bootstrap.sh
```

`bootstrap.sh` is idempotent. It prints one `PASS`/`FAIL` line per check and a
final summary; a failed check stops before generation. It ends by running one
real clip through the pipeline with the LoRA mandatory.

Then, in tmux:

```sh
tmux new -s gen
bash remote/generate.sh --limit 40      # 40 sentences x 2 voices, ~2-3 min
# listen to data/corpus-ru/audio/*/ (scp a few FLACs down), then:
bash remote/generate.sh                 # the full 16 000
```

Second session:

```sh
tmux new -s watch
bash remote/watch.sh
```

Partial runs: `--limit N` (sentences), `--voices f_neutral`, `--batch-size 8`,
`--split train`. `--dry-run` prints the exact command without running it.
Killing and restarting `generate.sh` is safe: accepted ids are read back from
`manifest.jsonl` and `review.jsonl` and skipped.

When generation finishes:

```sh
HF_TOKEN=hf_... bash remote/publish.sh --dry-run   # verify + stats + dataset card
HF_TOKEN=hf_... bash remote/publish.sh             # creates the private repo if missing
```

## What to check in the first 10 minutes

1. **`bootstrap.sh` summary is all PASS.** In particular `torch sees CUDA and
   bfloat16`, `all pinned assets verified`, and `SMOKE PASS` with
   `teacher_adapter` containing `lora`.
2. **The smoke clip has WER printed and stress marks.** `bootstrap` prints e.g.
   `PASS WER 0.00 %`, `PASS stress marks in prompt text: N`,
   `PASS meter2 stress: ok=True`.
3. **Generation starts at a sane rate.** `generate.sh` prints
   `[gen] f_neutral batch 1/… N clips …s`. On a 4090, expect 30–60 clips/min
   end-to-end; the first batches are short texts and look fastest.
4. **`watch.sh --once`** (before starting the loop) writes
   `data/corpus-ru/status.json` with `clips_done > 0`, `gpu_temp_c` present,
   `disk_free_gb > 90`, and no `ALARM` in `health.log`.
5. **VRAM headroom**: `nvidia-smi` should stay below ~20 GB on a 24 GB card. If
   it approaches the limit on long sentences, restart with `--batch-size 4`
   (resume keeps finished clips).
6. **Audio sanity**: pull 3 FLACs from `data/corpus-ru/audio/f_neutral/` and
   `m_neutral/`, listen; check `manifest.jsonl` `wer` values are mostly 0.

## Output

```
data/corpus-ru/
  manifest.jsonl        one accepted row per clip (text, norm_text with U+0301,
                        phonemes, voice, seconds, wer, stress_*, seed, split…)
  review.jsonl          dropped clips + reasons (created on the first drop)
  audio/<voice>/*.flac  24 kHz mono PCM16, peak 0.95
  rejected/<voice>/     audio of dropped attempts (audit, not published)
  run.log, run-stats.json
  status.json, health.log, status-history.jsonl   (watch.sh; not published)
```

## Alarms (`watch.sh`)

Non-zero exit and an `ALARM:` line in `health.log` when: no new clip for 20 min;
clips/min below 50 % of the first-10-minute average; stress disagreements above
2 % of checked words; mean WER above 5 %; disk free below 10 GB; GPU above 85 °C.
The loop exits on the first alarm (`--keep-going` to continue). It never writes
to the manifest or the audio.

## Troubleshooting

- **`CUDA_VISIBLE_DEVICES` / two GPUs**: the scripts use `cuda:0`. Set
  `VOCA_TTS_DEVICE`/`VOCA_ASR_DEVICE` before running if you need another index.
- **OOM on long sentences**: `--batch-size 4` (or 16 on a 40 GB+ card). Length bucketing is on by
  default; the longest bucket still sets the cost.
- **HF download throttled**: `hf download` resumes; rerun `bootstrap.sh`. If the
  host has `hf_transfer`, `HF_HUB_ENABLE_HF_TRANSFER=1` speeds it up. If a model
  repo asks you to accept terms, export `HF_TOKEN` before bootstrap too.
- **LoRA cannot load**: generation aborts (`--require-lora`). This is deliberate:
  Base without the adapter garbles stressed speech.
- **A corrupt model file**: `fetch_assets.py` prints `MISMATCH`; delete that one
  file and rerun bootstrap. Every file is SHA-256 pinned in `assets.lock`.
- **`HF_TOKEN` missing at publish**: verification runs first, then the upload
  refuses. Nothing is created on HF.

## What was and was not verified

Verified locally (no GPU): manifest/FLAC verification, health thresholds on fake
stats, asset-lock format and SHA-256 of the pinned files, the bundle file list,
shell syntax and `--dry-run`/`--help` paths.

Only verifiable on the server: CUDA availability, compute capability, VRAM
headroom at the chosen batch size, real throughput, GPU temperature, and the first real smoke
clip. Qwen3-TTS is CUDA-oriented upstream; MPS numbers from
`reports/G1-teacher.md` suggest 2–3× realtime at batch 8 on this workload, so
the 4–6 h estimate on a 4090 is an extrapolation, not a measurement.
