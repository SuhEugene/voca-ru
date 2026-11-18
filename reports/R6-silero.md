# R6 — Silero for Russian voca

**Owner:** R6 · **Branch:** `task/silero` (rebased on `origin/main`) ·
**Date:** 2026-11-19
**Deliverables:** [`docs/silero.md`](../docs/silero.md),
samples in `out/silero/` (git-ignored, 72 WAVs + 3 JSON).

## TL;DR

- **Silero TTS is a good second teacher, not a replacement for Qwen3-TTS.**
  It is native-sounding Russian (vendor UTMOS 3.04 vs 3.08 for real speech),
  fixed 5-speaker, 24 kHz, and 69–322× realtime on this M5 CPU — so a 19 h
  corpus is minutes, not GPU-hours. But it is **CC BY-NC-SA 4.0**, has no
  training/fine-tuning/cloning code, and **deletes digits** rather than
  reading them.
- **Only two Russian TTS models are MIT:** `v5_cis_base` and
  `v5_cis_base_nostress` (60 speakers, 29 `ru_*`), but they require manual
  stress and those speakers are cross-lingual (accent risk).
- **No Russian TTS model ships ONNX.** All `.onnx` probes 404; the artefacts
  are TorchScript packages whose `forward()` takes Python string lists. A
  Rust runtime fallback is a porting project, not a drop-in.
- **`silero-stress` is MIT but is a 31 M-parameter neural package, not a
  dictionary.** `ruaccent` (3.19 M forms, static table, 21 MB) remains the
  better base for the Python trainer + Rust runtime.
- **Silero STT has no Russian entry in the current catalogue** (en/de/es/ua
  only), so it cannot be our judge. Keep Whisper large-v3-turbo + GigaAM.
- **Silero VAD (MIT, 1.2 MB ONNX)** is directly useful for trimming clips.

Recommendation: keep Qwen primary; add `v5_6_ru` `baya` (F) + `aidar` (M) as
a second teacher / immediate CPU data source; use `ruaccent` for stress and
Silero VAD for trimming; do not plan Silero as the Rust fallback.

## Key numbers

| | `v4_ru` | `v5_6_ru` | `v5_cis_base_nostress` |
|---|---:|---:|---:|
| params | 8.89 M | 20.90 M + 31.4 M stress | 20.93 M |
| file | 40.1 MB | 147.7 MB | 91.7 MB |
| licence | CC BY-NC-SA 4.0 | CC BY-NC-SA 4.0 | **MIT** |
| load | 0.17 s | 1.00 s | ~1 s |
| first call | 0.45 s | 0.21 s | — |
| warm median | **15.7 ms** (default) / 24 ms (1 thr) | **75.7 ms** (default) / **38.2 ms (1 thr)** | 41.5 ms (1 thr) |
| ×realtime | 322× / 211× | 69× / **136×** | 141× |
| peak RSS | **648 MB** | **826 MB** | ~0.7 GB |
| MPS | works (9 ms warm) | **fails** (`Passed CPU tensor to MPS op`) | not tested |

Digit handling (verified through the model's own preprocessor):
`15:30 → ":"`, `2,5 → ","`, `2026 → ""`, `123 → ""`. All five voices failed
all three digit-bearing sentences; spelled-out numbers (`пятьсот`, `пять
тысяч`) were read correctly. This is a frontend finding, not a voice defect.

## What I did

1. Read `docs/plan-ru-qwen.md`, `docs/repo-map.md`, `docs/datasets-ru.md`,
   `AGENTS.md`; rebased `task/silero` on `origin/main`.
2. Read the actual Silero `LICENSE`, `LICENSE_CIS`, README licence sections
   and `models.yml`; measured model sizes with HEAD requests and by loading
   the archives.
3. Built a Python 3.12 env at `.cache/venvs/r6`, installed `silero`,
   `omegaconf`, `soundfile`, `faster-whisper`; downloaded `v3_1_ru`,
   `v4_ru`, `v5_6_ru`, `v5_cis_base_nostress`.
4. Synthesised the six required sentences with **five `v5_6_ru` voices**
   (baya, kseniya, xenia, aidar, eugene), **three `v4_ru` voices**, and
   **two MIT voices** (ru_oksana, ru_marat), plus SSML/question/emphasis and
   stressed vs automatic homograph variants. 72 WAVs at 24 kHz.
5. Ran a Whisper `small` round trip over 71 clips; checked numbers/dates;
   measured F0, peak/RMS, silence and HF energy.
6. Measured load/first/warm timings at 1/4/default threads, peak RSS, and
   MPS; verified offline operation with dead proxies.
7. Tested `silero-stress` on the homograph sentence and inspected its
   components (2.06 M accentor + 19 660 exceptions; 29.19 M homograph BERT,
   2 208 homographs, 1 477 phrase rules).
8. Wrote `docs/silero.md` with the quoted licences, catalogue, evidence,
   measurements, verdict table and recommendation.

## Reproduce

```sh
cd /Users/suheugene/projects/22-tiniest-voca-ru
. scripts/cache-env.sh
uv venv --python 3.12 .cache/venvs/r6
uv pip install --python .cache/venvs/r6/bin/python silero omegaconf soundfile faster-whisper

# models already at .cache/models/silero/*.pt; loading is offline
.cache/venvs/r6/bin/python - <<'PY'
import torch, soundfile as sf
torch.set_grad_enabled(False); torch.set_num_threads(1)
m = torch.package.PackageImporter(
    ".cache/models/silero/v5_6_ru.pt").load_pickle("tts_models", "model")
a = m.apply_tts(text="Сегодня двадцать третье января.", speaker="baya", sample_rate=24000)
sf.write("out/silero/check.wav", a.numpy(), 24000)
PY
```

The full experiment scripts were run from a temporary directory and removed
after the data was captured; the core code is embedded in
`docs/silero.md` §5. Machine-readable outputs that remain:
`out/silero/transcripts-whisper-small.json`,
`out/silero/stress-silero-stress.json`,
`out/silero/measure-threads-v1.json`.

## Final paths

| what | where |
|---|---|
| models | `/Users/suheugene/projects/22-tiniest-voca-ru/.cache/models/silero/` (`v3_1_ru.pt` 61.9 MB, `v4_ru.pt` 40.1 MB, `v5_6_ru.pt` 147.7 MB, `v5_cis_base_nostress.pt` 91.7 MB) |
| venv | `/Users/suheugene/projects/22-tiniest-voca-ru/.cache/venvs/r6/` |
| Silero source clones | `.cache/src/silero-models`, `.cache/src/silero-stress`, `.cache/src/silero-vad` |
| samples | `/Users/suheugene/projects/22-tiniest-voca-ru/out/silero/` (72 WAVs + 3 JSON) |
| doc | `docs/silero.md` |

## Cleanup performed

- Moved the three originally downloaded models from
  `/private/var/folders/.../T/opencode/silero-models-cache` to
  `.cache/models/silero/`; verified synthesis from the new path before
  deleting the temporary copy.
- Moved the `silero-models`, `silero-stress` and `silero-vad` clones from
  `/private/var/folders/.../T/opencode/` to `.cache/src/`.
- Deleted my temporary venv (`.../T/opencode/silero-env`), scratch scripts,
  `asr-extras`, and the stray `uvcache-r6` under the temp directory.
- Did not touch `qwen312`, `py315`, or any other agent's worktree/cache.

## Open issues / not verified

- **No human listening.** This agent cannot play audio. Naturalness/accent
  judgements in `docs/silero.md` are labelled as surrogates (ASR, F0,
  spectral). A human must listen to `out/silero/` before approving a voice.
- `ruaccent` runtime was **not benchmarked here**: `pip install ruaccent` +
  `acc.load(turbo3.1)` stalled for 26 min on a single-connection model
  download and was killed (partial files remain only inside the r6 venv).
  Comparisons use R2's measurements and Silero's published head-to-head.
- `v3_1_ru` feature flags (homograph support) are not documented in the v3
  README table; marked “not verified”.
- `v5_ru`/`v5_cis_base` parameter counts are inferred from the identical
  architecture files measured (`v5_6_ru`, `v5_cis_base_nostress`), not loaded
  individually.
- Timings were taken while other agents were running heavy Python downloads;
  absolute values vary ±20 %, ratios were stable across reruns.
- The MIT cross-lingual voices (`ru_oksana`, `ru_marat`) were not evaluated
  for accent by a native listener; do not treat the MIT route as
  quality-equivalent without that check.
