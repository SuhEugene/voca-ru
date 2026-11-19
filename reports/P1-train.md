# P1 — Russian tiny-voca training stack

**Owner:** P1 · **Branch:** `task/p1-train` · **Date:** 2026-11-19
**Goal:** bring the upstream Python training stack into this repo, adapted to
Russian, so a real run can start as soon as synthetic audio exists.

## TL;DR

- The upstream stack is cloned to `/Users/suheugene/projects/22-tiniest-voca-ru/.cache/src/tiny-voca`
  (revision `e7f1bde`, Apache-2.0; attribution in
  `training/voca_ru/train/NOTICE`). No model or data downloads.
- `config.py`, `data.py`, `losses.py` and `journal.py` are **byte-identical** to
  upstream. `model.py` differs only where MPS cannot run upstream code (13 lines).
- The model is exactly the released architecture: **4,537,291 parameters, 184
  tensors**, asserted in `tests/train/test_train.py`. The only change is the
  76-symbol Russian inventory already frozen by F3.
- Full smoke ran end-to-end on this M5: `prepare` → `annotate` (CTC aligner) →
  3 training steps → checkpoint → resume → 4th step → `inference.pt` (184
  tensors, format 4) → one Russian sentence synthesised. See §5.
- Measured on M5 Pro / MPS, padded batch 9000 frames (96 s audio):
  **0.394 s/step** spectral-only, **2.256 s/step** with GAN (0.44 steps/s).
  Projection for 104k steps (8k spectral + 96k GAN): **~61 h**, i.e. 2.5 days.
- Blocked on: synthetic audio (D2), and one Python venv that has both torch and
  `espeakng_loader`/`ruaccent` (today they are split between `.cache/venvs/r4`
  and `.cache/venvs/f3`).

## 1. Layout and what is where

```
training/prepare.py                 entry point → voca_ru.train.prepare
training/annotate.py                entry point → voca_ru.train.annotate
training/train.py                   entry point → voca_ru.train.cli
training/voca_ru/train/
    config.py  model.py  data.py  losses.py  journal.py    (upstream port)
    align.py    (upstream CTC aligner, Russian label mapping)
    prepare.py  (upstream acoustic features, speaker-column default)
    annotate.py (upstream annotation layer, no refine pass)
    cli.py      (upstream train loop + M5/AMP/hooks)
    text.py     (adapter over F3's voca_ru.text; NOT a second frontend)
    NOTICE
configs/base.yaml                   upstream-equivalent recipe
configs/smoke.yaml                  ≤100-step recipe
tests/train/test_train.py           param count, shapes, 3 steps, checkpoint
```

`training/voca_ru/text` was reused as required: normalisation, RUAccent stress,
eSpeak-ru G2P and `symbols_ru.py` are imported, never reimplemented.

## 2. Exact diff from upstream

`diff -u` against `.cache/src/tiny-voca/src/tiny_voca/*.py`:

| File | Change |
|---|---|
| `config.py` | **0 lines.** Same 4.5M config; `vocab_size=76`, speakers `F2/M2`. |
| `data.py` | **0 lines.** |
| `losses.py` | **0 lines.** Multi-resolution STFT/mel, MPD/MRD critics. |
| `journal.py` | **0 lines.** |
| `model.py` | **+11/−4.** Two MPS patches: `pulse_train` uses float32 phase on MPS (MPS has no float64); SDPA dropout is applied to the output because MPS SDPA rejects `dropout_p>0`. |
| `align.py` | **+5/−7.** Label lookup goes through `voca_ru.text.symbols_ru.ALIGNER_LABEL` (F6 table); optional local model directory instead of hub id. |
| `prepare.py` | **+1/−1.** `--speaker-column` default `speaker` (manifest contract) instead of `voice_id`. |
| `annotate.py` | **+10/−41.** Device auto-detect, `--aligner-dir`, Russian `FRICATIVES`, and the optional from-scratch refinement pass removed (it was not part of this deliverable and upstream has it off by default). |
| `train.py → cli.py` | **+124/−15.** `--config` YAML, device `auto`, bfloat16 AMP, gradient accumulation, periodic sample synthesis, `voca_ru.eval` WER hook, `--eval-every`/`--sample-every`, MPS guards. Optimisation core (losses, GAN gradient scaling, LR, teacher schedule) unchanged. |

Everything else (duration/pause loss, pitch-spread loss, adversarial gradient
scaling at ratio 0.7, teacher hold/ramp, bucket sampler, checkpoint format 4) is
verbatim.

`docs/repo-map.md` matches the code on every point I checked (config values,
tensor names/shapes, 184 tensors, 4,537,291 params). No contradiction found.

## 3. Russian adaptations

- **Inventory:** `training/voca_ru/text/symbols_ru.py` (76 ids, `FIRST_PHONE=28`,
  `FIRST_VOWEL=64`). `Config.vocab_size=76`, so the Rust shape contract holds.
- **Front end:** `training/voca_ru/train/text.py` adapts F3:
  `normalize → RUAccent stress → eSpeak-ru phonemize → symbols_ru ids`. Text
  already carrying U+0301 (manifest `norm_text`) is used as given, so teacher
  and student see the same stress.
- **Annotation:** upstream CTC Viterbi with
  `facebook/wav2vec2-lv-60-espeak-cv-ft`, 20 ms emissions, −19 ms onset,
  silence → BOS/EOS/pause token, durations sum to the acoustic frame count
  exactly. Labels come from F6's `ALIGNER_LABEL`, so every training phone has a
  recogniser label. The cached model is at
  `.cache/models/wav2vec2-lv-60-espeak-cv-ft`; pass `--aligner-dir` (the HF hub
  cache does not contain it).
- **No `refine` pass.** Upstream's optional classifier refinement is not
  delivered; `--stage`/`--refine` were removed rather than left dead.
- **Stress is a feature**, not a token — unchanged from upstream.

## 4. Configs and device

`configs/base.yaml` is upstream-equivalent: 104,000 steps, batch 9,000 padded
frames, max batch 32, AdamW 3e-4, warmup 1,000, half-life 100k, GAN from step
8,000, ratio 0.7, ramp 3,000, teacher hold 3,000/ramp 7,000/final 0.
`configs/smoke.yaml`: 100 steps, 1,200 frames, max batch 4, accumulation 2, GAN
from step 20.

`--device auto` (the default) selects MPS when available, else CPU. AMP is
bfloat16 (`--amp/--no-amp`); no GradScaler is needed. Gradient accumulation is
`--accumulate N` (loss divided by N, optimizer stepped on the last micro-batch,
critics updated per micro-batch).

Note: the contract's config naming scheme is `configs/<purpose>-<voice>-<variant>.yaml`;
this task explicitly asked for `configs/base.yaml` and `configs/smoke.yaml`, so
those names are used. Renaming to the contract form is a one-line `git mv`.

## 5. Smoke proof (this M5)

Commands actually run, all under the worktree `.cache` and deleted afterwards:

```sh
export PYTHONPATH=training:<f3 site-packages>          # espeakng_loader + ruaccent
PY=.cache/venvs/r4/bin/python                          # torch 2.15 / MPS

# 2 train + 1 validation + 1 test clip from out/silero (24 kHz), arbitrary texts
$PY training/prepare.py  --inputs .../train.jsonl --output .../features --split train
$PY training/annotate.py --features .../features --output .../annotations \
    --device mps --aligner-dir .cache/models/wav2vec2-lv-60-espeak-cv-ft
$PY training/train.py --config configs/smoke.yaml --train ... --validation ... \
    --output .../run --steps 3 --save-every 1 --sample-every 0 --device mps
$PY training/train.py ... --resume .../run/last.pt --steps 4
```

Observed:

- `prepare`: 4/4 clips accepted; 351/298/… frames at 24 kHz / hop 256.
- `annotate`: 4/4 aligned; timing probe found voicing flips at
  vowel→voiceless boundaries (mean |offset| 0.5–1.7 frames). With the *wrong*
  text, both clips were rejected (`weak_alignment`, `low_confidence`), which is
  the QC behaving as intended; with permissive QC thresholds the same run
  produced accepted manifests.
- `train`: 3 steps, then `--resume` at step 3 → step 4. `inference.pt` has
  `format_version=4`, `step=4`, **184 tensors**, frontend `voca_ru.text`.
- `model.synthesize` on the checkpoint: 1.54 s finite waveform for
  «Мой коллега уходит в отпуск на две недели.».

Tests (`python -m unittest discover -s tests/train`, MPS): **6 passed**,
including parameter count, forward shapes, 3 optimisation steps (third one
adversarial), checkpoint round-trip, YAML loading and `prepare.extract` shapes.

### Step-time benchmark

Random tensors of the real dtypes, batch 30 × 300 frames = 9,000 padded frames
(96 s of audio), bfloat16 autocast, MPS:

| Phase | s/step | steps/s | MPS driver memory |
|---|---:|---:|---:|
| spectral only | 0.394 | 2.54 | 3.47 GB |
| with GAN (2×8192-sample crops) | 2.256 | 0.44 | 9.50 GB |

Projection: `8,000×0.394 + 96,000×2.256 ≈ 219,700 s ≈ 61 h` for the 104k-step
base recipe. This is in the plan's "1–2 days, maybe more" band. CPU was not
benchmarked (expected several times slower; the vocoder is the expensive part).

## 6. Eval and sample hooks

- Every `--sample-every` steps, five fixed Russian sentences (plain, question,
  number, date, name) are phonemised and synthesised to
  `<output>/samples/step-XXXXXXXX/*.wav` with a `manifest.jsonl`.
- Every `--eval-every` steps, those wavs are scored by
  `training/voca_ru/eval` (`evaluate` + `make_backend("whisper")`) and the WER
  is written to metrics and `<output>/eval-XXXXXXXX.json`. Missing ASR weights
  or `faster-whisper` produce one JSON `eval_error` and training continues.
- Metrics are plain `metrics.jsonl`; no wandb. Checkpoints: `last.pt`,
  `inference.pt`, numbered `step-XXXXXXXX.pt` at `--keep-every` and at the GAN
  boundary, atomic writes.

## 7. Blocked on real data / what I could not verify

1. **No synthetic Russian corpus yet (D2).** All data-path checks used four
   Silero clips with *arbitrary* text, so they prove plumbing and QC, not
   alignment quality on the real corpus. Expect different QC threshold tuning.
2. **One venv, two halves.** `.cache/venvs/r4` has torch/librosa/soundfile;
   `.cache/venvs/f3` has `espeakng_loader`/`ruaccent`. I ran with r4 plus F3's
   site-packages on `PYTHONPATH`. The training venv must union them before a
   real run; otherwise `phonemize` fails and sample synthesis/the eval hook
   skip with `sample_error`.
3. **`scripts/cache-env.sh` resolves `ROOT` to the worktree**, so it points
   caches at `<worktree>/.cache`, not the absolute main-repo `.cache` required
   by `AGENTS.md`. I moved the upstream clone to the main cache manually. This
   is outside my owned paths; the script should use `git rev-parse
   --git-common-dir` like the Makefile.
4. **Aligner discovery.** Upstream downloads `facebook/wav2vec2-lv-60-espeak-cv-ft`
   through the HF hub cache; F6 stored it under `.cache/models/…`. `--aligner-dir`
   handles this, but a production command must pass it (or the model must be
   linked into the hub cache).
5. **Unknown phonemes are not tracked.** F3's `g2p.py` drops units not in
   `PHONE_MAP` and does not return them; the annotate QC's
   `unknown_phoneme` gate therefore cannot fire. Worth adding a dropped-unit
   return to `voca_ru.text.phonemize` before trusting the reject count.
6. **Eval hook unexercised.** `faster-whisper` is not installed in r4, so the
   WER call was not executed; only the fallback path was reasoned about. The
   smoke config sets `eval_every: 0`.
7. **MPS numerical details.** float32 phase in `pulse_train` is a deliberate
   fallback; on very long utterances phase drift is possible (3000 frames was
   not tested at full length). CUDA keeps float64 and is unaffected.
8. **CPU training** not measured.
9. **Config naming** deviates from `docs/contracts.md` §7 (see §4).

## 8. Open questions

- Should `prepare` enforce the contract's 0.5–35 s admission window in addition
  to `MIN_FRAMES=32`/`max_frames=3000`? It currently accepts 0.34 s at the low
  end (32 frames).
- Should sample synthesis use voice F2 only, or both voices alternating?
  Currently F2 (`speaker=0`) for all five.
- Should the eval hook score a frozen held-out text list instead of the five
  sample sentences? The latter measures the current model on training-adjacent
  text, which is a monitor, not a gate.
- Is 61 h on M5 acceptable, or should the base recipe start on rented CUDA?
  The training code is device-agnostic; only `--device cuda` is needed.

## 9. Reproduce the tests

```sh
cd worktrees/p1-train
PYTHONPATH=training /Users/suheugene/projects/22-tiniest-voca-ru/.cache/venvs/r4/bin/python \
  -m unittest discover -s tests/train -v
```

Expected: `Ran 6 tests ... OK`, including `4_537_291` parameters.
