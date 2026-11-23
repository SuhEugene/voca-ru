# SPLIT — generation and screening as separate passes

**Owner:** SPLIT · **Branch:** `task/split` · **Date:** 2026-11-24
**Scope:** run generation and screening as two independent passes, so the batch
size is chosen by VRAM alone and the GPU never idles behind Whisper/meter2.
Screening thresholds, teacher, LoRA, seeds and the manifest schema are unchanged.

## TL;DR

- New sibling script **`scripts/gen/split_corpus.py`** with three exclusive modes:
  `--generate-only` (teacher only), `--screen-only` (Whisper + meter2 only),
  `--wave-report` (reject histogram + id list for the next wave).
- Generation writes `--out/audio/<voice>/<id>.flac` + one `raw.jsonl` row per
  clip and wave; it prints `peak_device_gib` after **every** batch. It never
  imports or builds an ASR/stress judge.
- Screening reads `raw.jsonl`, runs the existing `screen.screen_clip` with the
  same stats, and appends to the unchanged `manifest.jsonl` / `review.jsonl`
  schema. It never loads the teacher. Re-running it screens nothing twice.
- Wave 1 is **byte-identical** to the sequential screened run for the same
  sentence set and `--batch-size`: same `make_batches`, same
  `seed_for([batch ids], voice, 0, seed)`, same `prepare_audio`/FLAC writer, same
  `screen_clip`. (The seed covers the whole batch, so a different `--batch-size`
  changes every seed — pick the batch size before the wave-1 run, do not resume
  with another one.)
- Wave 2+ regenerates exactly the ids in `--ids-file`, one clip per teacher call,
  with the sequential single-clip retry seed
  `seed_for([text_id], voice, wave - 1, seed)` and `attempt = wave`.
- **8 new tests** (`tests/gen/test_split.py`, model-free): generate-only never
  screens, resume skips known ids, wave file selects exactly the rejected ids,
  empty wave, screen never rescans, reason histogram, second wave is re-screened,
  per-batch peak print. All 17 tests in `tests/gen` pass.
- **Not verified here:** real CUDA peak VRAM (this laptop is MPS), real
  Whisper/meter2 throughput, and the max batch for a 24 GB 4090 — §5 gives the
  probe procedure and the number to watch.

## 1. What changed

| path | change |
|---|---|
| `scripts/gen/split_corpus.py` | new: `--generate-only`, `--screen-only`, `--wave-report` |
| `scripts/gen/build_corpus.py` | `accept()` split into `manifest_row()` + move, so screening can reuse the exact manifest schema without moving a file. Sequential behaviour unchanged |
| `tests/gen/test_split.py` | new: 8 model-free tests |
| `reports/SPLIT-two-phase.md` | this report |

`--pipeline` does not exist on this branch (it lives on `task/pipe`) and was not
touched. The sequential path is untouched apart from the mechanical
`accept` → `manifest_row` extraction; `tests/gen/test_corpus.py` still passes.

## 2. How it works

```
--generate-only:  sentences -> make_batches -> teacher -> trim/normalise -> audio/ + raw.jsonl
--screen-only:    raw.jsonl -> screen_clip -> manifest.jsonl | review.jsonl (+ rejected/)
--wave-report:    review.jsonl -> histogram + wave<N+1>.ids
```

### raw.jsonl

One row per (clip, wave), appended. The required keys are:

```json
{"id":"a1_f_neutral","voice":"f_neutral","text":"…","stressed_text":"…",
 "phonemes":["<bos>",…],"tokens":[1,…],"audio":"audio/f_neutral/a1_f_neutral.flac",
 "seconds":1.84,"seed":1234567890,"attempt":1,"wave":1}
```

It also carries `text_id`, `norm_text`, `category`, `split`, `origin`,
`source_license`, `source_url`, the pre-shaping stats (`raw_peak`, `raw_rms`,
`raw_clipped`, `lead_trim_s`, `tail_trim_s`, `gain`, `peak`, `rms`),
`synth_seconds` and `teacher_adapter`. Screening needs those to rebuild exactly
the row the sequential path would have written: provenance for the manifest,
raw stats for the clipping check, adapter for `teacher_adapter`.

### Resume rules

- **Generation**: a clip id already in `raw.jsonl` is skipped in wave 1. In wave
  N, an id whose latest raw row has `wave >= N` is skipped; anything older is
  regenerated (that is the point of a wave).
- **Screening**: pending = latest raw row per id, unless the id is in
  `manifest.jsonl` (accepted is terminal) or its `review.jsonl` row has
  `attempt >= raw.wave`. That makes a second `--screen-only` run screen 0 clips.
  When an id *is* re-screened at a newer wave, its stale review row is removed
  before the new verdict is written, so one clip never has two terminal rows and
  an id never appears in both files. A kill mid-screen leaves the old review row
  in place until the next run drops and re-screens it.
- `screen-stats.json` is rewritten each screening run.

## 3. Commands

All from the bundle/worktree root, `. scripts/cache-env.sh`, with
`PYTHONPATH=training`. On the pod the interpreter is `.venv-remote/bin/python`.

### (a) generation, largest batch that fits

Probe the peak on a throw-away output first (long-prose tail included):

```sh
for B in 8 12 16 20 24; do
  PYTHONPATH=training .venv-remote/bin/python scripts/gen/split_corpus.py \
      --generate-only --out "data/probe-b$B" --sources batch1.jsonl \
      --voices f_neutral --split all --limit 64 --batch-size "$B" 2>&1 \
      | grep -E '^\[gen\]|^\[generate\]'
done
```

Pick the largest `B` whose **last** (longest) batch printed
`peak_device_gib` below your headroom target, then run the real wave:

```sh
PYTHONPATH=training .venv-remote/bin/python scripts/gen/split_corpus.py \
    --generate-only --out "$PWD/data/corpus-ru" \
    --sources batch1.jsonl batch2.jsonl --voices f_neutral,m_neutral --split all \
    --batch-size 16 --seed 1234 --require-lora \
    2>&1 | tee -a data/corpus-ru/run.log
```

Every batch prints its peak, so the maximum over the run is visible while it
runs. If a batch OOMs the run stops (nothing is half-written); fix the batch size
and empty the output dir — do **not** continue the same corpus with a different
`--batch-size` (different buckets → different seeds → different audio).

### (b) screening (separate run, no teacher)

```sh
PYTHONPATH=training .venv-remote/bin/python scripts/gen/split_corpus.py \
    --screen-only --out "$PWD/data/corpus-ru" \
    --voices f_neutral,m_neutral \
    --asr-device cuda --asr-compute-type float16 --stress-device cpu \
    2>&1 | tee -a data/corpus-ru/screen.log
```

Prints and stores in `screen-stats.json`: `screened`, `accepted`, `rejected`,
`reject_reasons` (clips by first reason, numbers bucketed),
`wer_mean`, `wer_p95`, `stress_fail_pct`, `clips_per_min`.

### (c) wave loop

```sh
# 3a. which clips failed, and why; writes data/corpus-ru/wave2.ids
PYTHONPATH=training .venv-remote/bin/python scripts/gen/split_corpus.py \
    --wave-report --out "$PWD/data/corpus-ru" --ids-file data/corpus-ru/wave2.ids

# 3b. regenerate exactly those ids (single-clip calls, retry seed)
PYTHONPATH=training .venv-remote/bin/python scripts/gen/split_corpus.py \
    --generate-only --out "$PWD/data/corpus-ru" --wave 2 \
    --ids-file data/corpus-ru/wave2.ids \
    --sources batch1.jsonl batch2.jsonl --voices f_neutral,m_neutral --split all \
    --require-lora

# 3c. screen the new wave only (accepted ids are never re-screened)
PYTHONPATH=training .venv-remote/bin/python scripts/gen/split_corpus.py \
    --screen-only --out "$PWD/data/corpus-ru" --asr-device cuda --stress-device cpu

# repeat with --wave 3, 4, ... while wave-report still finds rejects
```

`--wave-report` prints the histogram (e.g. `{"wer N > N": 12, "stress N/N": 4}`)
and writes all rejected ids sorted; with no rejects it prints that and writes an
empty file, and a wave run over an empty list does nothing.

## 4. Why the batch can be bigger now

Screening used to hold Whisper (large-v3-turbo CTranslate2 float16, ~1.6 GB plus
workspace) on the same card as the teacher. The two passes never coexist now, so
generation has the card to itself and `--batch-size` is bounded only by the
teacher's activation peak. `screen-only` puts Whisper and meter2 on the device
you name (`--asr-device`, meter2 always CPU), independent of generation.

## 5. 24 GB 4090: batch sizes to try and what to watch

Measured on the M5 (D2/G1, bf16 eager, same teacher): batch 8 peaked at
**9.0 GiB** on short text and **13.6 GiB** on the long-prose bucket. CUDA eager
attention has different allocator behaviour, so treat those as a starting point,
not a prediction.

- Try **8, 12, 16, 20, 24** with the §3(a) probe, in that order. Start the real
  run at **16**; raise only if the probe's longest bucket stays under budget.
- **The number to watch is `peak_device_gib` printed by the last (longest)
  bucket**, plus `nvidia-smi --query-gpu=memory.used --format=csv -l 5` during
  the first ten minutes. Long prose is the worst case; the short-text buckets
  under-report it by ~4.5 GiB on the M5 measurement.
- Budget: keep the peak at or below **~20 GiB** so ~4 GiB remains for CUDA
  context, fragmentation and a bigger-than-expected sentence. A 24 GB card
  reporting 23.5 GiB used is already at risk of a late OOM on one long batch.
- If batch 20 fits and 24 does not, 20 is worth it only if `clips_per_min` from
  the `[generate]` line actually improved; long batches also raise latency to
  the first raw row.

## 6. Equivalence with the sequential run

- **Same text frontend**: `build_corpus.load_rows`/`select_pilot`/`prepare_texts`.
- **Same batching and seed**: `make_batches(rows, batch_size)` and
  `seed_for([ids of the whole batch], voice, 0, seed)`; a resumed batch keeps its
  full-batch seed, so a partial wave-1 resume also matches.
- **Same shaping and encoder**: `screen.prepare_audio` then `sf.write(PCM_16)`,
  exactly what `handle_clip`+`accept` write.
- **Same screening code and thresholds**: `screen.screen_clip` with the stats
  recorded at generation.
- **Same manifest schema**: built by `build_corpus.manifest_row`, the extracted
  body of the old `accept()`. `synth_seconds` differs (wall/clips of the batch,
  not of a 1-clip retry) but it is provenance, not audio.
- **Deliberate difference**: wave 2+ regenerates a clip alone with the
  single-clip retry seed, as the sequential and PIPE paths already document;
  the sequential path could group several simultaneous failures into one retry
  batch, so those clips can differ from a sequential retry. Wave 1 has no such
  difference.

## 7. Tests

```sh
. scripts/cache-env.sh
PYTHONPATH=training .cache/venvs/r4/bin/python -m unittest discover -s tests/gen
# 17 tests, OK
```

`tests/gen/test_split.py`, all with a stub teacher / stub ASR (no models):

| test | asserts |
|---|---|
| `test_generate_only_never_screens` | `screen.screen_clip` patched to raise; raw rows, audio and required keys exist; no manifest |
| `test_resume_skips_known_ids` | second generation makes no teacher call |
| `test_peak_is_printed_per_batch` | every batch line carries `peak_device_gib` |
| `test_wave_file_selects_exactly_rejected_ids` | review ids == ids file; wave 2 calls exactly those, one clip each, retry seed |
| `test_empty_wave` | empty ids file → 0 rejects, 0 generation calls |
| `test_screen_never_rescans` | second screen run screens 0 and calls no ASR |
| `test_reason_histogram_and_accepted_schema` | `{"wer N > N": 2}`; manifest row has adapter/text_id; rejected audio moved |
| `test_second_wave_is_rescreened` | wave-2 raw row is screened again; stale review row removed; accepted once |

## 8. Open issues / what could not be verified

1. **No real CUDA run here.** Peak VRAM at each batch size, throughput per pass
   and the 4090's true max batch must come from §3(a)/§5 on the pod.
2. **Sequence-of-waves policy.** Wave 2 fixes only clips that failed screening;
   the accepted set is frozen at wave 1. If a later human listen rejects an
   accepted clip, delete its manifest row and its id will be regenerated as a
   wave (its `raw.jsonl` wave must be older than the new wave).
3. **`--screen-batch` is a progress/IO chunk, not model batching.** Whisper's
   `transcribe` and meter2's `judge` are per-clip APIs; the chunk only bounds the
   loop and prints progress. If a batched ASR backend appears, screen that chunk
   in one call.
4. **Mixed corpora.** If the same `--out` already holds sequential rows, they are
   accepted/rejected terminal rows and are not re-screened; their review rows
   carry `attempt` 1/2 and are treated as screened at that wave. Do not point
   `--screen-only` at a corpus whose review rows predate its raw rows without
   checking `screen-stats.json`.
5. **`remote/generate.sh` is S1's** and does not forward these flags; use the
   explicit commands in §3 (same pattern as PIPE's report).
6. **Disk**: generation keeps every wave's audio under `audio/`; a rejected clip
   is moved to `rejected/` by screening, so a rejected id keeps one file, not
   two. Old wave audio for a regenerated id is overwritten in place.
