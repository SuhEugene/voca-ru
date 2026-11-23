# PIPE — staged generation/screening pipeline

**Owner:** PIPE · **Branch:** `task/pipe` · **Date:** 2026-11-24
**Scope:** make `build_corpus.py` run generation and screening concurrently so the
rented GPU never idles behind Whisper/meter2. Screening logic, thresholds, teacher,
LoRA and output schema are untouched; the sequential path stays the default.

## TL;DR

- `scripts/gen/pipeline.py` adds three stages over bounded queues: **A** generation
  (GPU, 1 thread) → **B** screening (**3** threads by default, Whisper on
  `--asr-device`, meter2 on `--stress-device cpu`) → **C** one writer thread.
  Wired into `build_corpus.py` behind `--pipeline`; the old loop is unchanged.
- **Retries** go back to Stage A through a retry queue as a **single-clip batch**
  (`seed_for([text_id], voice, 1, seed)`), exactly one retry, then `review.jsonl`.
- **Kill-safe resume:** the manifest is still the source of truth; on start a torn
  final line is truncated and duplicate ids collapsed, so a kill cannot lose or
  duplicate a row.
- **Fake benchmark proves the overlap without a model**: at pod-like per-clip
  costs, sequential **312.3 clips/min with the GPU idle 40.6 %** of wall; pipelined
  **486.7 clips/min with the GPU idle 7.4 %** (1.56×), 3 workers. At the faster
  cost model: 476.1 → 800.0 clips/min, idle 46.0 % → 9.1 % (1.68×).
- **19 tests pass** (9 old + 10 new): ordering, exactly-once writer delivery,
  retry→review, retry→accept seed, kill-and-resume (torn manifest), duplicate
  collapse, queue backpressure, no duplicate ids, benchmark assertion.
- **Not verified here:** real CUDA throughput, VRAM at batch 8/12/16, and the
  thread-safety of sharing one meter2/CT2 model across workers. §7 lists exactly
  what to measure on the pod.

## 1. Design

```
 sentences ──► [Stage A: teacher ×1]  ──raw wav on disk──►  [Stage B: screen ×3]
                    ▲                                              │ ok ─────┐
                    │ retry queue (attempt 2, single clip)         │        │
                    └────────────── fail at attempt 1 ◄────────────┘        │
                                                                           ▼
                                                     [Stage C: writer ×1] ──► manifest.jsonl
                                                                           └─► review.jsonl
```

| Stage | Threads | Work | Why |
|---|---|---|---|
| A | 1 | pull length-bucketed batch (or one retry clip), `generate`, write raw 24 kHz wav to `<out>/.work/`, push `(cid, row, voice, seed, attempt, wav_path)` | one model, one GPU: a second generator would only fight for VRAM |
| B | `--screen-workers` (default 3) | read raw wav → `build_corpus.handle_clip` (trim, peak-normalise, save rejected FLAC, `screen.screen_clip`: Whisper WER, duration/rate/level, meter2 stress) | the unchanged screening code; Whisper on `--asr-device`, meter2 on `--stress-device cpu` so it does not take VRAM |
| C | 1 | move accepted FLAC to `audio/`, append one manifest or review row, flush | the only writer: no interleaved appends |

Details that matter:

- **Length bucketing is preserved.** Stage A consumes the same
  `build_corpus.make_batches` lists in the same order, and the batch seed is still
  `seed_for([ids of the whole batch], voice, 0, seed)` — identical to the
  sequential path, including on resume.
- **Backpressure is bounded.** The generate→screen queue has `--queue-size`
  (default 32) slots and Stage A blocks on `put` when it is full, so RAM does not
  grow with corpus size. Raw wavs live at most one queue-depth ahead and are
  deleted by Stage B.
- **Retry semantics.** A failed attempt 1 does not touch the writer; it is pushed
  to the retry queue and regenerated as a one-clip batch with the sequential retry
  seed. Only attempt 2 reaching a verdict produces a row, so every clip gets
  exactly one terminal row (manifest or review).
- **Failure handling.** A stage exception records the error and sets a stop event;
  queues are drained by timeout and `run_pipelined` re-raises after joining, so a
  dead worker cannot hang a 19 h run. A watchdog marks the run failed if clips are
  in flight and nothing has been screened for 600 s.
- **`--limit`, `--voices`, `--split`, `--pilot`, `--measure-batch`, `--max-wer`,
  `--require-lora`** work unchanged: selection, text frontend and `run-stats.json`
  are still done by `build_corpus.main`; the pipeline only replaces the loop.
  `run-stats.json` gains `pipeline: true`, `gpu_busy_s`, `gpu_idle_pct`,
  `max_generate_screen_queue`, `screen_workers`, `asr_device`, `stress_device`.

### Resume proof (exactly-once)

1. Stage C writes `accept()`/`review_row()` with one `append_jsonl` per clip
   (open-append-write-flush). A kill can only tear the **last** line.
2. On start `pipeline.repair_outputs` truncates that torn line back to the last
   newline, then rewrites either file only if it contained duplicate ids (a clip
   id in both files keeps the manifest row, which is what training reads).
3. `done` from those repaired files filters the batches; everything else is
   regenerated with the same seed. Generated wavs are scratch: a kill before the
   row is written leaves at most orphan FLACs that the rerun overwrites, never a
   manifest row without audio.
4. The `.work/` directory is removed at start and at a clean end.

This is tested by truncating `manifest.jsonl` mid-line and by appending a
duplicate row (`tests/gen/test_pipeline.py`).

## 2. Fake benchmark — measured numbers

`python scripts/gen/pipeline.py --benchmark-fake` installs a stub teacher
(`sleep(gen_s)` per clip, synthetic tone) and a stub screener
(`sleep(whisper_s)` under a shared **GPU lock**, then `sleep(meter_s)` as CPU
meter2), and runs the same 96 clips through
(a) the sequential loop shape and (b) the real `run_pipelined`, batch 8.
The lock matters: TTS and Whisper kernels serialise on one card, so a fake that
let the two sleeps overlap would flatter the pipeline.

**Run 1 — pod-like ratios (gen 0.09 s, Whisper 0.02 s, meter2 0.07 s per clip):**

| workers | sequential | pipelined | speedup | GPU idle seq | GPU idle pipe | Stage A idle |
|---:|---|---:|---:|---:|---:|---:|
| 2 | 312.3 clips/min | 427.0 | 1.36× | 40.6 % | 19.1 % | 34.8 % |
| **3** | 312.3 | **486.7** | **1.56×** | 40.6 % | **7.4 %** | 24.6 % |
| 4 | 312.3 | 514.4 | 1.65× | 40.9 % | 2.3 % | 18.8 % |

Upper bound at these costs is `1 / (gen + whisper) = 545 clips/min` (GPU work per
clip), so 3 workers already reach 89 % of it. Two workers starve the GPU: meter2
keeps them busy while the teacher has nothing queued.

**Run 2 — default model in the CLI (gen 0.05, Whisper 0.015, meter2 0.05), 3 workers:**

| | clips/min | GPU idle |
|---|---:|---:|
| sequential | 476.1 | 46.0 % |
| pipelined | **800.0** | **9.1 %** |

The benchmark asserts `GPU idle < 25 %` and `speedup >= 1.3×`; both runs pass.
It also asserts the clip count is identical in both modes. Quoted values are
one run each; run-to-run variation on this laptop is about ±2 %.

**What the fake benchmark does and does not prove.** It proves the stages overlap,
the queue is bounded and no clip is lost, under a model where screening costs are
sleeps. It does *not* model cuDNN/CTranslate2 contention, meter2 CPU scaling with
threads, or disk/FLAC cost — those numbers must come from the pod (§7).

## 3. VRAM and batch size for a 24 GB card

Measured on MPS (D2/G1, bf16 eager, the same teacher): batch 8 with the short
buckets peaked at **9.0 GiB**, and the pilot's **long prose bucket reached
13.6 GiB** at batch 8. The pipeline adds Whisper resident on the GPU:
`large-v3-turbo` CTranslate2 float16 is ~1.6 GB of weights plus a small workspace.

Recommendation for 24 GB:

- **Default `--batch-size 8`.** It is the value D2 verified and leaves >8 GiB of
  headroom for the longest bucket plus Whisper.
- Try **12** on the pod with `--measure-batch 8,12,16`; adopt it only if
  `peak_device_gib` for the long bucket stays below ~19 GiB (4–5 GiB headroom for
  fragmentation and Whisper). 16 is the first choice for ≥40 GB cards, not for 24.
- `--measure-batch` uses the shortest texts; the real peak comes from the **last**
  (longest) bucket of the run, so watch the run's reported `peak_device_gib` and
  `nvidia-smi` for the first ten minutes rather than trusting the short-text probe.
- Batch size and throughput interact with the pipeline only through Stage A; a
  bigger batch shortens the GPU's per-clip cost but makes each queue item larger,
  which is why `--queue-size` defaults to 32 (roughly 4 batches in flight).

## 4. Exact command for the full corpus (16 000 clips)

On the pod, from the bundle root (the same wrapper the S1 runbook uses):

```sh
bash remote/generate.sh --split all --batch-size 8          # unchanged path
# staged path — add --pipeline to the python command generate.sh builds:
PYTHONPATH=training .venv-remote/bin/python scripts/gen/build_corpus.py \
    --out "$PWD/data/corpus-ru" \
    --sources batch1.jsonl batch2.jsonl \
    --voices f_neutral,m_neutral --split all \
    --batch-size 8 --pipeline --screen-workers 3 --queue-size 32 \
    --asr-device cuda --asr-compute-type float16 --stress-device cpu \
    --require-lora
```

`remote/generate.sh` is owned by S1 and was **not** modified; it does not forward
`--pipeline`. Either add `--pipeline` to its `CMD` array, or run the command above
under `tmux` with `2>&1 | tee -a data/corpus-ru/run.log` (what `generate.sh` does).

Smoke it first (3 min):

```sh
PYTHONPATH=training .venv-remote/bin/python scripts/gen/build_corpus.py \
    --out "$PWD/data/pipe-smoke" --sources batch1.jsonl --voices f_neutral --split all \
    --limit 8 --pipeline --batch-size 8 --asr-device cuda --require-lora
```

## 5. Tests

```sh
. scripts/cache-env.sh
PYTHONPATH=training .cache/venvs/r4/bin/python -m unittest discover -s tests/gen
# 19 tests, OK (~3.6 s)
```

`tests/gen/test_pipeline.py` (model-free, stub teacher/screener):

| test | asserts |
|---|---|
| `test_exactly_one_row_per_clip_and_unique_ids` | 10 clips in, 10 rows out, unique ids, `.work/` removed |
| `test_manifest_order_follows_length_buckets_with_one_worker` | generation order = bucket order; batch sizes 4,4,1 |
| `test_retry_then_review` | one review row, `attempt: 2`, two generation calls for that clip |
| `test_retry_then_accept_uses_attempt_2_seed` | manifest row attempt 2 with the sequential retry seed |
| `test_resume_after_torn_manifest_regenerates_only_missing` | torn line truncated, only the missing clip regenerated, valid JSONL |
| `test_repair_collapses_duplicate_rows` | duplicate id removed, done set correct |
| `test_resume_with_everything_done_generates_nothing` | no model call |
| `test_queue_backpressure_is_bounded` | queue depth ≤ `--queue-size`; screening starts before generation ends |
| `test_benchmark_proves_overlap` | benchmark runs and its assertions hold |
| `test_teacher_error_fails_fast_instead_of_hanging` | a dead stage raises instead of hanging a 19 h run |

## 6. Files

| path | change |
|---|---|
| `scripts/gen/pipeline.py` | new: stages, retry queue, writer, resume repair, fake benchmark |
| `scripts/gen/build_corpus.py` | `--pipeline`, `--screen-workers`, `--queue-size`, `--stress-device`, `--benchmark-fake`; pipeline branch; extra `run-stats` keys. Sequential loop unchanged (re-indented, logic identical) |
| `scripts/gen/README.md` | staged-pipeline section |
| `tests/gen/test_pipeline.py` | new tests |
| `reports/PIPE-pipeline.md` | this report |

## 7. What must be re-measured on the pod (and could not be here)

1. **Real speedup.** Run the smoke, then time ~500 clips sequential and with
   `--pipeline` on the same sentences: compare `clips_per_min`, `generation_wall_s`,
   `gpu_busy_s`, `gpu_idle_pct` in `run-stats.json`. The task measured
   **13.7 clips/min** sequential at batch 8; the pipeline can only remove the
   screening time from the critical path, so the expected ceiling is
   `1 / (generation + whisper)` per clip. If your generation is ~0.9 s/clip and
   Whisper ~0.2 s/clip, that is ~54 clips/min (≈4×); if meter2 is slower than the
   GPU stage on 3 workers, add workers until the GPU is the bottleneck. The fake
   model says 1.5–1.7× at pod-like ratios, but its screening split is a guess.
2. **VRAM** at batch 8/12/16 with the long bucket and Whisper resident
   (`peak_device_gib`, `nvidia-smi`). Adopt 12 only under ~19 GiB.
3. **Thread safety of the shared judges.** One `AsrJudge` (CTranslate2) and one
   `meter2` model are used from 3 threads. Both are inference-only forwards and
   CTranslate2 documents concurrent calls, but meter2 was only ever run with one
   clip at a time in D2. If it misbehaves, the fallback is `--screen-workers 1`
   (correct, still overlaps Stage A with meter2/IO but not with Whisper) — record
   what you see.
4. **Whether Whisper on `cuda` steals enough GPU to hurt TTS.** Compare
   `--asr-device cuda` with `--asr-device cpu` on a short run; the pipeline
   tolerates both. If CUDA memory is tight at batch 12+, Whisper on CPU is the
   cheap escape hatch.
5. **`--pipeline` with the 10 % stress-check workload is unchanged**; thresholds
   live in `screen.py` and were not touched.
6. Sequential vs pipeline output equivalence for a fixed seed is by construction
   (same batches, same seeds, same `handle_clip`), but only the single-item retry
   path differs from sequential by design: a retried clip is regenerated alone
   with `seed_for([id], voice, 1, seed)` instead of grouped with its batch mates.
   On the pilot no clip failed, so this changes no measured number; a rerun that
   hits a retry will produce the attempt-2 seed above.

## 8. Open issues / risks

1. **Retry divergence** from sequential when ≥2 clips of one batch fail (one
   single clip instead of a group) — intended by the task brief, documented.
2. **`--no-screen` semantics** in the pipeline path: it shapes and keeps audio
   without calling Whisper/meter2 (the sequential path raised inside
   `screen_clip` because `asr` was `None`). Debug-only flag, no contract impact.
3. **Writer throughput** is one thread doing `shutil.move` + one append; at
   ~1 clip/s the writer uses a few percent of a core, so it is not a bottleneck,
   but it is measured only in the fake benchmark.
4. **`remote/generate.sh` not updated** (S1 owns it): full runs need `--pipeline`
   added there or the explicit command in §4.
5. **`--benchmark-fake` is a model, not a measurement.** Only the pod run in §7
   can settle the real speedup.
