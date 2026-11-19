# F5 — evaluation harness

Status: **done** (2026-11-19). Real large-v3-turbo numbers measured on this M5;
stress error detection remains explicitly untested (no `meter2`).

## What this is

One instrument for every round-trip number in the project: WER/CER, per-category
breakdown, a 200-item stress set, ASR throughput, and synthesis RTF /
first-audio latency harvested from the manifest. It takes any WAV + text pair
and does not know how the audio was produced, so the baseline, the teacher check
and every trained checkpoint are scored by identical code (contracts §8).

## Deliverables

| path | what |
|---|---|
| `training/voca_ru/eval/__main__.py` | CLI `python -m voca_ru.eval` |
| `training/voca_ru/eval/asr.py` | faster-whisper large-v3-turbo (CTranslate2), optional GigaAM, fake |
| `training/voca_ru/eval/scoring.py` | `score_text()` — F3 `normalize` + comparison cleanup |
| `training/voca_ru/eval/metrics.py` | Levenshtein WER/CER, no jiwer |
| `training/voca_ru/eval/runner.py` | manifest evaluation, categories, timing, JSON/MD |
| `training/voca_ru/eval/stress.py` | stress audit over the built-in set |
| `training/voca_ru/eval/data/stress_ru.jsonl` | 200 hand-built stress items |
| `tests/eval/` | 14 tests, 5 tone/noise fixtures, espeak clip generator |
| `training/voca_ru/eval/README.md` | short operator guide |

Also changed: `.gitignore` `/.cache/` → `/.cache` so the shared-cache symlink in
a worktree is ignored.

## API

```python
from voca_ru.eval import evaluate, make_backend, wer, cer, score_text

backend = make_backend("faster-whisper", model_dir=None, device="mps",
                       compute_type="float16", language="ru")
report = evaluate(rows, backend, audio_dir="audio")   # rows: id,text,audio[,category]
```

`report` keys: `overall` (`wer`, `cer`, `asr_rtf`, `asr_speed_x`, `asr_seconds`,
`audio_seconds`), `categories`, `synthesis`, `rows`, `stress` (when requested),
`asr_backend`, `model_load_seconds`, `scoring_implementation`, `warnings`.
WER/CER are **micro** (summed edit distance / summed reference length), so a
long clip does not weigh the same as a short one. Per-row values are kept.

## CLI

```sh
. scripts/cache-env.sh
PYTHONPATH=training python -m voca_ru.eval \
    --manifest m.jsonl --audio-dir audio --out report.json --md report.md

# single WAV + text
PYTHONPATH=training python -m voca_ru.eval \
    --audio clip.wav --text "Сегодня 23 января." --out one.json
```

Manifest rows are `{"id","text","audio"}`. Optional: `category`,
`synth_seconds`, `first_audio_seconds`. When `category` is absent it is looked
up in `data/scripts-ru/*.jsonl` by id (`<text_id>_F2` also resolves) or by exact
normalised text; otherwise `uncategorized`. All corpus categories plus
`template` appear in the breakdown when present.

Synthesis timing: `synth_seconds` (wall time for the clip) and
`first_audio_seconds` (request → first audio) produce `synthesis.rtf`, mean,
p95 and max. Each field is optional, so any manifest works; first-audio columns
show `—` when absent.

## ASR backends

| backend | flag | status |
|---|---|---|
| faster-whisper large-v3-turbo | `--backend faster-whisper` (default) | measured below, CPU and MPS |
| GigaAM-v3 | `--backend gigaam --gigaam-model v3_e2e_rnnt` | adapter written; package not installed, **not run** |
| fake | `--backend fake --fake-transcripts j.json` | deterministic self-check |

Model: `deepdml/faster-whisper-large-v3-turbo-ct2`, `model.bin` 1 617 884 929
bytes, sha256 `e76620f8…018da` (verified), plus `config.json`,
`tokenizer.json`, `vocabulary.json`, `preprocessor_config.json` in
`.cache/models/whisper-large-v3-turbo`. The loader raises
`FileNotFoundError`/`RuntimeError` if the model or `faster-whisper` is missing;
there is no silent fallback.

Devices: CTranslate2 4.9 has an **experimental Metal backend**, and it works
here, so MPS is the primary device for throughput; `--device cpu` remains the
default for bit-reproducible numbers. `int8` on MPS is resolved to float16 by
the library and prints a warning. Audio is decoded by `ffmpeg` to 16 kHz
float32 instead of the bundled PyAV path, because this machine has PyAV 19 and
faster-whisper 1.2.1 calls `av.open(..., metadata_errors=...)`, which PyAV 19
removed.

## Scoring text

F3 merged while F5 was running. `score_text()` imports
`voca_ru.text.normalize` (numbers/dates/money/ordinals/abbreviations → words)
and adds only comparison cleanup: ё→е, strip stress marks, lowercase, drop
punctuation. `implementation` in every report is
`voca_ru.text.normalize + eval cleanup`. The thin shim remains only as an
import fallback for an isolated checkout and would be reported as
`eval.shim (TODO F3)`; it is not used when F3 is present.

## Metrics

* WER: word-level Levenshtein (substitution/insertion/deletion) over
  `score_text` output.
* CER: character-level Levenshtein, spaces not scored.
* Categories: number, date, money, name, question, reading, conversation,
  template, uncategorized.
* Stress: `data/stress_ru.jsonl` has 200 items, each `{id, word, correct,
  wrong, group}` with U+0301 marks (or `ё`). Groups: homograph, verb, noun,
  adjective, adverb/numeral, name.
* ASR RTF = ASR seconds / audio seconds; `asr_speed_x` = audio seconds per wall
  second. Model load is reported separately.

## Measured ASR speed and memory (Apple M5 Pro)

Command shape: `PYTHONPATH=training .cache/venvs/r4/bin/python -m voca_ru.eval
--manifest … --audio-dir …`; peak RSS from `/usr/bin/time -l` (maximum resident
set size), load time from the harness itself.

| device / compute | clips | audio s | ASR RTF | audio s per wall s | load s | peak RSS |
|---|---|---:|---:|---:|---:|---:|
| MPS float16 | 6 Qwen | 32.32 | 0.147 | **6.9×** | 0.59 | 1.75 GB |
| CPU int8 | 6 Qwen | 32.32 | 0.801 | 1.25× | 0.87 | 2.30 GB |
| MPS float16 | 3 espeak | 6.24 | 0.367 | 2.73× | 0.65 | 1.74 GB |
| CPU int8 | 3 espeak | 6.24 | 1.953 | 0.51× | 1.33 | 2.27 GB |

MPS is ~5.5× faster than CPU int8 on real TTS and uses less memory. On the
robotic espeak clips float16 decoding is less stable (see below); WER on the six
Qwen clips is identical for MPS and CPU (6.33 %), so decoding variance did not
change the real number. Report the device with every WER.

## Real run 1 — Qwen3-TTS Base smoke clips (first Russian TTS round trip)

Six clips from G1 (`worktrees/g1-teacher/out/g1/base/*.wav`, 24 kHz, Qwen Base +
reference clone). Scored with MPS float16; CPU int8 gives the same WER.

| clip | category | WER | CER | transcript |
|---|---|---:|---:|---|
| 0_number | number | 25.00 % | 0.00 % | Он приехал **навстречу** на 10 минут раньше. |
| 1_date | date | 0.00 % | 0.00 % | Мы уедем в отпуск в конце июля. |
| 2_question | question | 0.00 % | 0.00 % | Как бы в такой ситуации поступила организация объединенных наций? |
| 3_long | reading | 11.11 % | 0.78 % | `в` dropped, `до робости` → `доробости` |
| 4_name | name | 0.00 % | 0.00 % | «Ну, котик, сегодня ты играла как никогда», — сказал Иван Петрович… |
| 5_plain | conversation | 0.00 % | 0.00 % | Мой коллега уходит в отпуск на две недели. |

**Overall WER 6.33 %, CER 0.27 %**, 32.3 s audio. All three word errors are
tokenisation boundaries (`на встречу` → `навстречу`, `до робости` →
`доробости`, a dropped `в`), not wrong words; CER is 0.27 %. Dates, names,
questions and plain prose are otherwise word-perfect.
Synthesis RTF from G1's `wall_s` (no first-audio field recorded): **1.79**
(57.8 s synthesis for 32.3 s audio) on MPS bfloat16. This is the first real
round-trip datapoint on Russian TTS and is far below the 5 % teacher gate, but
six clips are a smoke sample, not the 200-clip teacher check.

## Real run 2 — espeak-ng sanity (no Qwen)

`tests/eval/espeak_clips.sh` makes three 24 kHz clips (2.17 / 2.44 / 1.63 s)
from fixed sentences. CPU int8, the default reproducible device:

| clip | category | WER | CER | transcript |
|---|---|---:|---:|---|
| date | date | 50.00 % | 34.48 % | Сегодня 23 дня |
| conversation | conversation | 25.00 % | 7.14 % | Кот сидит на к ней смотрит во двор. |
| question | question | 0.00 % | 0.00 % | Сколько стоит эта книга? |

**Overall WER 25.00 %, CER 15.58 %.** MPS float16 gives 37.5 % (the
conversation clip loses `во двор` → `вот мой`); the espeak formant synthesis is
far outside Whisper's training distribution and the ASR is allowed to differ
here. The point of this run is that the pipeline works end to end: CLI →
ffmpeg decode → turbo → F3 normalisation → WER/CER → JSON/MD.

## Self-check (no speech)

`python3 -m unittest discover -s tests/eval -v` → **14 tests, OK**. Five
generated 24 kHz tone/noise clips (0.5–0.7 s) and a fake ASR:

| fixture | expected | ASR | WER |
|---|---|---|---:|
| tone_exact | один два три | identical | 0 % |
| tone_other | четыре пять шесть семь | альфа бета гамма дельта | 100 % |
| noise_exact | сегодня хорошая погода | identical | 0 % |
| noise_other | завтра будет дождь | альфа бета гамма | 100 % |
| mixed_exact | цена пятьсот рублей | identical | 0 % |

Aggregate WER 43.75 % / CER 36.59 % — the expected mix of three 0 % and two
100 % rows; timing is recorded for every row. Missing audio and missing model
fail loudly.

## What is stubbed / not verified

1. **Stress error detection is untested.** The 200-item list is complete, but
   ASR transcripts carry no stress marks, so `evaluate_stress` reports the
   target-word miss rate and leaves `stress_error_rate: null` with
   `detector: null`. `meter2` (Apache-2.0, from the adapter repo) is the
   intended audio detector; it is not installed here and nothing was faked.
   The ≤3 %/≤1 % stress gate cannot be quoted from F5 yet.
2. **GigaAM cross-check** — adapter only; `gigaam` is not installed, no number.
3. **English checkpoint sanity number** (contracts gate) — no English
   checkpoint/audio in this worktree; not run.
4. **First-audio latency** — the instrument aggregates
   `first_audio_seconds` (mean/p95/max), but the G1 metrics JSON recorded only
   `wall_s`, so no first-audio datapoint exists yet; F4/D2 manifests should
   include it.
5. **Stress-set correctness** — hand-built and spot-checked (`цеме́нт`,
   `сверло́` corrected), but a native-speaker pass is required before the set
   is used as a gate.

## Reproduce

```sh
. scripts/cache-env.sh
PYTHONPATH=training python -m voca_ru.eval \
    --manifest .cache/eval/espeak/manifest.jsonl --audio-dir .cache/eval/espeak \
    --device mps --compute-type float16 --out /tmp/espeak.json --md /tmp/espeak.md
python3 -m unittest discover -s tests/eval -v
```

## Environment

Apple M5 Pro, macOS 27.0.1, Python 3.12.16 (`.cache/venvs/r4`), faster-whisper
1.2.1, CTranslate2 4.9.0, numpy 2.5.5, ffmpeg 9.0.2, espeak-ng 1.52.0.
Tests use only the standard library.
