# F5 — evaluation harness

Status: **harness complete and tested; real ASR run pending model download**
(branch `task/f5-eval`, 2026-11-19).

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
| `training/voca_ru/eval/scoring.py` | `score_text()` — F3 import or explicit shim (`TODO F3`) |
| `training/voca_ru/eval/metrics.py` | Levenshtein WER/CER, no jiwer |
| `training/voca_ru/eval/runner.py` | manifest evaluation, categories, JSON/MD |
| `training/voca_ru/eval/stress.py` | stress audit over the built-in set |
| `training/voca_ru/eval/data/stress_ru.jsonl` | 200 hand-built stress items |
| `tests/eval/` | 14 tests, 5 tone/noise fixtures, espeak clip generator |
| `training/voca_ru/eval/README.md` | short operator guide |

Also changed: `.gitignore` `/.cache/` → `/.cache` so the shared-cache symlink in
a worktree is ignored.

## API

```python
from voca_ru.eval import evaluate, make_backend, wer, cer, score_text

backend = make_backend("faster-whisper", model_dir=None, device="cpu",
                       compute_type="int8", language="ru")
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
normalised text; otherwise `uncategorized`. All seven corpus categories plus
`template` appear in the breakdown when present.

Synthesis timing: `synth_seconds` (wall time for the clip) and
`first_audio_seconds` (request → first audio) produce `synthesis.rtf`, mean,
p95 and max. They are optional so any manifest works.

## ASR backends

| backend | flag | status |
|---|---|---|
| faster-whisper large-v3-turbo | `--backend faster-whisper` (default) | model path `.cache/models/whisper-large-v3-turbo`; CPU or experimental MPS |
| GigaAM-v3 | `--backend gigaam --gigaam-model v3_e2e_rnnt` | adapter written, package not installed here, not run |
| fake | `--backend fake --fake-transcripts j.json` | deterministic self-check |

Model directory: `deepdml/faster-whisper-large-v3-turbo-ct2`, `model.bin`
1 617 884 929 bytes, sha256 `e76620f8…018da`, plus `config.json`,
`tokenizer.json`, `vocabulary.json`, `preprocessor_config.json`. The loader
raises `FileNotFoundError`/`RuntimeError` if the model or `faster-whisper` is
missing; there is no silent fallback to another model.

Devices: CTranslate2 4.9 has an **experimental Metal backend**, so
`--device mps` works (int8 is resolved to float16 by the library and prints a
warning); `--device cpu` is the default so numbers are reproducible. Audio is
decoded by `ffmpeg` to 16 kHz float32 instead of the bundled PyAV path, because
this machine has PyAV 19 and faster-whisper 1.2.1 calls
`av.open(..., metadata_errors=...)`, which PyAV 19 removed.

## Metrics

* WER: word-level Levenshtein (substitution/insertion/deletion) over
  `score_text` output.
* CER: character-level Levenshtein, spaces not scored.
* Categories: number, date, money, name, question, reading, conversation,
  template, uncategorized — from the corpus `category` field.
* Stress: `data/stress_ru.jsonl` has 200 items, each `{id, word, correct,
  wrong, group}` with U+0301 marks (or `ё`). Groups: homograph, verb, noun,
  adjective, adverb/numeral, name. ASR text cannot show stress; the audit
  reports target-word survival and leaves stress errors to an audio detector.
* ASR RTF = ASR seconds / audio seconds; `asr_speed_x` = audio seconds per wall
  second. Model load is reported separately.

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
100 % rows. Timing fields are recorded for every row; missing audio and missing
model fail loudly.

Real espeak-ng run: **pending model download** (orchestrator is fetching the
weights). Clips and manifest are reproducible with
`tests/eval/espeak_clips.sh`; three 24 kHz clips (date / conversation /
question), 2.17 s, 2.37 s, 1.63 s.

## Measured ASR speed

Pending the large-v3-turbo weights. A smaller `faster-whisper-small` smoke test
already proved the backend end to end: 3 clips, 6.2 s audio, CPU int8 2.1× audio
speed, MPS 1.6× faster than CPU on a single 2.2 s clip (0.98 s → 0.61 s
transcription, model already loaded). Small-model WER on robotic espeak audio
was 62.5 %, which says nothing about turbo.

## What is stubbed / not verified

1. **Text normaliser** — F3 has not merged. `scoring.py` imports
   `voca_ru.text.normalise_for_scoring` when present and otherwise runs the
   shim: lowercase, ё→е, strip stress marks/punctuation, digits→Russian
   cardinal words (0…999 999 999). Dates are cardinals (`23 января` →
   `двадцать три января`), not ordinals; years have no case agreement. Every
   report records `scoring_implementation`. Switching is one import.
2. **Stress error detection** — the 200-item list is complete, but ASR
   transcripts carry no stress marks. `evaluate_stress` reports word-miss rate
   and refuses to invent a stress-error rate without an audio detector
   (`detector: null`, note in the report). `meter2` from the adapter repo is the
   intended detector; it is not installed here.
3. **GigaAM backend** — adapter only; `gigaam` is not installed and no cross-check
   number exists.
4. **English checkpoint sanity number** (contracts gate) — no English
   checkpoint/audio was available in this worktree; not run.
5. **Synthesis timing** — the harness aggregates the manifest fields, but F5
   produced no synthesis run (no Qwen needed); RTF/first-audio numbers must come
   from F4/D2 manifests.
6. Stress-set correctness: hand-built by the agent; entries were spot-checked
   (`цеме́нт`, `сверло́` corrected), but a native-speaker pass is still required
   before the ≤3 % gate is quoted.

## Environment

Apple M5 Pro, macOS 27.0.1, Python 3.12.16 (`.cache/venvs/r4`), faster-whisper
1.2.1, CTranslate2 4.9.0, numpy 2.5.5, ffmpeg 9.0.2, espeak-ng 1.52.0.
Tests use only the standard library.
