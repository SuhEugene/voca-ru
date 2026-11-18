# `voca_ru.eval` — round-trip intelligibility

One instrument for the whole project: WER/CER, per-category breakdown, a stress
set, ASR throughput and synthesis timing. It compares any WAV + text pair and
does not know how the audio was made.

## Run

```sh
. scripts/cache-env.sh
PYTHONPATH=training python -m voca_ru.eval \
    --manifest data/synthetic-ru/test.jsonl --audio-dir data/synthetic-ru \
    --out reports/eval.json --md reports/eval.md

# single pair
PYTHONPATH=training python -m voca_ru.eval \
    --audio out/sample.wav --text "Сегодня 23 января." --out /tmp/one.json
```

`--manifest` rows: `{"id","text","audio"}`. Optional fields: `category`,
`synth_seconds`, `first_audio_seconds`. Relative `audio` paths resolve against
`--audio-dir`. Categories missing from the manifest are looked up in
`data/scripts-ru/*.jsonl` by id (also `<id>_F2` → `<id>`) or by normalised text.

## ASR backends

| flag | what | notes |
|---|---|---|
| `--backend faster-whisper` (default) | large-v3-turbo CTranslate2 | CPU; CTranslate2 has no MPS backend, so `--device mps` raises |
| `--backend gigaam --gigaam-model v3_e2e_rnnt` | GigaAM cross-check | needs `uv pip install gigaam`; fails loudly otherwise |
| `--backend fake --fake-transcripts j.json` | deterministic self-check | no speech involved |

The model directory defaults to `.cache/models/whisper-large-v3-turbo`; override
with `--model-path`. `--compute-type int8` (default) or `float32`.

## Scoring text

`training/voca_ru/eval/scoring.py` has `score_text()`. It imports
`voca_ru.text.normalise_for_scoring` when F3 lands and otherwise uses a shim
marked `TODO F3`. Every report records which implementation ran.

## Tests

```sh
python3 -m unittest discover -s tests/eval -v
```

Five generated tone/noise fixtures (no speech) prove 0 % / 100 % WER against a
fake ASR.
