# align_spike — can the upstream aligner label Russian eSpeak phones?

Measurement code behind `reports/F6-align.md`. Nothing here is part of the
training or inference path; it answers one gate question with numbers.

## What it does

1. `espeak_ru.py` — eSpeak NG 1.52 `-v ru --ipa --sep=_`, parsed exactly like
   upstream `tiny_voca.text` (stress marks become a parallel 0/1/2 feature).
2. `mapping.py` — the eSpeak-ru → `vocab.json` label table, one decision per
   mismatch (`zʲ→ʑ`, `ɕː→ɕ`, `u"→u`, `ɪ^→ɪ`, `#`-artefacts, foreign phones).
3. `align.py` — upstream CTC forced alignment (`facebook/wav2vec2-lv-60-espeak-cv-ft`,
   20 ms emissions, −19 ms onset offset, durations on the 24 kHz / hop-256 grid)
   plus a cheap autocorrelation voicing check.
4. `calibrate.py` — runs the lot over FLEURS `ru_ru` test clips and prints
   per-clip QC, per-phone duration/probability, and for every phone the model's
   top-3 labels (evidence for the mapping decisions).

## Run

```sh
. scripts/cache-env.sh
.cache/venvs/r4/bin/python scripts/align_spike/calibrate.py --clips 150 --tag fleurs
```

Inputs expected in the shared cache (see `reports/F6-align.md` for how they were
fetched):

- `.cache/models/wav2vec2-lv-60-espeak-cv-ft/` — model + `vocab.json`, commit `ae45363`
- `.cache/models/fleurs-ru/test.tsv` and `audio/test/*.wav` — FLEURS ru test

Output: `.cache/align-spike/alignment-<tag>.json` (full per-clip results).

The script is a measurement harness, not a library: no tests, because the
report's numbers are the check. Re-running it on the same clips reproduces the
same durations (deterministic CTC, no sampling).
