---
language:
  - ru
task_categories:
  - text-to-speech
  - audio-classification
pretty_name: voca-ru synthetic Russian TTS corpus
size_categories:
  - 10K<n<100K
license: cc0-1.0
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: validation
        path: data/validation-*.parquet
      - split: test
        path: data/test-*.parquet
  - config_name: annotations
    data_files:
      - split: train
        path: annotations/train-*.parquet
      - split: validation
        path: annotations/validation-*.parquet
      - split: test
        path: annotations/test-*.parquet
dataset_info:
  default:
    features:
      - {name: id, dtype: string}
      - {name: voice, dtype: string}
      - {name: text, dtype: string}
      - {name: stressed_text, dtype: string}
      - {name: category, dtype: string}
      - {name: split, dtype: string}
      - {name: source_license, dtype: string}
      - {name: source_url, dtype: string}
      - {name: seconds, dtype: float32}
      - {name: wer, dtype: float32}
      - {name: seed, dtype: int64}
      - name: audio
        dtype:
          audio:
            samplerate: 24000
            mono: true
  annotations:
    features:
      - {name: id, dtype: string}
      - {name: split, dtype: string}
      - {name: speaker, dtype: string}
      - {name: text, dtype: string}
      - {name: frames, dtype: int32}
      - {name: symbols, dtype: string}
      - {name: tokens, sequence: int32}
      - {name: stress, sequence: int8}
      - {name: durations, sequence: int32}
      - {name: seconds, dtype: float32}
      - {name: token_probability_mean, dtype: float32}
      - {name: weak_phone_fraction, dtype: float32}
      - {name: phones_per_second, dtype: float32}
      - {name: vowel_voiced, dtype: float32}
      - {name: origin, dtype: string}
      - {name: source_license, dtype: string}
      - {name: source_url, dtype: string}
      - {name: teacher_model, dtype: string}
---

# voca-ru — synthetic Russian speech corpus (two voices)

19 217 clips / **20.35 h** of Russian speech, 24 kHz mono FLAC, generated with
Qwen3-TTS-12Hz-1.7B-Base + a Russian stress LoRA from **text only** — no human
voice recordings are included or derived from. Two voices, `f_neutral` and
`m_neutral`, cloned from reference voices invented by Qwen3-TTS VoiceDesign
from a text description.

The corpus exists to train the 4.5M-parameter `sultson/voca` student for
Russian — the same recipe as `n4ze3m/tiny-voca` (English) and the community
ports (Spanish, Italian, Polish).

## Contents

```
data/train-*.parquet      9 shards of 2 000 rows, audio embedded
data/validation-0000.parquet
data/test-0000.parquet
manifest.jsonl       one row per accepted clip (19 217)
review.jsonl         clips rejected by screening (1 783 rows, reasons included)
screen-stats.json    screening counters of the last pass
audio/f_neutral/     9 622 flac
audio/m_neutral/     9 595 flac
```

`load_dataset("SuhEugene/voca-ru-corpus")` reads the Parquet shards: `audio`,
`text` and `stressed_text` per row. The FLAC tree is there for direct file
access; both carry the same 19 217 clips.

`manifest.jsonl` fields of interest: `text` (plain), `stressed_text` (every word
marked with U+0301 after the stressed vowel), `phonemes` (eSpeak-NG `ru`
inventory, token layer of the string above), `tokens` (ids in the project's
76-symbol table), `seconds`, `wer`, `voice`, `split`, `category`, `seed`,
`attempt`, `source_license`, `source_url`.

## Splits and batches

One config, three splits — `train` 18 057 / `validation` 592 / `test` 568 —
frozen **per sentence**: a sentence is never split across two sets, so
validation and test contain text the model has never heard.

Upstream `n4ze3m/tiny-voca` publishes **two configs**, `batch1` (train /
validation / test) and `batch2` (train only), because its corpus was generated
in two separate waves: batch 1 is LibriTTS-R passages plus written sentences
with the val/test split frozen on them, and batch 2 is a later wave of CC0 text
and sentence templates added to the training set only. Nothing about the model
needs that shape — it is a bookkeeping split reflecting two generation runs.

This dataset was built the same way but is presented as one config. Wave 1
(batch1 + batch2 from the English recipe's structure: 8 000 sentences,
16 000 clips) was generated first; a second wave of 2 500 new sentences
(batch3, 5 000 clips) was added afterwards to reach 20 h. All of it is
`train` except the frozen val/test sentences, and each row's provenance is in
`source_url` / `source_license`, so a second config would only duplicate the
same clips under another name. Use the `category` and `source_*` columns to
slice the data instead.

## Annotations

`annotations/` (config `annotations`) is the forced-alignment layer used to
train the student: for each accepted clip the phoneme sequence the aligner was
given and the frame duration it returned, from
`wav2vec2-lv-60-espeak-cv-ft` over the Russian eSpeak NG notation.

18 148 of the 19 217 clips (94.4 %, 19.63 h) pass the alignment gates and
appear here; train 17 056, validation 559, test 533. One row per clip, keyed by
the same `id` as the audio config:

| column | meaning |
|---|---|
| `symbols` | phoneme sequence in eSpeak-ru notation, `<bos>`/`<eos>`/`<sp>` delimiters |
| `tokens`, `stress` | the same sequence as the 76-symbol inventory ids and the stress flags the model is trained on |
| `durations` | frame-aligned duration per token (hop 256 @ 24 kHz ≈ 10.67 ms) |
| `token_probability_mean`, `weak_phone_fraction`, `phones_per_second`, `vowel_voiced` | the QC numbers that decided acceptance |

Rejected clips (1 069) are not published: 643 low confidence, 486 weak
alignment, 225 unvoiced vowels, 8 overlong phone.

Acoustic features (log-mel, pitch, energy) are deliberately **not** published:
they are ~8.8 GB of tensors that `prepare.py` recomputes from the FLAC in
minutes and that are tied to one frontend configuration.

## Numbers

| | |
|---|---|
| clips / duration | 19 217 / 20.35 h |
| voices | f_neutral 9 622, m_neutral 9 595 |
| splits | train 18 057, val 592, test 568 |
| round-trip WER (Whisper large-v3-turbo) | mean **0.87 %**, p95 9.09 % |
| stress errors (meter2 vs RUAccent marks) | **0** of 19 217 clips |
| sample rate / format | 24 000 Hz, mono, PCM_16 FLAC, peak 0.95 |

Splits are frozen **per sentence**: the same sentence never appears in two
splits, so val/test contain unseen texts, not unseen audio of seen texts.

Categories in the manifest: conversation 11 815, number 1 759, reading 1 649,
question 1 034, date 858, name 829, money 602, plus a few smaller ones. Numbers,
dates and money are written as digits in `text` and normalised to words before
synthesis.

## How it was made

1. **Texts** (10 500 sentences, see `docs/scripts-ru` in the repo): Mozilla
   Common Voice ru sentence collector (CC0), Russian Wikisource public-domain
   prose (authors died before 1917), project-written sentences and templates
   for numbers/dates/money/names/questions. All released here as CC0-1.0 apart
   from the Wikisource sentences, which stay public domain.
2. **Frontend**: normalisation (numbers, dates, money, abbreviations, «ё») →
   RUAccent stress (MIT) written as U+0301 → eSpeak-NG 1.52 `ru` phonemes →
   76-symbol token ids.
3. **Teacher**: Qwen3-TTS-1.7B-Base + LoRA
   [`siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf`](https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf)
   (Apache-2.0), batch 24, bf16, length-bucketed prompts, seed per batch.
4. **Screening**: every clip is transcribed by Whisper large-v3-turbo
   (`deepdml/faster-whisper-large-v3-turbo-ct2`) and rejected when WER > 15 %;
   stress is checked against the RUAccent marks by
   [`meter2`](https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf)
   (wav2vec2 CTC over the stressed string); duration ratio, silence, tempo and
   clipping are also checked. 8.3 % of clips were rejected; a rejected sentence
   is retired from the text list after one regeneration attempt.

## Licence

Text: **CC0-1.0** (Common Voice sentence collector, project-written sentences
and templates) and **public domain** (Wikisource prose).

Audio: generated by Qwen3-TTS (Apache-2.0), so no speaker rights are involved —
the voices do not belong to any person.

The dataset as a whole is released under **CC0-1.0**.

## Intended use and limits

For training small Russian TTS models, forced-alignment experiments and ASR
evaluation on synthetic speech. Not suitable as a model of spontaneous speech:
every sentence is read in a neutral tone from a script, with no emotion labels,
no noise, no overlapping speakers.
