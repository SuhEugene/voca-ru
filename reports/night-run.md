# Night run: 104 000-step training on a rented RTX 4090

Date: 2026-11-24 23:15 UTC → 2026-11-25 06:05 UTC.

## Result

| | |
|---|---|
| steps | **104 000 / 104 000** (full recipe, GAN from 8 000) |
| final loss | 9,73 |
| throughput | 5,5–5,7 steps/s (spectral phase 11,6 steps/s) |
| round-trip WER | **4,76 %** (Whisper large-v3-turbo, 5 fixed phrases, 21 words) |
| CER | **0,0 %** |
| wall time | 6 h 50 min |
| rent | $0,398/h → **$2,72** (budget $7, ceiling 15 h) |

Per-phrase result at step 104 000:

```
plain     Мой коллега уходит в отпуск на две недели.            perfect
question  Как бы в такой ситуации поступила ООН?                perfect (case only)
number    Он приехал на встречу на десять минут раньше.         "на встречу" → "навстречу"  (1 word error)
date      Мы уедем в отпуск в конце июля.                       perfect
name      Ну, Котик, сегодня ты играла…                         perfect (case only)
```

The single error is the known Whisper segmentation artefact (`на встречу` vs
`навстречу`), the same one the teacher corpus showed at 6,33 % — not a
synthesis defect.

WER is flat at 4,76 % from step 16 000 to 104 000: the model converges early on
this 5-phrase set and the set is too small to resolve further gains. A larger
eval set is needed before claiming a number comparable to upstream.

## Infrastructure notes

- vast.ai contract 62425809, RTX 4090 24 GB, driver 580.142, CUDA 13.0,
  image `pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime` (torch 2.14.1+cu126
  initialises CUDA on this driver).
- The host kills the training process roughly once an hour (exit 0, no
  traceback, no OOM in the logs). `tmux` alone does not survive it; a
  `setsid nohup` supervisor that restarts `run.sh train` does, resuming from
  `last.pt` (worst case 1 000 steps lost, ~3 min).
- `prepare.py` is single-process: 5 files/s at 9 % CPU on 96 cores. Sharding it
  across 12 processes gave 55 files/s (`scripts/prepare-parallel.sh`).
- Downloading 19 217 loose FLAC files topped out at 550 KB/s (HTTP round-trip
  bound). The same audio inside 12 parquet shards came down at 56 MB/s; unpack
  with `scripts/extract-parquet-audio.py`.
- WER eval needed three things that were missing on the pod: `espeakng-loader` +
  `ruaccent` (sample synthesis), the CTranslate2 Whisper model at
  `.cache/models/whisper-large-v3-turbo`, and `ffmpeg`.

## Artefacts (verified by sha256 both sides)

```
.cache/models/voca-ru-run/
  inference.pt     18 209 221 B  b9b2d65db1126f43fbfef093c92411e3b45c227f2b0018ca6e634e96816d37e6
  last.pt          87 142 376 B  f5823ca1bc5467d3f077942ce1b5d21bc3278a05f4fe4e74df0d5bbd6f8210ab
  metrics.jsonl     10 405 rows  cd5965a692eff02c4bcc70ec7bdbbf4d2233f275161ebc25b7711ce63fb67a4a
  eval-*.json       45 files
  samples/          48 step directories (5 wavs each)
```

`inference.pt` is the deployable checkpoint: 184 tensors, 4 537 291 parameters,
the same shape as the English release, Russian 76-symbol table.

## What is not done

- No listening test of the 104 000-step samples (only WER).
- Eval set is 5 phrases; a real evaluation needs the full 568-clip test split
  with per-category WER and a stress-error pass through meter2.
- The Rust side still speaks English: `crates/tiny-voca` needs the Russian
  symbol table, the `voca-ru-text` frontend wired into `tts.rs`, and a Russian
  `voca-data.tar.gz`.
- Model card and licence review (eSpeak-derived G2P lexicon, RUAccent MIT,
  research-only scope) are untouched.
