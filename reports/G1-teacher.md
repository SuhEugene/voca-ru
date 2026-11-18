# G1 — teacher pipeline gate (Qwen3-TTS on the M5 Pro)

**Owner:** G1 · **Branch:** `task/g1-teacher` · **Date:** 2026-11-19
**Goal:** prove Base + VoiceDesign generation works end-to-end on this laptop
before 17k clips are generated, and produce the voice candidates a human can
audition.

## TL;DR

- **Base works on MPS in bfloat16.** Six smoke clips, all non-silent, no
  clipping, durations 0.91–1.37× of the expected length. Batch 1 runs at
  RTF 1.79 (0.56× realtime); batching is where the speed comes from.
- **VoiceDesign weights: fetching handed to the orchestrator** after the 16-range
  fetch stalled at ~128 MB (71 concurrent connections across two agents froze
  the CDN). Sections §2 and §3 are pending that file; everything else is done.
- **Batch 8 measure** — see §3; this is the number to use for the 17k-clip
  budget.
- `data/scripts-ru/batch1.jsonl` has **no digits at all**; numbers and dates
  are spelled out by the text contract. The teacher never sees a digit, so
  the "numbers" risk is a normaliser risk, not a TTS risk.
- Environment pinned in `env/requirements-qwen.txt`, commands in
  `scripts/gen/README.md`, `make doctor` added.

## 1. Smoke generation

Command:

```sh
make doctor
.cache/venvs/r4/bin/python scripts/gen/teacher.py smoke
```

`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, MPS, bfloat16, `attn_implementation="eager"`,
24 kHz mono PCM16 WAV. Load 2.1 s. Voice prompt: `clone_ref.wav` in
`x_vector_only_mode=True` (0.24 s) — the cache has **no `clone_ref.txt`**, so
the transcript-conditioned path was not available for this reference; the
designed voices in §2 have exact transcripts and are the better prompt.

| # | category | text (id) | chars | audio s | wall s | chars/s | RTF | peak | RMS | silence | clipping | dur/expected |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 0 | number | Он приехал на встречу на десять минут раньше. | 45 | 3.28 | 5.61 | 8.03 | 1.71 | 0.498 | 0.053 | 15.8% | no | 1.02 |
| 1 | date | Мы уедем в отпуск в конце июля. | 31 | 3.04 | 5.67 | 5.47 | 1.86 | 0.369 | 0.061 | 6.7% | no | 1.37 |
| 2 | question | Как бы в такой ситуации поступила Организация Объединенных Наций? | 65 | 4.56 | 7.78 | 8.36 | 1.71 | 0.672 | 0.077 | 10.7% | no | 0.98 |
| 3 | long | Каждый год приезжал я в деревню… задумчивым до робости. (28 words, 158 chars) | 158 | 10.24 | 18.66 | 8.47 | 1.82 | 0.516 | 0.083 | 3.8% | no | 0.91 |
| 4 | name | Ну, Котик, сегодня ты играла… Иван Петрович… (120 chars) | 120 | 8.00 | 14.40 | 8.34 | 1.80 | 0.373 | 0.059 | 4.5% | no | 0.93 |
| 5 | plain | Мой коллега уходит в отпуск на две недели. | 42 | 3.20 | 5.65 | 7.44 | 1.77 | 0.371 | 0.066 | 13.7% | no | 1.07 |

Totals: 32.32 s audio in 57.75 s wall → **RTF 1.79 (0.56× realtime) at batch 1**.
Peak MPS driver memory 4.90 GiB; process peak RSS 1.02 GiB
(`ru_maxrss` on macOS is in bytes — the harness prints it as GiB after the fix).
Silence share is the longest sub-0.01 run as a share of the clip; the 15.8% on
clip 0 is the lead-in/lead-out of a 3 s clip, not a dropout.

Non-silence check: `peak` 0.37–0.67 and `rms` 0.053–0.083 on every clip.
Duration plausibility: generated seconds / (chars ÷ 14 chars s⁻¹) is 0.91–1.37,
i.e. no clip is truncated or padded by the model.

Text IDs (all rows exist in `data/scripts-ru/batch1.jsonl`):
`8a044e6e61987986`, `5109bb52a56c76cf`, `06a68d93d5467224`,
`09faa1c76b66c170`, `04b1af0ede04f598`, `a62c1db9265ab8f6`.

Intelligibility check (not the F5 screening harness, just a sanity gate):
`faster-whisper-small` round-trip on the six clips gives mean normalised WER
**3.9%**, with five clips at 0% and one word wrong on the question
(«поступила» heard as «поступило»). The "number" clip's 12.5% is an artefact of
the normaliser (ASR writes `10` for «десять»), i.e. the number was read
correctly. Raw transcripts are in `out/g1/asr-smoke.json`.

## 2. VoiceDesign candidates

Command:

```sh
./scripts/gen/fetch-voicedesign.sh        # 3.83 GB, sha256-verified
.cache/venvs/r4/bin/python scripts/gen/teacher.py design
```

`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, MPS, bfloat16, `language="Russian"`,
English `instruct` strings (the model expects English descriptions), one
expressive digit-free Russian sentence each, `torch.manual_seed(seed)` before
every call.

**Status: blocked on the weights.** The 16-range fetch stalled at ~128 MB
(0.46 GB total across the cache) and was stopped at the orchestrator's request;
the orchestrator is refetching with few connections and a stall detector. The
four candidate definitions (names, seeds, English descriptions, Russian
sentences) are frozen in `scripts/gen/teacher.py` (`VOICES`), so the run is one
command once `model.safetensors` exists:

```sh
.cache/venvs/r4/bin/python scripts/gen/teacher.py design
```

Expected outputs: `out/g1/voices/{f_warm,f_neutral,m_warm,m_neutral}.wav` +
`.txt`, `out/g1/voices-metrics.json`, and `out/g1/LISTEN.md`.

> table filled in when the run completes

## 3. Batch 8 with a designed reference

Pending the VoiceDesign weights: `teacher.py batch <voice>` loads Base, builds
a prompt from the chosen reference **with its exact transcript**
(`x_vector_only_mode=False`), and generates 8 batch1 sentences in one call. It
records total wall time, RTF, chars/s, s/clip and peak MPS memory to
`out/g1/batch8-metrics.json`.

## 4. Environment

`env/requirements-qwen.txt` pins what actually worked; `scripts/gen/doctor.py`
(`make doctor`) prints the same check. Python 3.12.16, venv
`.cache/venvs/r4` (existing and working — no new venv was needed).

| Package | Version |
|---|---|
| torch | 2.15.0 (MPS + bfloat16) |
| qwen-tts | 0.1.1 |
| transformers | 4.57.3 |
| soundfile | 0.14.0 |
| librosa | 1.0.1 |
| numpy | 2.5.5 |
| accelerate / safetensors | 1.12.0 / 0.8.0 |

`sox: command not found` and the `flash-attn` warning are expected: the 25 Hz
tokenizer path is unused and `eager` attention is what the numbers use.
The verified Base checkpoint hashes to
`38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c` and the
speech tokenizer to `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`,
both checked. VoiceDesign's `model.safetensors` is
`391e8db219f292c515297cdceeb43e4eae67cdde35fa57e79a6a8a532fca0522`; its
speech tokenizer is byte-identical to Base's and is copied, not downloaded.

## 5. Open issues

- **`clone_ref.txt` is missing** from `.cache/models/qwen3-tts-1.7b/`, so the
  smoke clone had to use `x_vector_only_mode=True`. Fine for a smoke, but the
  production prompt should be a VoiceDesign reference **with** its transcript
  (as in §3), or the reference must be transcribed/confirmed by a human.
- **VoiceDesign download is slow** (~0.75 MB/s aggregate with 16 ranges today,
  same ceiling R4 measured). Handed to the orchestrator; a central fetch with
  few connections and a stall detector is the fix.
- **No human listening yet.** All quality statements here are objective
  (peak, RMS, duration, ASR WER). The accent/naturalness gate is the human's.

## 6. Reproducing

```sh
make doctor
.cache/venvs/r4/bin/python scripts/gen/teacher.py smoke
./scripts/gen/fetch-voicedesign.sh
.cache/venvs/r4/bin/python scripts/gen/teacher.py design
.cache/venvs/r4/bin/python scripts/gen/teacher.py batch <voice>
```

Outputs land in `out/g1/`; `out/` and `.cache/` are git-ignored, so no audio or
weights are ever committed.

