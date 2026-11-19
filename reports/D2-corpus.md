# D2 — corpus builder + pilot (f_neutral, m_neutral)

**Owner:** D2 · **Branch:** `task/d2-corpus` · **Date:** 2026-11-19
**Scope:** build the generation/screening pipeline, run a 40-clip pilot across
categories, stop before the full 8 000-sentence run.

## TL;DR

- **LoRA loads with `peft`.** 392 adapter tensors, 0 missing, 0 unexpected
  (`inject_adapter_in_model` on `model.talker.model`; `adapter.json` r=16,
  alpha=32). The run records `teacher_adapter = .cache/models/qwen3-tts-ru-stress/lora`.
- **Pilot: 40/40 clips accepted, 0 dropped, 0 retries, 107.7 s of audio.**
  WER mean **0.67 %**, max **14.3 %**; 38/40 clips at 0 %.
- **Stress: 0 failed clips.** meter2 checked **184** multi-vowel word instances
  against the RUAccent marks; **0** disagreements (margin ≥ 0.5, align ≥ 0.5).
- **Speed:** measured batch 8 on length-bucketed short text = **2.43× realtime,
  56.5 clips/min**; the whole pilot including screening = **11.4 clips/min,
  RTF 1.23** (0.81× realtime). Peak RSS **1.96 GiB**, peak MPS **13.63 GiB**.
- One real teacher error found: `m_neutral` says «по́езд» as «пояс» in two date
  sentences (Whisper WER 12.5 %/14.3 %). `f_neutral` reads the same sentence at
  0 %; flagged for the human listen (`out/d2/LISTEN.md`).
- **Not started:** the full run. Waiting for the adapter verdict + human go-ahead.

## 1. What was built

| path | job |
|---|---|
| `scripts/gen/build_corpus.py` | sentence selection, F3 text path, Qwen3-TTS Base + LoRA, length buckets, retries, resume, manifest/review/stats |
| `scripts/gen/screen.py` | trim + peak-normalise, Whisper WER, duration/rate/level checks, meter2 stress check |
| `tests/gen/test_corpus.py` | 9 model-free tests (trim, peak, buckets, seed stability, pilot selection, stress comparison, good/reject screen) — all pass |
| `scripts/gen/README.md` | operator guide |
| `assets/voices/{f_neutral,m_neutral}.{wav,txt}` | clone references copied from `out/g1/voices/` (audio is git-ignored) |
| `out/d2/samples/` + `out/d2/LISTEN.md` | 4 samples + Russian listening note (git-ignored) |
| `data/corpus-ru/` (git-ignored) | `manifest.jsonl`, `review.jsonl` (empty), `audio/<voice>/*.flac`, `rejected/`, `run-stats.json` |

The manifest row matches the requested schema and adds provenance/screening
fields:

```json
{"id":"<text_id>_<voice>","text":"…","norm_text":"…","stressed_text":"…",
 "phonemes":["<bos>",…],"voice":"f_neutral","audio":"audio/f_neutral/<id>.flac",
 "seconds":1.84,"wer":0.0,"stress_ok":true,"seed":1234567890, ...}
```

`norm_text` is F3's `norm_text`, which by contract carries the stress marks;
`stressed_text` is the same string (kept separately for clarity).

## 2. Text path (F3, not reimplemented)

`voca_ru.text.analyze()` per sentence: `normalize` → RUAccent stress →
eSpeak G2P. 20 pilot sentences, F3 frontend 3.9 s (mostly the 1.5 GiB RUAccent
dictionary load). Every prompt string carries U+0301 after the stressed vowel;
the run asserts this and records five examples (`run-stats.json` `mark_examples`):

| source | prompt string | marks |
|---|---|---:|
| Ты не забыл принять таблетку? | `ты́ не забы́л приня́ть табле́тку?` | 4 |
| Что Елена думает о нашем плане? | `что еле́на ду́мает о на́шем пла́не?` | 4 |
| Кто звонил Ольге сегодня утром? | `кто́ звони́л о́льге сего́дня у́тром?` | 5 |
| Я передал документы Марии вчера. | `я́ переда́л докуме́нты мари́и вчера́.` | 5 |
| Ах, мне хорошо, — проговорила она. | `а́х, мне́ хорошо́, — проговори́ла она́.` | 5 |

Phonemes in the manifest are F3's 76-symbol token stream (BOS/EOS included).

## 3. Teacher and adapter

- Base `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, MPS, bfloat16, `eager`.
- LoRA `qwen3-tts-ru-stress/lora`: `peft.inject_adapter_in_model` with the
  adapter's regex target list; state keys remapped from
  `…lora_A.weight` to `…lora_A.default.weight`. `load_state_dict` reports
  **0 missing LoRA tensors, 0 unexpected**. Base load 2.3 s; LoRA injection
  under 1 s.
- Clone prompt per voice = reference WAV + exact transcript,
  `x_vector_only_mode=False`. Prompt build 0.22–0.26 s.
- If the adapter ever fails, the script prints a 78-character `!!` banner and
  continues on Base; `teacher_adapter` then contains `none (LoRA failed: …)`.
  Nothing is dropped silently.

## 4. Batching (measured, not assumed)

Same 8 shortest pilot sentences, one voice, one seed, measured before the run:

| batch | clips | audio s | wall s | RTF | × realtime | clips/min | peak MPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 8.88 | 6.49 | 0.731 | **1.37×** | 37.0 | 7.01 GiB |
| 8 | 8 | 20.64 | 8.50 | 0.412 | **2.43×** | 56.5 | 9.01 GiB |

**Batch 8 chosen.** Batch 16 was not measured: G1 saw 14.05 GiB at batch 8 with
a 12 s outlier, already close to the 24 GiB budget, and the pilot's long prose
bucket reached 13.63 GiB at batch 8.

Batches are formed by sorting sentences by `len(plain_text)` and chunking.
The pilot's last bucket (4 long sentences) ran at 4.6–9.3 s/clip — padding is
gone between neighbours, but a bucket of long texts still costs more each.

## 5. Pilot result

Selection: 20 sentences, evenly spread per category, both voices = 40 clips.

| category | clips | mean WER | max WER |
|---|---:|---:|---:|
| number | 8 | 0.0 % | 0.0 % |
| date | 8 | 3.35 % | 14.3 % |
| money | 6 | 0.0 % | 0.0 % |
| name | 6 | 0.0 % | 0.0 % |
| question | 6 | 0.0 % | 0.0 % |
| conversation | 4 | 0.0 % | 0.0 % |
| reading | 2 | 0.0 % | 0.0 % |

WER distribution: **0.0 % × 38, 12.5 % × 1, 14.3 % × 1**. Speaking-rate and
duration checks: duration ratio 0.67–1.25 (window 0.6–1.6), no clip outside
0.5–35 s or 0.5–7 words/s, no raw clipping, RMS 0.100–0.197 after peak
normalisation. All 40 FLACs are 24 000 Hz mono PCM16; every peak is 0.950.
Silence trim removed up to 0.555 s lead / 0.284 s tail.

The two WER misses are the same word in `m_neutral`: «По́езд прибывает…» was
transcribed as «Пояс прибывает…» (the `f_neutral` reading of the identical
sentence is 0 %). This is either a teacher pronunciation error or an ASR
artefact; the human listen decides. It passed the ≤15 % screen, so it was kept.

## 6. Stress screening with meter2

`screen.py` preloads the meter2 CTC encoder and head from
`.cache/models/wav2vec2-xlsr-ru` and
`.cache/models/qwen3-tts-ru-stress/meter/head.pt`, then for every multi-vowel
word whose mark is present, meter2 aligned confidently (`align ≥ 0.5`) and its
own margin ≥ 0.5, compares `best` with the RUAccent vowel index.

| metric | value |
|---|---:|
| clips with ≥1 checked word | 40 / 40 |
| checked word instances | 184 |
| disagreements | **0** |
| clips failing the stress rule (>10 % checked words wrong, or any error on clips with ≤2 checked) | **0** |
| meter2 load / judge time | ~2 s once; ≈0.7 s per clip on CPU |

The threshold is per clip, not the corpus-level 3 % gate; the corpus gate will
be computed once the full run exists.

## 7. Speed and memory

| measure | value |
|---|---:|
| clips generated (incl. no retries) | 40 |
| audio produced | 107.73 s |
| generation wall | 132.33 s |
| generation RTF | 1.228 (0.81× realtime) |
| whole pilot wall (load + F3 + screen) | 210.63 s |
| clips/min (whole pilot) | 11.39 |
| measured batch-8 RTF on short text | 0.412 (2.43× realtime) |
| process peak RSS | 1.96 GiB |
| peak MPS driver memory | 13.63 GiB |

At the measured 2.43× and 19 h of target audio, generation is ≈ 2.2 h; batch 8
with long prose buckets will be slower, so budget **3–4 h** on this M5.

## 8. Open issues / what could not be verified

1. **Full run not started** (per instructions). Resume was verified by rerunning
   the pilot command: 0 clips generated, `manifest.jsonl` stayed at 40 rows,
   `run-stats.json` was left untouched. A kill *mid-batch* (partial batch) was
   not tested; the id filter and stable batch seeds make it equivalent in code,
   but it is unproven on disk.
2. **`m_neutral` «поезд»** — see §5; human listen needed. If it is a teacher
   error, expect a small WER tail on other words too; the 40-clip pilot is too
   small to estimate it.
3. **meter2 CTC checkpoint is 9 bytes short and fails its recorded SHA-256.**
   `.cache/models/wav2vec2-xlsr-ru/pytorch_model.bin` is 1 262 093 719 bytes
   (expected 1 262 093 728, `sha256 d1cdb1a7…` vs `5874e4e0…` in
   `download.log`; `parts-pytorch_model.bin/part.15` is 9 bytes short).
   `torch.load` reads all 424 tensors and meter2 returns plausible answers, but
   this file is **not trusted** for publication numbers. Re-fetch it before the
   full run. Workaround used here: meter2's own `load_ctc` calls
   `hf_hub_download` even for a local path, so `screen.py` preloads the encoder
   from the local directory instead (no network).
4. **No first-audio latency.** The `qwen_tts` API is not streamed; the manifest
   has `synth_seconds` (batch wall / clips) and no `first_audio_seconds`.
5. **Stress check only covers words meter2 is confident about.** 184 checked
   instances over 40 clips, none flagged; the adapter's headline 96 % adherence
   against the dictionary cannot be confirmed or refuted at this sample size.
6. **Retry path exercised separately.** No pilot clip failed, so the pilot
   itself never retried. A forced-failure smoke (`--max-wer -1`, one sentence,
   one voice) generated attempt 1, regenerated once as attempt 2, then dropped:
   `data/corpus-ru-retry/review.jsonl` has the reason `wer 0.000 > -1.0` and
   `attempt: 2`; 0 accepted rows. The branch works, but a *real* quality failure
   on the pilot corpus has not been seen.

## 9. Reproduce

```sh
cd worktrees/d2-corpus
. scripts/cache-env.sh
cp ../../out/g1/voices/{f_neutral,m_neutral}.{wav,txt} assets/voices/   # if missing
PYTHONPATH=training .cache/venvs/r4/bin/python -m unittest discover -s tests/gen
PYTHONPATH=training .cache/venvs/r4/bin/python scripts/gen/build_corpus.py \
    --pilot --voices f_neutral,m_neutral --batch-size 8 --measure-batch 4,8

# force the retry→drop path (no model output is kept):
PYTHONPATH=training .cache/venvs/r4/bin/python scripts/gen/build_corpus.py \
    --pilot --limit 1 --voices f_neutral --batch-size 1 --max-wer -1 \
    --out data/corpus-ru-retry
```

Outputs: `data/corpus-ru/` (git-ignored), `out/d2/LISTEN.md`,
`out/d2/samples/*.flac`. To run the full corpus later, drop `--pilot` (and
optionally split it per voice); the same resume rules apply.
