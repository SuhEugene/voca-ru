# Silero for Russian `voca` — what it can and cannot give us (R6)

**Date:** 2026-11-19 · **Branch:** `task/silero` · **Experiments:** Apple M5 Pro,
macOS 27.0.1, Python 3.12.16, torch 2.15.0, CPU (MPS where noted).

Silero is a family of Russian-first speech models from the same team:
TTS, stress/accentuation, VAD, text enhancement, and (historically) STT. This
document separates what we verified by running the models from what the
upstream repositories claim.

**Short answer:** Silero is a credible **second teacher and an immediate
CPU-only data source**, not a replacement for Qwen3-TTS and not a realistic
Rust runtime fallback. Its TTS is CC BY-NC-SA 4.0 (except the MIT
`v5_cis_base*` models), fixed-speaker only, has no training code, and **deletes
digits instead of reading them**, so our Russian normaliser is mandatory
whichever teacher we use. `silero-stress` is MIT but is a 31M-parameter neural
model, not a static dictionary; `ruaccent` remains the better base for the
Python trainer + Rust runtime. Silero STT has no Russian model in the current
catalogue, so it cannot be the judge.

---

## 1. Silero TTS

### 1.1 Licence — exact text and what it obliges

Three upstream statements disagree, so read the licence *file*, not the badge:

- `README.md` badge: `License: CC BY-NC 4.0`, hyperlinked to `LICENSE`.
- `LICENSE`, line 1: **“Attribution-NonCommercial-ShareAlike 4.0 International”**.
  This is the operative legal text: **CC BY-NC-SA 4.0**.
- `README.md` “Licence” section:
  > “All of the models are published under the main repo license (i.e.
  > CC-NC-BY) except for the `base` cis-tts models, which are under MIT.”
- `README.md`, V5 CIS Base section:
  > “All of the below models are published under `MIT` licence” (links
  > `LICENSE_CIS`).

`LICENSE_CIS` is a plain MIT licence (“Permission is hereby granted, free of
charge, to any person obtaining a copy…”), copyright “2025-present Silero
Team”. `v5_cis_base` and `v5_cis_base_nostress` are the only Russian-capable
models covered by it.

**Obligations if we use the main models (`v4_ru`, `v5_ru`, `v5_1_ru` … `v5_6_ru`):**

| Clause | What it means for us |
|---|---|
| **BY** | Credit Silero, give the licence link, indicate changes made. The model file and any published derivative must carry this. |
| **NC** | No use “primarily intended for or directed towards commercial advantage or monetary compensation”. Research-only use is fine; a commercial `voca` release is not. |
| **SA** | If we remix, transform, or build upon the material, the adaptation must be distributed under **the same licence** (CC BY-NC-SA 4.0). Training a model on Silero audio is a conservative reading of “build upon”; plan for the resulting weights and dataset to be **CC BY-NC-SA 4.0**, not Apache-2.0. |
| No extra restrictions | We may not add DRM/terms that forbid what the licence allows. |

This is a licence reading, not legal advice. The practical consequence is
simple: **a Silero-distilled Russian checkpoint cannot be released under the
project's Apache-2.0/MIT aspiration.** It can be researched and published
NC-SA with attribution. The MIT `v5_cis_base*` route avoids NC/SA entirely.

### 1.2 Russian catalogue (from `models.yml` + model files)

All Russian TTS entries accept **8 000 / 24 000 / 48 000 Hz** and produce mono
audio. “Stress” below means the model can add lexical stress itself from
unmarked text; `v5_cis_*` expect the caller to mark stress (`к+ошка`).

| model id | licence | speakers (gender) | stress / homographs / questions / emphasis | params | file size | ONNX |
|---|---|---|---:|---:|---:|---|
| `v3_1_ru` | CC BY-NC-SA 4.0 | aidar (M), baya (F), kseniya (F), xenia (F), eugene (M), random (5+random) | auto-stress (homograph support not verified) / ? / no / no | 14.3 M | 61.9 MB | **no** (404) |
| `v4_ru` | CC BY-NC-SA 4.0 | aidar (M), baya (F), kseniya (F), xenia (F), eugene (M), random | yes / yes / no / no | **8.89 M** | **40.1 MB** | **no** (404) |
| `v5_ru` | CC BY-NC-SA 4.0 | aidar, baya, kseniya, xenia, eugene (3 F + 2 M) | yes / yes / no / no | ~20.9 M acoustic (+ stress model in package) | 145.4 MB | **no** (404) |
| `v5_1_ru`, `v5_2_ru`, `v5_3_ru` | CC BY-NC-SA 4.0 | same 5 | yes / yes / no / no | ~20.9 M | ~145 MB each | no |
| `v5_4_ru` | CC BY-NC-SA 4.0 | aidar, baya, kseniya, xenia | yes / yes / **yes** / no | ~20.9 M | ~145 MB | no |
| `v5_5_ru` | CC BY-NC-SA 4.0 | same 5 | yes / yes / yes / no | ~20.9 M | ~145 MB | no |
| `v5_6_ru` | CC BY-NC-SA 4.0 | same 5 | yes / yes / yes / **yes** | 20.90 M acoustic + 31.4 M stress model (quantised) | **147.7 MB** | **no** (404) |
| `v5_cis_base` | **MIT** | 60 speakers, including **29 `ru_*`** | manual stress for all languages / no / no / no | ~20.9 M | 91.7 MB | no |
| `v5_cis_base_nostress` | **MIT** | same 60 / 29 `ru_*` | manual stress for Slavic only / no / no / no | 20.93 M | 91.7 MB | no |
| `v5_cis_ext` | CC BY-NC-SA 4.0 | Kazakh, Kalmyk, Tatar, Uzbek, Ukrainian, Chuvash; **no Russian speakers** | manual stress / no / no / no | — | 91.7 MB | no |
| `v5_turkic`, `v5_caucasian` | CC BY-NC-SA 4.0 | other languages only; no Russian | **none at all** | — | — | no |

Notes:

- Parameter counts and file sizes were measured by loading the downloaded
  `.pt` files, not copied from a card. `v5_6_ru` is 20.9 M acoustic
  parameters plus an embedded Silero stress package (2.06 M accentor +
  29.19 M homograph BERT, stored quantised), which explains 147.7 MB vs
  91.7 MB for the stress-less MIT model.
- **No Russian TTS model ships ONNX.** `models.yml` only lists ONNX for
  English, German, Spanish and Ukrainian STT/TTS. `v4_ru.onnx`,
  `v5_ru.onnx`, `v5_cis_base*.onnx`, `.jit` etc. all return HTTP 404 from
  `models.silero.ai`. Community conversions exist (e.g.
  `ggml-org/silero-v5.1.2`, labelled MIT, and HF mirrors of v3/v4), but none
  is an official artefact with a clear provenance chain.
- Speaker ids are not gender-labelled upstream. I measured median F0 on the
  same sentence: **baya 239 Hz, kseniya 253 Hz, xenia 195 Hz** (female);
  **aidar 129 Hz, eugene 89 Hz** (male). That matches the usual Silero
  descriptions.
- `v3_1_ru` speakers were read from the model archive (`speakers` list); the
  v3-specific stress/homograph feature flags are not documented in the README
  table and are left as “not verified”.

### 1.3 Quality evidence

**Published / third-party:**

- Silero's own v5 CIS wiki (December 2025 revision): Russian **UTMOS
  3.04 ± 0.02** (`v5_cis_base`) and **2.97 ± 0.02**
  (`v5_cis_base_nostress`) against a **control of real recordings at
  3.08 ± 0.01**. On that metric the MIT model is within 0.04 of real speech;
  the stress-less variant loses ~0.07.
- Silero's TTS v1 human evaluation (24 assessors, 37 403 ratings, 5-point
  naturalness): synthesis/original ratio **aidar 96.8 %**, **baya 94.0 %**,
  **kseniya 94.5 %** (8 kHz numbers; 16 kHz similar). Small, old, and
  company-run, but the methodology is described honestly.
- Independent evaluation by Nickolay Shmyrev / Vosk (July 2024, audiobooks,
  ~1 000 utterances, ~100 speakers):
  **Silero v4 CER 1.0 % (aidar) / 0.9 % (baya)**, UTMOS **1.76 / 2.14**;
  **Silero v3_1 CER 0.7 %**, UTMOS **2.54 / 2.98**; xRT CPU 0.054 (v4) and
  0.126 (v3_1), i.e. 18× and 8× realtime on their machine. The author's
  qualitative note: FastSpeech-2-family systems (Silero, Yandex, EdgeTTS)
  “are not very good in intonation but clarity is hard to beat”. This is the
  most useful independent Russian TTS comparison found.

So: **intelligibility is excellent (sub-1 % CER), naturalness is mid-tier.**
The high-frequency (sibilance) energy I measured differs a lot by voice —
HF energy ratio above 6 kHz was 0.21 for baya, 0.13 for kseniya, 0.12 for
xenia, 0.02 for aidar and eugene — consistent with brighter female voices,
not proof of artefacts.

**My own surrogate check (not a listening test).** This agent cannot play
audio, so I did **not** form a human listening impression. Instead I ran a
Whisper `small` round trip over all 71 clips (the 72nd was added later as an offline check) and measured F0/duration/peak.
The transcripts are intelligible and stable across voices (see
`out/silero/transcripts-whisper-small.json`); the recurring errors are the
missing digits described below, not mangled words. A human must listen to
`out/silero/` before approving a voice — the brief's subjective judgement is
still outstanding.

### 1.4 Cloning, and the two voices we need

**Fixed speakers only.** There is no speaker encoder, no reference-audio
input and no fine-tuning code in the repository. `random` picks a random
embedding from the trained set, not a new voice. The repo is inference-only:
`src/` is 28 KB of loader/utility code and the models are `torch.package`
archives; there is no `train.py`, optimiser, dataset loader or config for
training in the tree.

Best candidates:

| role | model | speaker | why |
|---|---|---|---|
| **female** | `v5_6_ru` | **`baya`** | lowest-loss choice: independent v4 CER 0.9 %, F0 239 Hz, clear; v5_6 adds question/emphasis prosody. `kseniya` is a close second (slightly brighter, 253 Hz). |
| **male** | `v5_6_ru` | **`aidar`** | independent v4 CER 1.0 %, F0 129 Hz, least sibilance; `eugene` (89 Hz) is darker/slower but had weaker ASR on the date sentence. |
| female (MIT) | `v5_cis_base_nostress` | `ru_oksana` | only if NC/SA is unacceptable; non-native accent risk, needs manual stress |
| male (MIT) | `v5_cis_base_nostress` | `ru_marat` | same caveat |

Do **not** use `xenia` as the primary female: it is the lightest/youngest and
had the only ASR number slip (`50` for `500`).

### 1.5 Numbers, dates, money — the important negative

Silero TTS has **no text normalisation for digits**. It strips every
character that is not in its symbol table, and digits are not in the table.
Verified by calling the model's own preprocessor:

```
'Сегодня встреча в 15:30.'        -> 'сегодня встреча в :.'
'Курс вырос на 2,5 процента.'     -> 'курс вырос на , процента.'
'В 2026 году.'                    -> 'в году.'
'Тест 123.'                       -> 'тест .'
```

Consequences measured over all five voices:

| input | spoken result | verdict |
|---|---|---|
| `Сегодня двадцать третье января, встреча в 15:30.` | “Сегодня двадцать третье января, встреча…” — **15:30 missing** | **fails** (word date is correct) |
| `…на 2,5 процента… 91,4 рубля.` | “Курс доллара вырос на процента и составил рубля.” — **both values missing** | **fails** |
| `В 2026 году…` | “В году…” — **year missing** | **fails** |
| `…стоит пятьсот рублей, а не пять тысяч.` | “500 рублей, а не 5000” (ASR digits; audio is correct) | **passes** |
| `Позвони мне, пожалуйста…` | exact | passes |
| homograph sentence | words exact; stress variant needs human ears | inconclusive |

Every failure is the same root cause, not voice-specific. This is exactly why
F3's normaliser is non-negotiable; it also means a Silero-generated corpus
must be fed **fully verbalised text**, and the manifest should store that
verbalised form.

### 1.6 Speed on this machine

Model loaded once, then the same 3.4–3.7 s sentence synthesised repeatedly;
median of 7 warm calls. `torch.set_num_threads` set explicitly.

| model | params | load | first call | warm median | ×realtime | peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `v4_ru` | 8.89 M | 0.17 s | 0.45 s | **15.7 ms** (default) / 15.8 ms (4 thr) / 24 ms (1 thr) | **322×** / 319× / 211× | **648 MB** |
| `v5_6_ru` | 20.9 M + stress | 1.00 s | 0.21 s | **75.7 ms** (default) / 74.6 ms (4 thr) / **38.2 ms (1 thr)** | 69× / 70× / **136×** | **826 MB** |
| `v5_cis_base_nostress` | 20.93 M | ~1 s (not isolated) | — | 41.5 ms (1 thr) / 78.9 ms (4 thr) | 141× / 74× | ~0.7 GB |

Two counter-intuitive but reproducible results on M5 Pro:

- **v5 is faster on one thread than on four** (136× vs 69×). The model is
  small enough that thread synchronisation and P/E-core scheduling dominate.
  For bulk generation, run v5 with `torch.set_num_threads(1)`.
- v4 scales normally (211× → 322×).

The project's tiny-voca model is ~30 ms/sentence on CPU. Silero v5 is the same
order (38 ms on one thread); v4 is 2.5× faster. **Silero is not a speed
problem — it is a size and licence problem.**

**MPS:** `v4_ru` moves to MPS and works (first call 1.51 s, warm median
**9 ms**). `v5_6_ru` **fails** with
`RuntimeError: Passed CPU tensor to MPS op` inside `torch.nn.functional.embedding`,
surfaced by the package as
`Exception: Model couldn't generate your text, probably it's too long`. Do not
plan v5 on MPS without patching the package.

### 1.7 Offline and caching

- Direct loading with `torch.package.PackageImporter(path).load_pickle(
  "tts_models", "model")` is **fully offline** once the `.pt` files exist.
  Verified by running a synthesis with `HTTP(S)_PROXY`/`ALL_PROXY` pointed at
  a dead port; it produced 1.61 s of audio.
- Our files live in `<repo>/.cache/models/silero/` (`v3_1_ru.pt`,
  `v4_ru.pt`, `v5_6_ru.pt`, `v5_cis_base_nostress.pt`).
- The `pip install silero` wrapper is **not** offline-clean on first use: it
  downloads `latest_silero_models.yml` into the current working directory and
  caches model files under `site-packages/silero/model/`. That would duplicate
  ~270 MB. Use the direct `PackageImporter` API as the experiment does.

---

## 2. `silero-stress` (MIT) as the stress component

### 2.1 What it actually is

Not a dictionary: a **hybrid neural package** (`accentor.pt`, 56.0 MB) with

| part | size | role |
|---|---:|---|
| `AccentorNgram` | 2.06 M params (8.2 MB fp32) | character-level stress prediction for any word, plus **19 660 exception forms** |
| `HomoSolver` | **29.19 M params** BERT-style classifier (embedding table 83 830 tokens), **2 208 homographs**, 181 `ё`-homographs, 1 477 regex/phrase rules | context disambiguation of the homograph list |

Runtime deps: **PyTorch** and the Python standard library only (README:
“depends only on PyTorch as a neural engine and the Python standard library”).
CPU, single thread. Measured on M5: **load 0.95 s**, **607 MB peak RSS**;
0.5–2.9 ms per ordinary sentence, 8–21 ms on first call / sentences with
numbers. Licence is **MIT** (`LICENSE`, `CITATION.cff`), so vendoring the
model and code into a Rust project is permitted with the copyright notice.

Accuracy claims (README): “covers ~4M known Russian words and word forms with
100% accuracy … ~2.2K homographs with F1 score of 0.92, per-word accuracy of
94 % and total dataset accuracy of 94 %”. The independent-of-vendor version is
Silero's own wiki benchmark (v1.5, 2 027 manually labelled homographs):
**F1 0.92, word_acc 0.94, total_acc 0.94**. Same-vendor, but the methodology
and baseline are published.

**Own test.** On the brief's homograph sentence the repo's `accentor.pt`
returned:

```
Зам+ок н+а двер+и б+ыл сл+оман, н+о зам+ок в гор+ах сто+ял н+а холм+е.
```

That is the **lock** stress on “замок в горах” (should be `з+амок`, castle):
one miss in the one sentence designed to expose it. The accentor embedded in
`v5_6_ru` returned the correct `з+амок` for the same sentence — the shipped
copy and the TTS-embedded copy are not identical. Treat single-sentence
accuracy claims with suspicion; the 0.92 F1 is a corpus average.

### 2.2 Versus `ruaccent` (MIT)

| | `silero-stress` | `ruaccent` (R2 survey, `docs/datasets-ru.md`) |
|---|---|---|
| form | neural accentor + neural homograph BERT | **3 194 879-form stressed dictionary** + omograph dictionary/model |
| size | 56 MB `.pt`, ~31 M params, 607 MB RAM | **21 MB gz** dictionary, 19 741 omographs; ~33 MB with tiny omograph model + tokenizer |
| runtime | PyTorch (or a custom port) | Python/ONNX; dictionary is plain data |
| homographs | 2 208, BERT context model, F1 0.92 (vendor bench) | 19 741, ONNX model; F1 0.56–0.64 in Silero's benchmark (vendor-competitive caveat) |
| OOV | neural, claimed 60–70 % | neural tiny model / rules |
| static table? | **No** — the “4M words” are generalisation, not stored entries | **Yes** — dictionary can be compiled to a Rust `phf`/fst/DAWG |
| licence | MIT | MIT |

Silero's wiki also publishes a head-to-head (its own test set): silero-stress
v1.5 F1 **0.92** vs RUAccent-turbo3.1 **0.64**, RUAccent-tiny2.1 **0.56**,
omogre 0.46. This is a vendor benchmark that favours Silero's phrase rules, but
it is the only same-set comparison available.

**Verdict:** `ruaccent` is the better base for us. A 3.19 M-form dictionary is
a deterministic, inspectable, compressible artefact that ports to Rust without
a neural runtime, and its 19.7 k omograph list is an order of magnitude larger.
`silero-stress` is useful as a **second opinion / oracle** in Python when
building the trainer's stress labels, and its homograph model could be
exported to ONNX if we ever want a neural fallback — but it is not the runtime
component to bet the Rust frontend on.

---

## 3. Silero TTS as a runtime fallback in Rust

**Premise check: the Russian models do not ship ONNX.** I probed
`models.silero.ai` directly: `v4_ru.onnx`, `v5_ru.onnx`,
`v5_cis_base.onnx`, `v5_cis_base_nostress.onnx`, `v3_1_ru.onnx` → **HTTP 404**.
`models.yml` lists ONNX only for en/de/es/ua. The Russian artefacts are
`torch.package` TorchScript archives whose `forward()` takes **Python lists of
strings** (`sentences`, `clean_sentences`, `break_lens`, …) and performs
tokenisation, accentuation and SSML parsing inside the scripted graph. There
is no tensor-only entry point in the public package.

What a Rust path would actually require:

1. Re-export just the acoustic/source-filter submodule to ONNX (the
   string-processing graph cannot be exported as-is), and
2. Re-implement in Rust: symbol mapping, the 31 M-parameter Silero stress
   model (or supply pre-stressed text), SSML parsing, speaker embedding
   lookup, and the post-filter/denormalisation, and
3. Ship a ~40–92 MB ONNX plus a stress component, versus 19 MB for tiny-voca.

Evidence that this is non-trivial: even on Apple's own MPS backend, `v5_6_ru`
fails with `Passed CPU tensor to MPS op`; the package is tuned for
CPU/CUDA TorchScript, not for a neutral graph runtime.

What we would gain/lose versus the 4.5 M tiny-voca model:

| | Silero `v4_ru` | Silero `v5_6_ru` | tiny-voca (planned ru) |
|---|---|---|---|
| params | 8.89 M | 20.9 M + 31 M stress | **4.54 M** |
| on-disk | 40 MB | 148 MB | **19 MB** |
| inference | 15.7 ms/sentence (322× RT) | 38 ms/sentence (136× RT) | ~30 ms/sentence |
| quality | CER 0.9–1.0 %, mid UTMOS | vendor UTMOS 3.04 ≈ real speech | our target, unknown |
| voices | 5 fixed | 5 fixed + questions/emphasis | 2 fixed (ours) |
| licence | NC-SA | NC-SA | our choice (MIT/Apache) |
| Rust integration | none today | none today | native |

So the trade is **“works today in Python, 2–8× larger, NC-SA, no Rust path”**
against **“native, 19 MB, licence-clean, not trained yet”**. If the training
project fails entirely, shipping Silero behind the *same Rust CLI* is not a
small fallback — it is a second inference engine plus a frontend rewrite. A
Python sidecar would work but violates the project's “no runtime dependency”
property. **Not a plausible fallback without a dedicated port.**

If a fallback is still wanted, the cheapest honest one is: keep the Rust CLI
for tiny-voca, and document “install Silero separately” rather than embedding
it. The MIT `v5_cis_base_nostress` (91.7 MB) is the only version that could be
redistributed without NC/SA.

---

## 4. Silero STT / other Silero work

### 4.1 STT Russian: not available

Current `models.yml` `stt_models` contains **`en`, `de`, `es`, `ua` only** —
there is **no `ru` entry**, and the README documentation confirms the same
three-language STT list. Probes of the old paths
(`models/stt/v3_1_ru.pt`, `stt/v4_ru.pt`, …) return 404. Silero's Russian STT
was withdrawn from the public CE distribution; only old blog posts and
third-party mirrors remain.

Licence for what remains is the main repo licence (CC BY-NC-SA 4.0); only the
CIS TTS base models are MIT. Even if a mirror is found, it is **NC-SA and
unsupported**, so it cannot be the screening/eval judge we ship.

**Comparison for Russian read speech (published):**

| model | Russian read/audiobook WER | Russian average | licence |
|---|---:|---:|---|
| GigaAM v3 RNNT (240 M) | **4.4 %** RuLS; 8.2 % OpenSTT Audiobooks | **8.3 %** across 10 ru sets | MIT |
| GigaAM v3 CTC | 4.7 % RuLS; 8.7 % Audiobooks | 9.1 % | MIT |
| Whisper large-v3 | 9.4 % RuLS; 14.3 % Audiobooks | 21.0 % | MIT |
| Silero STT ru | — (no current model) | — | NC-SA (withdrawn) |
| Whisper large-v3-turbo (project measurement, R2) | 4.44 % FLEURS-ru | — | MIT |

GigaAM's official evaluation is unkind to Whisper on spontaneous/telephone
Russian, but on clean read speech the project's own FLEURS number puts
large-v3-turbo on par with GigaAM. Either is a better judge than any Silero
STT we can obtain. **Recommendation: keep the F5 plan (Whisper large-v3-turbo)
and add GigaAM as cross-check, as R2 already proposed; drop Silero STT from
consideration.**

### 4.2 VAD (MIT) — useful

`snakers4/silero-vad` is MIT. It ships **1.2 MB** 16 kHz ONNX models
(`silero_vad.onnx`, `silero_vad_16k_op15.onnx`, sequence variants) plus a
2.2 MB TorchScript. This is an excellent fit for trimming leading/trailing
silence in generated clips and for splitting long teacher output; it runs in
Rust via `ort` with a tiny tensor interface. **Use it.** Note it is 16 kHz, so
resample 24 kHz generated audio down for VAD and cut the original 24 kHz
buffer with the returned timestamps.

### 4.3 Text enhancement / punctuation (NC-SA)

`te_models/v2_4lang_q.pt` (91.7 MB) supports `ru`, `en`, `de`, `es`, restoring
punctuation `.,-!?—` and capitalisation. Published Russian sentence-level
quality: validation **WER_p/WER_c 7/7** (baseline 13/12); books 17/7
(baseline 19/9); punctuation F1 roughly 0.67–0.81 for `, - ! ?`. It is under
the main CC BY-NC-SA 4.0 licence, so it is usable for research-side text
preparation but **not shippable** in an Apache/MIT binary. Our F3 normaliser
should not depend on it.

### 4.4 SSML

`v4`/`v5` support `<speak>`, `<break time/strength>`, `<prosody
rate/pitch>`, `<p>`, `<s>`; `v5_6_ru` additionally does questions and
emphasis. Verified: an SSML sample with `<prosody rate="slow">` and a 1 s
break produced the expected longer clip (`out/silero/v5_6_ru_baya_ssml.wav`).
SSML is irrelevant to our Rust runtime unless we adopt Silero itself.

---

## 5. The experiment (reproducible)

Environment (all inside the repo, per `AGENTS.md`):

```sh
cd /Users/suheugene/projects/22-tiniest-voca-ru
. scripts/cache-env.sh
uv venv --python 3.12 .cache/venvs/r6
uv pip install --python .cache/venvs/r6/bin/python silero omegaconf soundfile faster-whisper

# models (already fetched; 40–148 MB each)
# .cache/models/silero/{v3_1_ru,v4_ru,v5_6_ru,v5_cis_base_nostress}.pt
```

Core loop (the exact code used):

```python
import torch, soundfile as sf
torch.set_grad_enabled(False)
torch.set_num_threads(1)          # v5 is faster on one thread
m = torch.package.PackageImporter(
    ".cache/models/silero/v5_6_ru.pt").load_pickle("tts_models", "model")
a = m.apply_tts(text="Сегодня двадцать третье января.",
                speaker="baya", sample_rate=24000)   # 24 kHz mono
sf.write("out/silero/test.wav", a.numpy(), 24000)
```

- **Voices synthesised:** 5 × `v5_6_ru` (baya, kseniya, xenia, aidar,
  eugene), 3 × `v4_ru` (baya, kseniya, aidar), 2 ×
  `v5_cis_base_nostress` (ru_oksana, ru_marat), plus SSML, question and
  emphasis samples — **72 clips, 24 kHz WAV** (ASR ran on the first 71, before the offline check).
- **Sentences:** the six required, plus a stressed-homograph variant and an
  automatic-stress variant (`*_03_homograph` vs `*_03b_homograph_auto`).
- **ASR:** faster-whisper `small`, CPU int8, language `ru`; transcripts in
  `out/silero/transcripts-whisper-small.json`. (Small is a screen, not a
  benchmark; a few word substitutions such as “встречу/встречов” are ASR
  errors, not TTS errors.)
- **Timing:** median of 7 warm calls after one warm-up, M5 Pro; the machine
  was shared with other agents, so absolute numbers are ±20 %, the v4 vs v5
  and 1-thread vs 4-thread ratios are stable.

---

## 6. Verdict

| Use case | Silero fit | Why |
|---|---|---|
| **Primary teacher instead of Qwen3-TTS** | **partial** | Works today on CPU, native-sounding Russian and 3.04 UTMOS, but NC-SA contamination, fixed voices, mid-tier intonation, and no digit normalisation. Qwen3-TTS remains the better teacher; Silero is the safety net. |
| **Second teacher / extra voice source** | **yes** | 5 fixed Russian voices, 24 kHz, 69–322× realtime on CPU, deterministic and tiny to generate. Ideal for a 200–2 000-clip pilot and for data diversity if Qwen is unavailable. |
| **Voice cloning / new voices** | **no** | No speaker encoder, no reference input, no training/fine-tuning code. Only the fixed speaker list. |
| **`silero-stress` as the shipped stress layer** | **no** | MIT, accurate, but 56 MB and 31 M params of PyTorch; not a static dictionary. `ruaccent` (3.19 M forms, 21 MB, static table) is the better Rust base. |
| **`silero-stress` as a Python oracle** | **partial** | Useful cross-check for the trainer's stress labels; the repo copy and the TTS-embedded copy disagree on `замок`, so never trust a single sentence. |
| **Runtime fallback behind the Rust CLI** | **no** (partial only with a dedicated ONNX export + frontend port) | No official Russian ONNX; TorchScript graph takes strings; v5 fails on MPS; 40–148 MB vs 19 MB. |
| **STT / screening judge** | **no** | No Russian STT in the current catalogue; withdrawn model would be NC-SA and unsupported. Whisper large-v3-turbo + GigaAM are better and permissive. |
| **VAD for trimming generated audio** | **yes** | MIT, 1.2 MB ONNX, 16 kHz, trivial Rust integration. |
| **Punctuation/capitalisation** | **partial** | Ru-supported and good, but 91.7 MB and NC-SA; do not put it in the shipped path. |

### Recommendation

1. **Keep Qwen3-TTS as primary teacher.** Silero does not beat a 1.7 B
   VoiceDesign+Base teacher on naturalness or licence cleanliness.
2. **Add Silero `v5_6_ru`, voices `baya` (F) and `aidar` (M), as a second
   teacher and immediate CPU-only data source.** Generate a small pilot
   (200–500 sentences, fully verbalised text) and listen next to Qwen before
   any full corpus. This de-risks the GPU dependency: 19 h of audio is
   ~15 min at 136× on one thread.
3. **Record the licence.** Any Silero-derived corpus/checkpoint is
   **CC BY-NC-SA 4.0** with attribution to Silero. If the project ever needs a
   permissive release, generate that portion from the MIT
   `v5_cis_base_nostress` (`ru_oksana`/`ru_marat`) or Qwen, not from
   `v5_6_ru`.
4. **Use `ruaccent` for the stress layer; keep `silero-stress` as an offline
   oracle** in the Python label pipeline only.
5. **Use Silero VAD** to trim generated clips.
6. **Do not plan Silero as the Rust runtime fallback.** If a fallback is
   required, treat it as a separate, explicitly-documented Python dependency,
   not part of the single-binary story.
7. **Do not pursue Silero STT.** Stick to Whisper large-v3-turbo + GigaAM as
   R2 proposed.

---

## 7. Samples and data

All under `/Users/suheugene/projects/22-tiniest-voca-ru/out/silero/`
(git-ignored, 24 kHz mono WAV unless noted).

**Main comparison — same six sentences, five voices (`v5_6_ru`):**

```
v5_6_ru_baya_01_date.wav        v5_6_ru_aidar_01_date.wav
v5_6_ru_kseniya_01_date.wav     v5_6_ru_eugene_01_date.wav
v5_6_ru_xenia_01_date.wav
… _02_money / _03_homograph / _03b_homograph_auto / _04_request /
  _05_numbers / _06_year   (same stem per voice)
```

**Smaller/older model (`v4_ru`, 8.9 M):** `v4_ru_{baya,kseniya,aidar}_*.wav`.

**MIT model, manually stressed (`v5_cis_base_nostress`):**
`v5_cis_base_nostress_ru_{oksana,marat}_*.wav`.

**Prosody:** `v5_6_ru_baya_ssml.wav`, `v5_6_ru_baya_question.wav`,
`v5_6_ru_baya_emphasis.wav`.

**Offline proof:** `offline_check_v4_baya.wav`.

**Machine-readable:** `transcripts-whisper-small.json` (ASR of all clips),
`stress-silero-stress.json` (accentor output/timings),
`measure-threads-v1.json` (first thread sweep, noisier than §1.6).

Listening order for a human: compare `baya` vs `aidar` on `_01_date` and
`_02_money`, then the auto vs explicit stress pair `_03` / `_03b`, then the
same voice in `v4_ru` vs `v5_6_ru`.
