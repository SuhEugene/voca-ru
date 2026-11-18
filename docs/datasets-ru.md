# Russian speech and text data for a small TTS model

Survey for `sultson/voca` (4.5M-param Rust TTS), target: small single-/few-speaker
Russian model at ~22.05 kHz, plus a small ASR set for round-trip WER.

**Verified:** 2026-11-19. Every URL below was fetched live on that date; HTTP
status is in the verification log at the end. Hours/speakers are from the
publisher (or from MFA's corpus metadata where the publisher is vague). Do not
treat remembered numbers as verified — the numbers here are the ones that were
on the live pages.

Legend for licence: **NC** = non-commercial, **SA** = share-alike,
**BY** = attribution. "Commercial?" answers whether a commercial product may
ship the *data or a model trained on it*; share-alike is called out separately
because model-weight obligations are legally unsettled.

---

## 1. TTS corpora

| name | modality | hours | speakers | sample rate | transcript quality | licence (commercial? attribution? share-alike?) | access method + URL | expected disk size | quality notes | recommended weight in the mix |
|---|---|---|---|---|---|---|---|---|---|---|
| **RUSLAN** | TTS | **31.53 h** (31:32:05, 22,200 utts) | **1** (male) | **44.1 kHz, 16-bit PCM** | Manual, verified; sentence case + punctuation retained; numbers/dates/acronyms expanded | **CC BY-NC-SA 4.0 — NC**, BY, SA. Official: “Corpus is available here (7 Gb) under the CC BY-NC-SA 4.0 license.” | Official page <https://ruslan-corpus.github.io/>; download folder <https://drive.google.com/drive/folders/1QjaIKtPHmj-baiUMjjQqe8XjZ5XpiNoC> (browser). Scriptable HF mirror: <https://huggingface.co/datasets/akuzdeuov/ruslan-tts> (22.05 kHz parquet; **no licence on card**) | **7 GB** official; **9.83 GB** HF mirror; ~5.0 GB after 22.05 kHz/16-bit conversion | Studio, quiet room, SNR ≈ 90 dB, one speaker, per-utterance files. Best Russian single-speaker TTS corpus. Licence is the blocker, not quality. | **1.0** for first model **if NC is approved**; otherwise 0 |
| **Russian LibriSpeech (RuLS)** | TTS / ASR | **98.24 h** (57,224 utts) | **13** (5 F, 8 M) | **16 kHz**, mono, 16-bit (verified from WAV header) | `text_no_preprocessing` = case + punctuation (e.g. “Для вас, души моей царицы, Красавицы…”); `text` = lowercased, no punctuation. Forced-aligned from Gutenberg text. | **Public Domain in the USA** (OpenSLR SLR96). US-only wording; LibriVox PD status may not hold in every jurisdiction. | <https://www.openslr.org/96/>; tar <https://www.openslr.org/resources/96/ruls_data.tar.gz>; HF mirror <https://huggingface.co/datasets/istupakov/russian_librispeech> | **9.13 GB** tar; test parquet 0.30 GB | LibriVox audiobooks, already segmented LibriSpeech-style. Recording quality varies by reader; filter one reader (13 readers / 98 h ≈ 7.5 h average). Punctuation/case available. | **1.0** (one reader) for licence-clean first model; **0.25** in larger mix |
| **M-AILABS Russian** | TTS | **46.78 h** (20,494 utts) | **3** (1 F, 2 M) | **16 kHz** (sample WAV header verified 16 kHz) | Gutenberg book text; case + punctuation expected (verify on unpack); alignment from LibriVox | **Ambiguous.** 2020 repo licence: “Redistribution and use in any form, **including any commercial use**, with or without modification are permitted” (BY, no-endorsement). An HF mirror labels it **CC BY-NC-ND 4.0**. Original `caito.de` page is dead. | <https://github.com/i-celeste-aurora/m-ailabs-dataset>; direct archive <https://ics.tau-ceti.space/data/Training/stt_tts/ru_RU.tgz> (3.87 GB, HTTP 200) | **3.87 GB** download; ~5.1 GiB unpacked | Audiobook read speech, 3 speakers; split per speaker ≈ 15 h each, enough for a small model. Licence provenance conflict must be resolved by a human. | **0.15** larger mix **only if licence is cleared**; 0 otherwise |
| **SOVA RuAudiobooksDevices** | TTS / ASR | **298 h** | many (audiobook readers; count not published) | **16 kHz**, mono WAV | Manual/forced-alignment; transcripts are **lowercased, no punctuation** (verified samples) | **CC BY 4.0** — commercial yes, BY. Repo: “SOVA Dataset is licensed under Creative Commons BY 4.0 license by Virtual Assistant, LLC.” | HF mirror <https://huggingface.co/datasets/dangrebenkin/sova_rudevices_audiobooks>; source table <https://github.com/sovaai/sova-dataset> | **29.7 GB** (parquet) | Volunteer audiobook recordings, “unprofessional equipment”; 16 kHz. Good for ASR / multi-speaker pretraining. Per-book speaker identity is plausible but not guaranteed. | **0.10** larger mix (pretrain); not first model |
| **Golos (crowd + farfield)** | ASR / TTS pretrain | **1,240 h** total (1,095 crowd + 132 farfield train; 12.6 h test) | many (crowd; exact count not published) | **16 kHz**, Opus/WAV | Manual ASR annotation; **lowercased, no punctuation**, numbers spelled out (verified sample) | **Custom “Public license with attribution and conditions reserved”** — a BY-SA-4.0 rework, explicitly “not a Creative Commons license”. Commercial yes, BY + **SA**. | <https://www.openslr.org/114/>; opus tar 18.47 GB; repo <https://github.com/sberdevices/golos>; licence PDF <https://github.com/sberdevices/golos/blob/master/license/en_us.pdf> | **18.5 GB** full Opus; HF 10 h subset card claims 16 GB (inflated) | Crowd read speech + far-field smart-speaker audio; variable noise. ASR-oriented transcripts; not studio TTS. Share-alike ambiguity for weights. | **0.10** larger mix (acoustic pretrain); 0 first model |
| **Common Voice ru (CV21 mirror)** | ASR / TTS pretrain | **220.45 h validated** (132.25 train + 55.21 dev + 32.99 test), `down_votes=0` | many | **32 kHz MP3, 64 kbps** (frame header verified) | Crowd prompts: **case + punctuation** (verified: “Никогда со мной этого не бывало.”), but reading errors/noise remain | **CC0 1.0** (mirror card). Official release is now via Mozilla Data Collective (login). | Mirror <https://huggingface.co/datasets/Sh1man/common_voice_21_ru>; official info <https://commonvoice.mozilla.org/en/datasets> | **11.7 GB** | Crowd recordings, accents/noise/mic variety. Good for ASR eval and speaker diversity; poor for clean single-speaker TTS. | **0.05** optional larger mix; 0 first model |
| **RUSLAN-stressed (stilletto)** | TTS | 31.5 h | 1 | 22.05 kHz | RUSLAN text with stress marks added by an LLM + validation | Card claims **CC BY 4.0**, but audio is a derivative of **CC BY-NC-SA 4.0** RUSLAN. **The relicense is not valid.** | <https://huggingface.co/datasets/stilletto/ruslan-stressed> | **~9.5 GB** | Convenient stress-marked transcript, but do not rely on its licence. Use ruaccent (MIT) to add stress to the official RUSLAN text instead. | 0 (licence-risk); use only as a cross-check |

### TTS notes

- **RUSLAN is the only true studio single-speaker Russian TTS corpus found.**
  31.5 h at 44.1 kHz clears the “5–50 h” bar for a 4.5M-param model with room
  to spare. It is NC-SA, so the orchestrator must ask the human before use.
- **RuLS is the best licence-clean substitute**: public domain in the USA,
  98 h / 13 readers, and `text_no_preprocessing` already has case and
  punctuation. Pick the reader with the most utterances; 7–15 h is the
  expected range, which is still enough.
- **M-AILABS** is the best “few-speaker, ~15 h per speaker” option, but its
  licence is contradictory (2020 permissive relicense vs CC BY-NC-ND on
  mirrors). Do not ship it without a human decision.
- **Golos / SOVA / Common Voice** are ASR-style corpora: lowercase,
  unpunctuated transcripts and variable acoustics. Useful for pretraining or
  ASR, not for a clean single-speaker TTS front end.
- The RUSLAN derivatives (`akuzdeuov/ruslan-tts`, `stilletto/ruslan-stressed`,
  `intexcp/ruslan_sova_ai`, `Diana2di/ruslan_data`) are **not** independently
  licensed. Treat them as RUSLAN (NC-SA) or avoid.

---

## 2. ASR / round-trip evaluation sets

Small eval sets only; a WER judge does not need more than a few hours.

| name | modality | hours | speakers | sample rate | transcript quality | licence (commercial? attribution? share-alike?) | access method + URL | expected disk size | quality notes | recommended weight in the mix |
|---|---|---|---|---|---|---|---|---|---|---|
| **FLEURS ru** | ASR (eval) | **11.63 h** total; **2.50 h / 775 utts test** (8.05 train, 1.08 dev) | ≈3 per split, disjoint train/test | **16 kHz** | Human translations/transcripts; `raw_transcription` has case + punctuation, normalized `transcription` lowercased | **CC BY 4.0** — commercial yes, BY | <https://huggingface.co/datasets/google/fleurs>; TSVs <https://huggingface.co/datasets/google/fleurs/tree/main/data/ru_ru> | **2.64 GB** full ru config; ~0.2–0.5 GB for test audio | Read Wikipedia sentences, clean-ish, disjoint test speakers. Best primary WER judge. | **eval only** (primary) |
| **RuLS test** | ASR (eval) | **~2.3 h** (1,352 utts) | 13 total; test subset | 16 kHz | `text_no_preprocessing` case + punctuation | **Public Domain in the USA** | <https://huggingface.co/datasets/istupakov/russian_librispeech> (test parquet 0.30 GB) | **0.30 GB** | Audiobook read speech; same domain as training if RuLS is used, so keep it out of training. | **eval only** (secondary / in-domain check) |
| **SOVA RuDevices** | ASR (eval) | **101 h** (test ~6 h) | many | **16 kHz** | Manual; lowercased, no punctuation | **CC BY 4.0** | <https://huggingface.co/datasets/bond005/sova_rudevices>; source <https://github.com/sovaai/sova-dataset> | **6 h test** ≈ 0.6 GB; full 10.4 GB | Live speech, noisy; useful as a hard/out-of-domain WER check. | **eval only** (optional hard set) |
| **Golos test** | ASR (eval) | **12.6 h** (11.2 crowd + 1.4 farfield) | many | 16 kHz | Manual; lowercased, no punctuation | Custom BY-SA-like (see Golos row) | <https://www.openslr.org/114/> (`test.tar`, 1.3 GB) | **1.3 GB** | Crowd + far-field; good second-domain check. Share-alike applies. | **eval only** (optional) |
| **Common Voice ru test** | ASR (eval) | **32.99 h** validated | many | 32 kHz MP3 | Case + punctuation prompts | **CC0 1.0** | <https://huggingface.co/datasets/Sh1man/common_voice_21_ru> | included in 11.7 GB mirror | Huge for an eval; sample 1–2 h. Accent/noise diversity. | **eval only** (optional) |
| **OpenSTT** | ASR (train, optional) | **~20,000 h** | many | mostly 16 kHz | Mixed manual/ASR; noisy | **CC BY-NC 4.0 — NC** | <https://github.com/snakers4/open_stt> | **356 GB** Opus; 2.3 TB WAV | Only if a large ASR pretrain is ever wanted. Too big and NC for this project. | **0** |

**Eval recommendation:** FLEURS ru test (775 utts, 2.5 h) as the primary
round-trip judge; add RuLS test only if RuLS was *not* used for training.
FLEURS is small enough to run a Whisper-class judge in minutes and has clean
punctuated references.

---

## 3. Text, stress, and G2P resources

Russian TTS quality is dominated by **lexical stress**, not raw text volume.
The good news: the best accentuation resource is MIT-licensed.

| name | modality | hours | speakers | sample rate | transcript quality | licence (commercial? attribution? share-alike?) | access method + URL | expected disk size | quality notes | recommended weight in the mix |
|---|---|---|---|---|---|---|---|---|---|---|
| **RUAccent dictionary + models** (`ruaccent`) | text (stress) | n/a | n/a | n/a | **3,194,879 stressed word forms** + 19,741 omograph entries (counted from downloaded `accents.json.gz`) | **MIT** (repo `LICENSE`, HF model card `license: mit`) | <https://github.com/Den4ikAI/ruaccent>; files <https://huggingface.co/ruaccent/accentuator>; dictionary `dictionary/accents.json.gz` (21.0 MB), `omographs.json.gz` (1.5 MB), small ONNX model ~0.8 MB | **~21 MB** compressed dictionary + ~0.8 MB model; 175.5 MB raw JSON | Best practical stress resource: dictionary covers ~3.2M forms; omograph model handles context (`за́мок`/`замо́к`). Use dictionary for lexicon, model for OOV/context. | **1.0** (stress layer) |
| **RUAccent dictionaries (LeeAeron mirror)** | text (stress) | n/a | n/a | n/a | Same family of RUAccent dictionaries in `.bin`/`.json.gz` | **Apache-2.0** (HF card) | <https://huggingface.co/datasets/LeeAeron/RUAccent_dictionaries> | **0.85 GB** | Redundant with `ruaccent`; use only if the Apache-2.0 packaging is preferred over MIT. | 0 (alternative) |
| **Russian Wiktionary via kaikki** | text (stress / lexicon) | n/a | n/a | n/a | Headwords/forms; stress marks where Wiktionary records them; morphology | **CC BY-SA 4.0 + GFDL** — commercial yes, BY + **SA**. Wiktionary: “dual-licensed … CC-BY-SA … and … GFDL.” | <https://kaikki.org/dictionary/Russian/>; JSONL <https://kaikki.org/dictionary/Russian/kaikki.org-dictionary-Russian.jsonl> | **904.6 MB** | Good supplement for OOVs and morphology; share-alike affects derived dictionaries. Not needed if ruaccent covers the vocabulary. | **0.1** (supplement) |
| **Russian Wikipedia dump** | text (LM / context) | n/a | n/a | n/a | Article text, case + punctuation | **CC BY-SA 4.0 + GFDL** — commercial yes, BY + **SA** | <https://dumps.wikimedia.org/ruwiki/latest/>; `ruwiki-latest-pages-articles.xml.bz2` = 6,044,355,020 bytes | **6.04 GB** bz2 | Only useful for an LM or for context sentences around stressed words. Not needed for the first model. | **0.1** (optional LM/context) |
| **russian_g2p** (Accentor + Transcriptor) | text (G2P + stress) | n/a | n/a | n/a | Ships `Accents.json` 54.7 MB, `All_Forms_UTF8.txt` 64.5 MB, homographs, DAWG; returns phonemes with stress (`D0 I A L O0 K`) | **MIT** | <https://github.com/nsu-ai/russian_g2p> | **~34 MB** git repo; ~130 MB unpacked data | Best open Russian G2P + accentor pair; old (2018) but self-contained. Use it to build the phoneme lexicon from ruaccent-stressed text. | **1.0** (G2P layer) |
| **espeak-ng Russian** | text (G2P) | n/a | n/a | n/a | Rule-based phonemes; stress by rule | **GPL-3.0** — copyleft code; phoneme output is generally not a derivative, but shipping the binary in an Apache-2.0 product is a concern | <https://github.com/espeak-ng/espeak-ng> | **~76 MB** source | Quick baseline; no lexical homograph disambiguation. Avoid if the shipped binary must stay Apache-2.0. | 0 (baseline only) |
| **Epitran** | text (G2P) | n/a | n/a | n/a | Rule-based IPA; no lexical stress | **MIT** | <https://github.com/dmort27/epitran> | **~3 MB** repo | Simple fallback; Russian mapping is basic. | 0 (fallback) |
| **RUPhon** | text (G2P) | n/a | n/a | n/a | IPA with stress, based on RUAccent encoder; 14M/28M models | **No licence declared** (GitHub API: `NOASSERTION`) — **unclear** | <https://github.com/Den4ikAI/ruphon> | **~30 KB** code + runtime model download | Modern and convenient, but cannot be used until a licence is added. | 0 (blocked) |
| **Silero Stress** | text (stress) | n/a | n/a | n/a | Neural accentor + homograph disambiguation | **MIT** (repo badge + `LICENSE`) | <https://github.com/snakers4/silero-stress> | **~71 MB** repo + model download | Alternative to ruaccent; MIT. Larger runtime than ruaccent's tiny model. | 0 (alternative) |
| **russtress** | text (stress) | n/a | n/a | n/a | Python accentor | **No licence declared** (GitHub API: `license: null`) — **unclear** | <https://github.com/MashaPo/russtress> | **~0.7 MB** repo | Do not use without clarification. | 0 (blocked) |
| **Omogre** | text (stress + IPA) | n/a | n/a | n/a | Accentuator + IPA transcriber | **NOASSERTION** — **unclear** | <https://github.com/omogr/omogre> | **~71 MB** repo + model download | Do not use without clarification. | 0 (blocked) |

### Stress / G2P recommendation

Use **ruaccent (MIT)** as the stress layer and **russian_g2p (MIT)** as the
phoneme layer. Both are permissively licensed and self-contained; together they
cover the training-text vocabulary and produce a stress-marked phoneme lexicon
without touching NC data. `ruaccent`'s 3.19M-form dictionary is more valuable
for Russian TTS quality than adding another 50 h of unaccented audio.

Practical pipeline:

1. Normalise text (case, punctuation, `ё`).
2. Add stress with `ruaccent` dictionary + omograph model.
3. Convert stressed text to phonemes with `russian_g2p`.
4. Train the acoustic model on phoneme or character+stress tokens.

Do **not** train a stress predictor from scratch unless the off-the-shelf
dictionary shows a coverage gap on the target corpus; 3.19M forms is already
larger than the vocabulary of any 31 h corpus.

---

## 4. Recommended mixes

All paths are relative to the repo root; `data/` is gitignored.

### 4.1 Minimum, best quality — RUSLAN (requires NC approval)

| component | size | destination | weight |
|---|---:|---|---:|
| RUSLAN 31.5 h (HF 22.05 kHz mirror; official 44.1 kHz if using GDrive) | 9.83 GB (7 GB official) | `data/raw/ruslan/` | 1.0 acoustic |
| ruaccent dictionary + small omograph model | 0.02 GB | `data/raw/stress/ruaccent/` | 1.0 stress |
| FLEURS ru test (775 utts, 2.5 h) | ~0.2–0.5 GB | `data/raw/fleurs/` | eval only |

**Total download ≈ 10.0 GB** (≈ 7.3 GB if the official 7 GB RUSLAN archive is
used). After converting RUSLAN to 22.05 kHz/16-bit mono, the corpus is ~5.0 GB;
the original can be deleted. **This is the recommended first model if the human
approves CC BY-NC-SA 4.0.** It is the only studio single-speaker corpus and the
only one that comfortably exceeds the 5–50 h guidance.

### 4.2 Minimum, licence-clean — RuLS single reader (no NC)

| component | size | destination | weight |
|---|---:|---|---:|
| RuLS 98 h, filtered to the reader with most utterances (expect 7–15 h) | 9.13 GB | `data/raw/ruls/` | 1.0 acoustic |
| ruaccent dictionary + model | 0.02 GB | `data/raw/stress/ruaccent/` | 1.0 stress |
| FLEURS ru test | ~0.2–0.5 GB | `data/raw/fleurs/` | eval only |

**Total download ≈ 9.35 GB.** Commercial-friendly in the US (public domain in
the USA + MIT + CC BY 4.0), but audiobook acoustics and 16 kHz source; expect a
noticeably weaker voice than RUSLAN. Pick the reader by counting the first path
component of `audio_filepath` (e.g. `2671/2145/...` → reader `2671`).

### 4.3 Minimum, few-speaker — M-AILABS (only if licence cleared)

| component | size | destination | weight |
|---|---:|---|---:|
| M-AILABS Russian 46.8 h, filtered to one speaker (~15 h) | 3.87 GB | `data/raw/mailabs-ru/` | 1.0 acoustic |
| ruaccent + model | 0.02 GB | `data/raw/stress/ruaccent/` | 1.0 stress |
| FLEURS ru test | ~0.2–0.5 GB | `data/raw/fleurs/` | eval only |

**Total download ≈ 4.1 GB.** Smallest option, but the licence is contradictory
(see §1) and the source page is dead; require a human decision first.

### 4.4 Larger mix (better model; ~48.5 GB)

| component | size | destination | weight |
|---|---:|---|---:|
| RUSLAN 31.5 h | 9.83 GB | `data/raw/ruslan/` | 0.40 |
| RuLS 98 h (multi-speaker pretrain, then fine-tune one reader) | 9.13 GB | `data/raw/ruls/` | 0.25 |
| M-AILABS Russian 46.8 h (if cleared) | 3.87 GB | `data/raw/mailabs-ru/` | 0.15 |
| Golos full Opus 1,240 h (acoustic pretrain only) | 18.47 GB | `data/raw/golos/` | 0.10 |
| SOVA RuAudiobooksDevices 298 h (optional, +29.7 GB) | 29.7 GB | `data/raw/sova/audiobooks/` | 0.10 (replace Golos if used) |
| ruaccent dictionary + model | 0.02 GB | `data/raw/stress/ruaccent/` | stress |
| Russian Wiktionary (kaikki) | 0.90 GB | `data/raw/text/kaikki-ru.jsonl` | 0.10 |
| Russian Wikipedia dump | 6.04 GB | `data/raw/text/ruwiki/` | 0.10 (LM/context) |
| FLEURS ru test | ~0.2–0.5 GB | `data/raw/fleurs/` | eval only |

**Total ≈ 48.5 GB** without SOVA; ≈ 78 GB with SOVA. Strategy: pretrain a
multi-speaker model on the union (ASR-style transcripts are fine for acoustic
pretraining), then fine-tune on 10–30 h of the chosen target speaker with
punctuated, stress-marked text. The small model can also be trained directly on
RUSLAN alone; the larger mix mainly buys robustness, not peak quality.

### 4.5 Download recipes

```sh
mkdir -p data/raw/{ruslan,ruls,mailabs-ru,golos,sova,fleurs,stress,text,g2p}

# RUSLAN: scriptable 22.05 kHz mirror (treat as NC-SA; original is 44.1 kHz)
huggingface-cli download akuzdeuov/ruslan-tts --repo-type dataset \
  --local-dir data/raw/ruslan

# Or official 7 GB folder (browser; Google Drive may ask for confirmation):
# https://drive.google.com/drive/folders/1QjaIKtPHmj-baiUMjjQqe8XjZ5XpiNoC

# RuLS
curl -L -o data/raw/ruls/ruls_data.tar.gz \
  https://www.openslr.org/resources/96/ruls_data.tar.gz

# M-AILABS Russian (3.87 GB)
curl -L -o data/raw/mailabs-ru/ru_RU.tgz \
  https://ics.tau-ceti.space/data/Training/stt_tts/ru_RU.tgz

# Golos (18.5 GB; do not fetch for the first model)
curl -L -o data/raw/golos/golos_opus.tar.gz \
  https://www.openslr.org/resources/114/golos_opus.tar.gz

# SOVA audiobooks (29.7 GB)
huggingface-cli download dangrebenkin/sova_rudevices_audiobooks \
  --repo-type dataset --local-dir data/raw/sova/audiobooks

# FLEURS ru (test audio + TSVs)
huggingface-cli download google/fleurs --repo-type dataset \
  --include "data/ru_ru/*" --local-dir data/raw/fleurs

# Stress dictionary + tiny omograph model (MIT)
curl -L -o data/raw/stress/ruaccent/accents.json.gz \
  https://huggingface.co/ruaccent/accentuator/resolve/main/dictionary/accents.json.gz
curl -L -o data/raw/stress/ruaccent/omographs.json.gz \
  https://huggingface.co/ruaccent/accentuator/resolve/main/dictionary/omographs.json.gz

# Text (optional)
curl -L -o data/raw/text/ruwiki-latest-pages-articles.xml.bz2 \
  https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2
curl -L -o data/raw/text/kaikki-ru.jsonl \
  https://kaikki.org/dictionary/Russian/kaikki.org-dictionary-Russian.jsonl

# G2P code + bundled lexicon (MIT)
git clone https://github.com/nsu-ai/russian_g2p data/raw/g2p/russian_g2p
```

**Note:** `ruaccent`'s HF repo is 6.26 GB because it bundles several neural
models and dictionary variants. Fetch only `dictionary/accents.json.gz`
(21 MB), `dictionary/omographs.json.gz` (1.5 MB), and
`nn/nn_accent/model.onnx` (~0.8 MB); do not clone the whole repo.

---

## 5. Licence red flags — ask the human before using

| dataset | flag | why |
|---|---|---|
| **RUSLAN** | **NC + SA** | CC BY-NC-SA 4.0. Commercial use prohibited; model weights may inherit share-alike. The canonical Russian single-speaker corpus is unusable for a commercial model without permission. |
| **M-AILABS Russian** | **ambiguity** | 2020 repo text permits commercial use; an HF mirror says CC BY-NC-ND 4.0. Two incompatible licences on the same data. |
| **Golos** | **custom SA** | “Public license with attribution and conditions reserved” — a BY-SA rework, not CC. Commercial use is granted, but share-alike scope for model weights is unclear. |
| **OpenSTT** | **NC** | CC BY-NC 4.0. Do not use for a commercial model. |
| **RUSLAN derivatives** (`stilletto/ruslan-stressed`, `akuzdeuov/ruslan-tts`, `intexcp/ruslan_sova_ai`, `Diana2di/ruslan_data`) | **invalid/unclear licensing** | They relicense NC-SA audio as CC BY 4.0/MIT/none. Do not trust their cards. |
| **RuLS** | **jurisdiction** | “Public Domain in the USA”; LibriVox recordings may not be PD everywhere. Fine for a US-based project; ask if the product ships globally. |
| **RUPhon, Omogre, russtress** | **no licence** | GitHub API returns `NOASSERTION`/`null`. Unusable until clarified. |
| **espeak-ng** | **GPL-3.0** | Copyleft code; fine for local experiments, risky to embed in an Apache-2.0 binary. |
| **Taiga corpus** | **unavailable** | `TatianaShavrina/taiga_social_corpus` is 404; `taiga_site` has no licence and no data. Not used. |
| **Common Voice official release** | **access blocked** | Since Oct 2025 bulk data is behind a Mozilla Data Collective login. Use the CC0 mirror or record the blocker. |

Non-issues (commercial-friendly with attribution): **SOVA (CC BY 4.0)**,
**FLEURS (CC BY 4.0)**, **Common Voice (CC0)**, **ruaccent (MIT)**,
**russian_g2p (MIT)**, **Epitran (MIT)**, **Silero Stress (MIT)**,
**RUAccent dictionaries mirror (Apache-2.0)**.

---

## 6. Verification log (all fetched 2026-11-19)

| URL | what was verified | HTTP |
|---|---|---|
| `https://ruslan-corpus.github.io/` | 31.5 h / 1 speaker / CC BY-NC-SA 4.0 quote / GDrive link | 200 |
| `https://drive.google.com/drive/folders/1QjaIKtPHmj-baiUMjjQqe8XjZ5XpiNoC` | official RUSLAN download folder reachable | 200 |
| `https://huggingface.co/datasets/akuzdeuov/ruslan-tts` | 22,200 rows, 22.05 kHz, 9.83 GB download | 200 |
| `https://huggingface.co/datasets/stilletto/ruslan-stressed` | 22,200 lines, CC BY 4.0 claim (invalid) | 200 |
| `https://www.openslr.org/96/` | RuLS 98 h, “Public Domain in the USA” | 200 |
| `https://www.openslr.org/resources/96/ruls_data.tar.gz` | 9,129,924,586 bytes | 200 |
| `https://huggingface.co/datasets/istupakov/russian_librispeech` | 54,472 train / 1,400 val / 1,352 test; WAV 16 kHz header verified | 200 |
| `https://mfa-models.readthedocs.io/en/latest/corpus/Russian/Russian%20LibriSpeech.html` | 98.24 h, 57,224 utts, 13 speakers (5 F/8 M) | 200 |
| `https://github.com/i-celeste-aurora/m-ailabs-dataset` | 46h47m RU row; 2020 commercial-use licence text | 200 |
| `https://ics.tau-ceti.space/data/Training/stt_tts/ru_RU.tgz` | 3,867,126,697 bytes | 200 |
| `https://ics.tau-ceti.space/data/Training/stt_tts/samples/ru_RU/f.wav` | WAV header: 16 kHz mono 16-bit | 200 |
| `https://mfa-models.readthedocs.io/en/latest/corpus/Russian/M-AILABS%20Russian.html` | 46.78 h, 20,494 utts, 3 speakers (1 F/2 M) | 200 |
| `https://github.com/sovaai/sova-dataset` | 298 h audiobooks / 101 h devices; CC BY 4.0 | 200 |
| `https://raw.githubusercontent.com/sovaai/sova-dataset/master/LICENSE` | “Creative Commons BY 4.0 … Virtual Assistant, LLC.” | 200 |
| `https://huggingface.co/datasets/dangrebenkin/sova_rudevices_audiobooks` | 182,835 train / 20,315 test, 16 kHz, 29.7 GB | 200 |
| `https://www.openslr.org/114/` | Golos 1,240 h, 18 GB Opus, custom licence link | 200 |
| `https://raw.githubusercontent.com/sberdevices/golos/master/license/en_us.pdf` | “not a Creative Commons license”; BY-SA rework | 200 |
| `https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd` | 16 GB parquet, lowercase transcript sample | 200 |
| `https://huggingface.co/datasets/Sh1man/common_voice_21_ru` | 220.45 h validated, CC0, 32 kHz MP3 verified | 200 |
| `https://commonvoice.mozilla.org/en/datasets` | “exclusively available through Mozilla Data Collective” | 200 |
| `https://huggingface.co/datasets/google/fleurs` | CC BY 4.0, 3,693 ru rows | 200 |
| `https://huggingface.co/datasets/google/fleurs/tree/main/data/ru_ru` | TSVs; test 775 utts / 2.50 h / 16 kHz | 200 |
| `https://huggingface.co/datasets/google/fleurs/raw/main/README.md` | `license: cc-by-4.0` | 200 |
| `https://github.com/Den4ikAI/ruaccent` | MIT; README; 3.19M-form dictionary | 200 |
| `https://huggingface.co/ruaccent/accentuator` | `license: mit`; 6.26 GB repo; 21 MB dictionary | 200 |
| `https://huggingface.co/datasets/LeeAeron/RUAccent_dictionaries` | Apache-2.0; 0.85 GB | 200 |
| `https://en.wiktionary.org/wiki/Wiktionary:Copyrights` | CC BY-SA 4.0 + GFDL | 200 |
| `https://kaikki.org/dictionary/Russian/` | 428,275 word forms; 904.6 MB JSONL | 200 |
| `https://dumps.wikimedia.org/ruwiki/latest/` | 6,044,355,020-byte pages-articles dump | 200 |
| `https://dumps.wikimedia.org/legal.html` | CC BY-SA 4.0 + GFDL, commercial allowed | 200 |
| `https://github.com/nsu-ai/russian_g2p` | MIT; Accentor + G2P; bundled 54.7 MB lexicon | 200 |
| `https://github.com/espeak-ng/espeak-ng` | GPL-3.0 | 200 |
| `https://github.com/dmort27/epitran` | MIT | 200 |
| `https://github.com/Den4ikAI/ruphon` | no licence (`NOASSERTION`) | 200 |
| `https://github.com/snakers4/silero-stress` | MIT | 200 |
| `https://github.com/omogr/omogre` | no licence (`NOASSERTION`) | 200 |
| `https://github.com/MashaPo/russtress` | no licence (`null`) | 200 |
| `https://github.com/snakers4/open_stt` | ~20,000 h; CC BY-NC 4.0 badge | 200 |

### Blocked / not usable

| item | status |
|---|---|
| Common Voice official bulk download | **Blocked**: Mozilla Data Collective login required (“Effective October 2025 … exclusively available through Mozilla Data Collective”). CC0 mirror used instead. |
| M-AILABS original `caito.de` page | **Blocked**: DNS `NXDOMAIN`. Direct `ics.tau-ceti.space` archive and the GitHub relicense page work. |
| Taiga corpus | **Unavailable**: `TatianaShavrina/taiga_social_corpus` 404; `taiga_site` has no licence/data. |
| RUSLAN Google Drive bulk download | **Browser-only**: folder page 200, but Drive may require confirmation; HF mirror is scriptable. |
| OpenSTT | **Not downloaded** (356 GB Opus / 2.3 TB WAV, NC). Surveyed only. |
| SOVA Yandex Disk links | **Not exercised** (browser/possible quota); HF mirrors used. |

---

## 7. Notes for the download/preprocessing scripts

- **Target rate:** resample to 22.05 kHz, mono, 16-bit PCM. RUSLAN at 44.1 kHz
  is an exact 2:1 decimation (low-pass first). RuLS/M-AILABS/Golos/SOVA are
  16 kHz; upsampling adds no information but keeps one pipeline.
- **Trim/pad:** RUSLAN already has leading/trailing silence removed. RuLS is
  segmented; trim to ~0.1 s silence. Golos far-field has long silences.
- **Transcripts:** for TTS use `text_no_preprocessing` (RuLS) or the official
  RUSLAN text; do not use Golos/SOVA lowercase ASR transcripts as TTS text
  without restoration. Common Voice prompts do have case+punctuation, but the
  audio is crowd-quality.
- **Stress:** apply ruaccent at preprocessing time and store the stressed text
  next to the original. Do not modify the original transcripts in place.
- **Leakage:** if RuLS is used for training, exclude RuLS test from the WER
  judge. Prefer FLEURS test as the judge because its speakers are disjoint from
  train/dev.
- **Disk:** `data/` is gitignored; keep raw archives out of git. Convert to
  22.05 kHz and delete the originals once checksums pass.
