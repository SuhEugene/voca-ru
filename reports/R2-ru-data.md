# R2 — Russian data survey

**Owner:** R2 · **Branch:** `task/ru-data` · **Date:** 2026-11-19
**Deliverable:** [`docs/datasets-ru.md`](../docs/datasets-ru.md)

## TL;DR

- **RUSLAN is the only true studio single-speaker Russian TTS corpus**
  found: 31.5 h, one male speaker, 44.1 kHz/16-bit, punctuation + case in the
  transcripts, SNR ≈ 90 dB. It is **CC BY-NC-SA 4.0** — non-commercial and
  share-alike. **The orchestrator must ask the human before using it.**
- **Best licence-clean fallback: Russian LibriSpeech (RuLS).** 98.24 h,
  13 readers (5 F / 8 M), 16 kHz, **public domain in the USA**, and its
  `text_no_preprocessing` field already has case and punctuation. Filter one
  reader for a 7–15 h single-speaker model.
- **Russian stress is solved cheaply:** `ruaccent` is **MIT**, ships a
  **3,194,879-form stressed dictionary** (21 MB gz) plus 19,741 omographs and a
  ~0.8 MB ONNX model. `russian_g2p` (**MIT**) supplies the phoneme layer.
  This matters more than another 50 h of unaccented audio.
- **Small ASR eval set:** FLEURS ru test — 775 utts / **2.50 h**, 16 kHz,
  **CC BY 4.0**, disjoint test speakers, punctuated references. One 300 MB
  parquet of RuLS test is a useful in-domain cross-check.
- **No large downloads were made.** All numbers come from live pages/APIs on
  2026-11-19; the full URL log is in the doc.

## Recommended first model

### Option A — best quality (needs NC approval)

| component | size | destination | weight |
|---|---:|---|---:|
| RUSLAN 31.5 h (HF 22.05 kHz mirror) | 9.83 GB (7 GB official) | `data/raw/ruslan/` | 1.0 acoustic |
| ruaccent dictionary + omograph model | 0.02 GB | `data/raw/stress/ruaccent/` | 1.0 stress |
| FLEURS ru test | ~0.2–0.5 GB | `data/raw/fleurs/` | eval only |

**≈ 10 GB download**, ≈ 5 GB after 22.05 kHz conversion. This is the
recommended first model *if the human accepts CC BY-NC-SA 4.0*.

### Option B — licence-clean (no NC)

| component | size | destination | weight |
|---|---:|---|---:|
| RuLS filtered to one reader (expect 7–15 h) | 9.13 GB | `data/raw/ruls/` | 1.0 acoustic |
| ruaccent dictionary + model | 0.02 GB | `data/raw/stress/ruaccent/` | 1.0 stress |
| FLEURS ru test | ~0.2–0.5 GB | `data/raw/fleurs/` | eval only |

**≈ 9.35 GB download.** Commercial-friendly in the US (PD-US + MIT + CC BY 4.0),
but audiobook acoustics and 16 kHz source. Expect a weaker voice than RUSLAN.

### Option C — few-speaker, smallest (needs licence decision)

M-AILABS Russian 46.78 h / 3 speakers, filter one (~15 h), **3.87 GB** +
ruaccent + FLEURS = **≈ 4.1 GB**. The 2020 repo licence explicitly permits
commercial use, but an HF mirror labels the same data **CC BY-NC-ND 4.0**, and the
original `caito.de` page is dead. Do not use until the human rules on it.

## Larger mix (~48.5 GB, no SOVA)

RUSLAN 0.40 · RuLS 0.25 · M-AILABS 0.15 (if cleared) · Golos 0.10 ·
ruaccent · Wiktionary 0.10 · ruwiki 0.10 · FLEURS eval.
Total ≈ 48.5 GB: 9.83 + 9.13 + 3.87 + 18.47 + 0.02 + 0.90 + 6.04 + 0.2.
Add SOVA RuAudiobooks 298 h (+29.7 GB) if more multi-speaker pretraining is
wanted. Pretrain multi-speaker, then fine-tune 10–30 h of the target speaker.

## Licence flags — require human decision

| dataset | flag |
|---|---|
| **RUSLAN** | **NC + SA** (CC BY-NC-SA 4.0). Best corpus, unusable commercially without permission. |
| **M-AILABS Russian** | **Contradictory licences**: 2020 commercial-use text vs CC BY-NC-ND 4.0 on an HF mirror. |
| **Golos** | Custom **BY-SA-like** licence (“not a Creative Commons license”); commercial granted, share-alike scope unclear for weights. |
| **OpenSTT** | **CC BY-NC 4.0**; 20,000 h, 356 GB Opus. Not for this project. |
| **RUSLAN derivatives** | `stilletto/ruslan-stressed` (CC BY 4.0 claim), `akuzdeuov/ruslan-tts` (no licence), `intexcp/ruslan_sova_ai` (MIT claim), `Diana2di/ruslan_data` (no licence) all relicense NC-SA audio. **Invalid; do not trust their cards.** |
| **RuLS** | “Public Domain in the USA” — jurisdiction caveat outside the US. |
| **RUPhon / Omogre / russtress** | **No licence** (`NOASSERTION`/`null`). Blocked until clarified. |
| **espeak-ng** | **GPL-3.0**; baseline only, do not embed in the Apache-2.0 binary. |
| **Taiga corpus** | No working repo/licence; not used. |

## Blockers

1. **Common Voice official bulk download is login-gated** since Oct 2025
   (Mozilla Data Collective). Used the CC0 mirror
   `Sh1man/common_voice_21_ru` (220.45 h validated) instead.
2. **M-AILABS original site is dead** (`caito.de` NXDOMAIN); direct archive at
   `ics.tau-ceti.space` works (3.87 GB, HTTP 200).
3. **RUSLAN official download is a Google Drive folder** — reachable (HTTP 200)
   but not script-friendly. A HF mirror exists; its card has no licence, so
   treat it as RUSLAN NC-SA.
4. **SOVA Yandex Disk links were not exercised**; HF mirrors used and verified.

## Why this mix

- A 4.5M-param model needs roughly 5–50 h for one speaker. RUSLAN (31.5 h)
  and a single RuLS reader (7–15 h) both fit; M-AILABS gives ~15 h per speaker.
- Transcripts must carry **case and punctuation** for a TTS front end. RUSLAN
  and RuLS `text_no_preprocessing` do; Golos/SOVA ASR transcripts do not
  (CV prompts do, but the audio is crowd-quality). This is the main reason
  Golos and SOVA are pretrain-only.
- **Stress is the Russian-specific risk.** ruaccent (MIT) removes it for the
  training vocabulary; russian_g2p (MIT) converts stressed text to phonemes.
  No NC data is needed for this layer.

## Verification / acceptance

- [x] Every table row has a URL fetched on 2026-11-19; HTTP status recorded.
- [x] Licences quoted or linked, not assumed (RUSLAN quote, Golos PDF,
      SOVA LICENSE, Wiktionary copyrights, HF cards).
- [x] Hours / speakers / sample rates checked against publisher or MFA metadata
      (RuLS 98.24 h / 13 spk; M-AILABS 46.78 h / 3 spk; FLEURS 2.50 h test;
      Golos 1,240 h; SOVA 298 h; RUSLAN 31:32:05).
- [x] Sample rates verified from WAV headers (RuLS 16 kHz, M-AILABS 16 kHz,
      RUSLAN 44.1 kHz from paper), MP3 frame header (CV 32 kHz), FLEURS 16 kHz.
- [x] Recommended mix is specific enough to write download scripts; commands
      are in `docs/datasets-ru.md` §4.5.
- [x] No full datasets downloaded.
- [x] `git status` clean after commit; commit prefix `R2:`.

## Next actions

1. Orchestrator asks the human: **RUSLAN NC-SA y/n** and **M-AILABS licence
   conflict y/n**. These decide Option A vs B vs C.
2. If approved, run the RUSLAN download + 22.05 kHz conversion, then build the
   stress-marked phoneme lexicon with ruaccent + russian_g2p.
3. Wire FLEURS ru test into the round-trip WER harness (Whisper-class judge).
4. Do not use Golos/OpenSTT until the share-alike/NC questions are settled.
