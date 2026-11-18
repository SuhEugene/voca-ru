# Plan: Russian voices for voca, trained the tiny-voca way (synthetic Qwen3-TTS distillation)

Supersedes plan Sections 4–5 ("Branch A/B") for this project. Branch is **B′: synthetic teacher distillation**, chosen by the human on 2026-11-19. Do not collect a Russian speech corpus; do not fine-tune the English weights. We reproduce the upstream recipe with Russian text and Russian voices.

## 1. The upstream recipe, as it actually is

Verified from `n4ze3m/tiny-voca` (`README.md`, `docs/model.md`, `scripts/generate_audio.py`, `scripts/build_text_v2.py`, dataset card `n4ze3m/tiny-voca`):

1. **Voices are designed, not recorded.** Qwen3-TTS-12Hz-1.7B-**VoiceDesign** turns a text description into a reference clip. Two references exist: F2 (female) and M2 (male). Stored as `references/F2.wav`, `references/M2.wav` + `voices.json` (description, seed, reference transcript). No real speaker involved.
2. **Scripts.** ~4,905 sentences (LibriTTS-R transcripts, CC BY 4.0 + LLM-written English) for batch 1; 3,479 CC0/template sentences for batch 2 (numbers, dates, money, names, questions). Every sentence is read by **both** voices.
3. **Audio.** Qwen3-TTS-12Hz-1.7B-**Base**, voice-clone prompt from the reference, batches of 8, seeded. 24 kHz mono FLAC, peak-normalised to 0.95.
4. **Screening.** Every clip transcribed with Whisper `small.en`; normalised WER > 15%, implausible duration (0.5–35 s), speaking rate, or excessive silence → regenerate once; still bad → drop (~3–4 %). Upstream: 17,371 accepted clips / 19.1 h.
5. **Labels.** `prepare.py` extracts log-mel, pitch/voicing, energy. `annotate.py` takes phonemes + stress from the front end and gets phoneme durations from forced alignment with `facebook/wav2vec2-lv-60-espeak-cv-ft`, using eSpeak phonemes as the CTC label inventory. eSpeak NG and `dizione` produce the same 76-symbol inventory.
6. **Model.** 4.5M params, FastSpeech-2-style with rotary transformers + a 25.9k-param source-filter (DDSP-style) waveform head. Two training phases: spectral only for 8k steps, then + adversarial (MPD/MRD) at ratio 0.7. Reference run: 104,737 steps, 7.7 h on one RTX 4090, ~$9.56 total project spend.
7. **Eval.** Whisper round-trip WER on validation/test/new sentences. Upstream: 0.6 / 0.9 / 0.0 %.

## 2. What changes for Russian

| Upstream | Russian plan |
|---|---|
| English scripts | Russian scripts: public-domain/CC0 text + LLM-written + Russian templates. Same "every sentence in both voices" property. |
| `language="English"` in `generate_audio.py` | `language="Russian"` — Qwen3-TTS 1.7B Base/VoiceDesign officially support ru. |
| Whisper `small.en` screen | Russian-capable ASR screen: `large-v3` (or `large-v3-turbo`) + Russian text normaliser. Threshold re-calibrated, not copied. |
| eSpeak `en-us` / dizione | eSpeak NG `ru` phonemes (or an equivalent compact Russian inventory) + **stress** feature. Needs a Russian text normaliser (numbers, dates, money, abbreviations, `ё`, Latin words). |
| 76-symbol English inventory | Russian inventory (see D3). Model symbol table may differ; checkpoints are per-language anyway. |
| Rust `dizione` frontend | New Rust Russian frontend. **Biggest integration risk** — decision in §5. |
| Two English voices | Two Russian voices, F and M, same interface as `f2`/`m2`. |

## 3. Locked decisions

- Personal/research licence scope; record restrictions in the model card. Qwen3-TTS is Apache-2.0; upstream dataset is CC BY 4.0; text sources must be CC0/PD or project-own.
- Two voices (F + M).
- Local M5 training; **no training run starts without the human's explicit go-ahead**; GPU rental needs a cost shown first.
- Fork `SuhEugene/voca-ru`; no upstream PR unless asked.

## 4. Waves and tasks

Wave 0 (running): R1 repo map, R2 data/text/stress survey, R3 base-model survey (fallback reference), R4 compute, R5 community-port study.

### Wave 1 — foundations (parallel, after Gate 0)

| ID | Task | Owns | Acceptance |
|---|---|---|---|
| F1 | Python 3.12 env via `uv`, pinned: torch, `qwen-tts`, faster-whisper, transformers, librosa, soundfile; `make setup`, `make doctor` (device, versions, model cache) | `env/`, `Makefile` | `make doctor` runs on this M5; records whether MPS works |
| F2 | Russian script corpus: sources + templates + LLM-written; dedupe, filter, split by sentence; target ~4.5k sentences for batch 1 (train/val/test) and ~3.5k train-only batch 2 | `scripts/text/`, `data/scripts-ru/` | JSONL matching upstream columns; val/test frozen; every row has source+licence |
| F3 | Russian text normaliser + G2P (Python): numbers/dates/money/abbreviations, `ё` policy, Latin, punctuation, stress; eSpeak `ru` adapter | `training/voca_ru/text/`, `tests/text/` | ≥200 unit cases; deterministic; same code used by generation and training |
| F4 | Teacher harness: VoiceDesign refs, batch clone generation, SQLite resume, screening, manifest | `training/generate/` | `--limit 8` smoke produces 8 accepted clips locally or documents the blocker |
| F5 | Eval harness: round-trip WER/CER (Whisper large-v3 ru + normaliser), per-category breakdown (numbers, names, questions), stress-error list, RTF, JSON+MD | `training/eval/`, `tests/eval/` | Runs on upstream English checkpoint+data as a sanity check |
| F6 | Alignment spike: prove `wav2vec2-lv-60-espeak-cv-ft` aligns Russian eSpeak phonemes; measure accept rate on 200 clips; fallback (Russian CTC model / MFA) | `reports/F6-align.md` | Go/no-go, with numbers |
| F7 | Rust Russian frontend prototype: text → phoneme+stress tokens, same inventory as F3; parity test against Python on 1k sentences | `crates/voca-ru-text/` | ≥99 % token parity; no eSpeak dependency at runtime |

### Wave 2 — data and training prep

| ID | Task | Owns |
|---|---|---|
| D1 | Design + approve two Russian voices (VoiceDesign descriptions, seeds), human listens | `references/`, `voices.json`, `reports/D1-voices.md` |
| D2 | Generate the corpus (2 voices × scripts), resume-safe, screening, manifest | `data/synthetic-ru/` (ignored) |
| D3 | Prepare + annotate (adapt upstream `prepare.py` / `annotate.py` to F3/F6) | `training/prepare/`, `training/annotate/` |
| T1 | Training entrypoint: same architecture/hyperparameters as upstream, resume, periodic eval hook, sample synthesis of a fixed Russian sentence list | `training/train/`, `configs/` |
| T2 | Configs: upstream-equivalent baseline + one cheaper/smaller run; hardware-specific batch sizes from R4 | `configs/*.yaml` |

**Gate 2:** ≤100-step smoke train completes; checkpoint synthesises a Russian sentence; F5 scores it.

### Wave 3 — training (each run needs human approval)

TR1 upstream-equivalent (104k steps, GAN from 8k). TR2 optional shorter/ablation. M1 monitors, kills diverging runs. Selection by dev round-trip WER + human listening, confirmed once on the frozen test set.

### Wave 4 — integration into voca (Rust)

I1 port the Rust frontend from F7, wire it behind language selection; I2 `--lang ru` / config defaults and error messages; I3 embed the new weights (follow `scripts/convert_weights.py` and `weights.rs` conventions); I4 keep English path byte-identical; I5 integration tests with Russian audio fixtures.

### Wave 5 — release

L1 model card (teacher model, synthetic-data provenance, licences, metrics, limitations incl. "synthetic voice, not a person"); L2 README/docs; L3 independent reproduction.

## 5. Open decisions (need the human)

1. **Frontend strategy.** (a) Port eSpeak-ru rules to Rust; (b) our own Russian G2P + stress dictionary modelled on `dizione`; (c) relax the "no runtime dependency" property. Recommend (b), decided after R5/ports and F7 prototype evidence.
2. **Voice approval.** You listen to candidate F/M references before 17k clips are generated.
3. **Generation compute.** Teacher generation on this M5 is unproven (qwen-tts is CUDA-oriented). If MPS fails, a rented 4090 for a few hours is ~$10–20; that is a spend decision.
4. **Script sourcing.** Which Russian text sources you accept (R2 will list licences).

## 6. Cost and time (order of magnitude)

- Teacher generation: 1–3 GPU-hours on a 4090 (upstream's whole project was $9.56).
- Training: 4.5M params, ~104k steps, 19 h audio. Upstream 7.7 h on a 4090; expect roughly 2–4× that on M5 MPS, i.e. 1–2 days local, if MPS is supported by every op. CPU fallback is slower.
- Storage: ~19 h of 24 kHz FLAC ≈ 4–6 GB; features/annotations a few GB.

## 7. Risks

1. **Qwen3-TTS Russian quality** is the ceiling: errors in the teacher are baked in. Mitigation: listen to a 200-clip sample before full generation; screen with a Russian ASR.
2. **Alignment labels**: the upstream aligner is an eSpeak-phoneme recogniser; Russian coverage unproven (F6).
3. **Rust frontend** is the largest engineering item and the one with no upstream analogue (R5 studies how other ports did it).
4. **MPS coverage**: some ops (custom source-filter, adversarial) may fall back to CPU or fail; R4/F1 answer this before Wave 3.
5. **Licences**: synthetic audio is clean; text sources and any stress dictionary are not automatically so.
