# STATUS

Project: Russian model for `sultson/voca` (Rust TTS, 4.5M params, English only).

Remotes: `origin` = github.com/SuhEugene/voca-ru (fork), `upstream` = sultson/voca (read-only).

## Human decisions (2026-11-19)

| Question | Answer |
|---|---|
| Licence scope | **Personal / research only.** Non-commercial corpora allowed; model card must record restrictions. |
| Compute | **Local M5 first. Never start training autonomously.** Renting GPU needs a concrete cost shown to the human first. |
| Release target | **Two Russian voices (female + male)**, mirroring `f2` / `m2`. |
| Publishing | Fork under `SuhEugene` — done (`voca-ru`). No upstream PR unless asked. |
| Voice choice (iteration 1) | **`f_neutral` (F) + `m_neutral` (M)** — from the four VoiceDesign candidates, chosen by ear 2026-11-19. `f_warm`/`m_warm` kept as alternates. |
| Adding voices later | Accepted to be **expensive**: new voice = VoiceDesign reference → full corpus generation for it → **retrain** the student with the expanded speaker list (`nn.Embedding(len(speakers))` in `model.py:254`). No speaker-embedding redesign for now. |

## Tasks

Branch: **B′ synthetic teacher distillation** — see `docs/plan-ru-qwen.md`. Russian audio comes from Qwen3-TTS VoiceDesign + Base clone; the 4.5M model is trained from scratch on it. No collected speech corpus.

| ID | Task | Owner | Branch | State | Blockers |
|----|------|-------|--------|-------|----------|
| —  | Scaffold | orchestrator | main | done | — |
| R1 | repo analysis | subagent | task/recon | **merged** (a067aa0) | — |
| R2 | russian data/text/stress survey | subagent | task/ru-data | **merged** (302fb86, both passes) | — |
| R3 | teacher + fallback survey | subagent | task/base-models | **merged** (71df916) | — |
| R4 | compute + Qwen-on-M5 test | subagent | task/compute-env | **merged** (f127c36) — MPS works, batch 16 ≈ 5× realtime, teacher gen ≈ 3.8 h | checkpoint sha mismatch (prefix only, splice in progress) |
| R4b | stand-down + corruption forensics | subagent | task/compute-env | **merged** (b17ee08) | — |
| R5 | community ports study | subagent | task/ports | **merged** (7c92dfe) | — |
| R6 | Silero applicability (teacher/stress/runtime) | subagent | task/silero | **merged** (fbf8ecc) | — |
| Gate 0 | `docs/contracts.md` | orchestrator | main | **done** | — |
| F2 | Russian scripts (8k sentences) | subagent | task/f2-texts | **merged** (4b66df2) — 4500 batch1 (3870/315/315) + 3500 batch2, tests green | — |
| F5 | eval harness (WER/CER, stress set, RTF) | subagent | task/f5-eval | **merged** (2be31d3) — 14 tests green; real ASR numbers pending whisper weights | whisper model.bin |
| F3 | Russian frontend (norm+stress+G2P, 76 symbols) | subagent | task/f3-frontend | **merged** (db75488) — 59 tests green in `.cache/venvs/f3`; OOV 2.05 % of tokens; 18/20 stress | — |
| P1 | training stack ported (4,537,291 params) | subagent | task/p1-train | **merged** (716b6c2) — 61 h for 104k steps on M5; decision pending | training time |
| F7 | Rust Russian frontend | subagent | task/f7-rust | **merged** (9cb41da) — 100 % token parity on 8k sentences; 5 MB dictionary | eSpeak-derived lexicon licence |
| G1 | teacher harness + voices + throughput | subagent | task/g1-teacher | **merged** (38b2a8c) — 4 voices in `out/g1/voices/`, batch 8 bucketed 3.0-3.2x realtime | human voice pick |
| G2 | stress adapter A/B | subagent | task/g2-adapter | partial — LoRA merges cleanly; Base+marks without LoRA = garbled speech; full A/B/C table missing | — |
| D2 | corpus builder + pilot | subagent | task/d2-corpus | **merged** (d790979) — 40/40 clips kept, WER 0.67 %, stress 184/184; full run not started | full run |
| S1 | remote generation bundle | subagent | task/s1-remote | **merged** (5212d6e) — bootstrap/generate/watch/publish/pack, 44 tests; bundle built `out/remote-bundle.tar.zst` | HF token, server |
| F6 | aligner spike (Russian eSpeak labels) | subagent | task/f6-align | **merged** (add64c1) — 97.7 % accept on FLEURS, 0 unalignable phones, mapping contract in `docs/aligner-mapping.md` | — |

## Risks

1. Upstream is a **Rust** rewrite; training code was removed (NOTICE). We must recover the original Python training stack from `n4ze3m/tiny-voca`.
2. This machine is an Apple M5 Pro (no CUDA). Training budget is MPS/CPU only unless the human rents compute.
3. Licences: upstream Apache-2.0; CMUdict data ships with the phonemizer; Russian corpora licences unverified. Research-only scope relaxes corpus choice but not attribution.
4. Test-set leakage / normalisation mismatch — see plan section 7.
5. Two voices doubles data and training work; plan for it in Gate 0 contracts.

## Downloaded assets (verified, do not re-download)

All under `<repo>/.cache/models` and git-ignored. SHA-256 verified against HF.

| Asset | Size | SHA-256 (prefix) | Used by |
|---|---:|---|---|
| `qwen3-tts-1.7b` (Base) | 3.86 GB | `38fc7fc5…` | teacher generation |
| `qwen3-tts-1.7b/speech_tokenizer` | 682 MB | `836b7b35…` | teacher |
| `qwen3-tts-1.7b-voicedesign` | 3.83 GB | `391e8db2…` | voice design |
| `qwen3-tts-ru-stress/lora/adapter.safetensors` | 35 MB | — | stress adapter (peft) |
| `qwen3-tts-ru-stress/meter/{head.pt,meter2.py}` | 4.2 MB | — | audio stress screening |
| `whisper-large-v3-turbo/model.bin` | 1.62 GB | `e76620f8…` | WER judge |
| `gigaam-v3/pytorch_model.bin` | 428 MB | `afc6dcba…` | cross-check judge |
| `wav2vec2-lv-60-espeak-cv-ft/pytorch_model.bin` | 1.26 GB | `3173bde9…` | forced alignment |
| `ruaccent/{dictionary,nn_accent}` | 24 MB | — | stress dictionary + NN |
| `fleurs-ru/test` (131 clips) | 91 MB | n/a (tar extract) | eval set |

Environment: one venv `.cache/venvs/r4` has torch 2.15, peft 0.21, faster-whisper 1.2.1,
qwen-tts, ruaccent, espeakng-loader, librosa, torchaudio, matplotlib.
Rust toolchain and its crates live in `.cache/{rustup,cargo}`.
Run `. scripts/cache-env.sh` in every shell; it pins the absolute cache path.

## Decisions log

Newest last. Every decision that constrains future work goes here, not only in memory.

| Date | Decision | Why / consequence |
|---|---|---|
| 2026-11-19 | Synthetic teacher (Qwen3-TTS), no recorded corpus | Plan branch B′; recorded corpora demoted to fallback |
| 2026-11-19 | RUAccent for training-time stress; Rust subset allowed at runtime under 50 MB / 300 MB gates | Russian stress is lexical; hidden from the model otherwise |
| 2026-11-19 | 76-symbol inventory (5 special + 23 punctuation + 48 phonemes) | Rust model/vocoder shapes unchanged |
| 2026-11-19 | Silero TTS excluded as a generator | Human decision; Qwen is the only teacher |
| 2026-11-19 | Teacher = Base + RUAccent marks + Russian LoRA | Base alone follows stress ~65 %; G2 measures the delta |
| 2026-11-19 | Voices = `f_neutral` + `m_neutral` for iteration 1 | Human listened to the same-phrase samples |
| 2026-11-19 | Adding a voice later = new VoiceDesign ref + corpus + **retrain** | Speaker set is compile/config time (`nn.Embedding`); accepted cost |
| 2026-11-19 | Batch corpus by length, not corpus order | 1.64× vs 3.0× realtime measured by G1 |
| 2026-11-19 | Big HF files via `hf download` or `fetch-hf-file.sh` (no `curl --retry` + append) | `--retry` restarts ranges and silently corrupts |

## Release artifacts (2026-11-27, v2)

| what | where |
|---|---|
| Model v2 at the repo root (new female voice `s3_lower_register`, 104k steps, WER 1,45 %) | https://huggingface.co/SuhEugene/voca-ru — `voca-ru.onnx`, `gguf/voca-ru-js.gguf` (27,3 MB), `inference.pt`, `samples/` |
| Previous voice set (104k / 134k / 164k) | same repo, `variants/<steps>/` — each with its own ONNX and GGUF |
| Rust runtime bundle | same repo, `rust/voca-data.tar.gz` (18,3 MB: safetensors + phonemizer tables) |
| Frontend dictionaries (all versions) | `crates/voca-ru-text/src/data/` — `stress_dict.txt` 150 000 форм, `g2p_lexicon.txt` 53 710 слов; на held-out FLEURS покрытие 70,7 % / 79,9 % (было 58,1 % / 66,4 %) |
| Corpus | https://huggingface.co/datasets/SuhEugene/voca-ru-corpus |
| Browser demo | https://tiny-voca.fnnk.dev — Dokploy app `tiny-voca` in project VL Self-Host, server `[Hoster] VLab`, built from GitHub `SuhEugene/voca-ru` branch `original` (Dockerfile → nginx:alpine serving `js/`), auto-deploy on push |
| Rust inference | branch `master` (Russian frontend inside the binary, 349× real time) |
| Upstream source | branch `original` (Russian JS frontend, Russian-first page) |
