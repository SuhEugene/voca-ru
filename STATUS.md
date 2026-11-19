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
| G1 | teacher harness + MPS smoke | subagent | task/g1-teacher | **merged** (b48b31c); Base works on MPS; voice candidates pending VoiceDesign weights | VoiceDesign download |
| F6 | aligner spike (Russian eSpeak labels) | subagent | task/f6-align | **merged** (add64c1) — 97.7 % accept on FLEURS, 0 unalignable phones, mapping contract in `docs/aligner-mapping.md` | — |

## Risks

1. Upstream is a **Rust** rewrite; training code was removed (NOTICE). We must recover the original Python training stack from `n4ze3m/tiny-voca`.
2. This machine is an Apple M5 Pro (no CUDA). Training budget is MPS/CPU only unless the human rents compute.
3. Licences: upstream Apache-2.0; CMUdict data ships with the phonemizer; Russian corpora licences unverified. Research-only scope relaxes corpus choice but not attribution.
4. Test-set leakage / normalisation mismatch — see plan section 7.
5. Two voices doubles data and training work; plan for it in Gate 0 contracts.
