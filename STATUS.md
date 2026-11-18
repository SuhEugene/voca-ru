# STATUS

Project: Russian model for `sultson/voca` (Rust TTS, 4.5M params, English only).

| ID | Task | Owner | Branch | State | Blockers |
|----|------|-------|--------|-------|----------|
| —  | Scaffold | orchestrator | main | done | — |
| R1 | repo analysis | — | task/recon | pending | — |
| R2 | russian data survey | — | task/ru-data | pending | — |
| R3 | base model survey | — | task/base-models | pending | — |
| R4 | compute + env | — | task/compute-env | pending | — |

## Risks

1. Upstream is a **Rust** rewrite; training code was removed (NOTICE). We must recover the original Python training stack from `n4ze3m/tiny-voca`.
2. This machine is an Apple M5 Pro (no CUDA). Training budget is MPS/CPU only unless the human rents compute.
3. Licences: upstream Apache-2.0; CMUdict data ships with the phonemizer; Russian corpora licences unverified.
4. Test-set leakage / normalisation mismatch — see plan section 7.
