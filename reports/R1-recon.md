# R1 reconnaissance report

Task: map the `sultson/voca` Rust TTS codebase and the Python training stack it
was forked from, so the orchestrator can choose a Russian strategy.
Worktree: `/Users/suheugene/projects/22-tiniest-voca-ru/worktrees/recon`
(branch `task/recon`). Deliverable: `docs/repo-map.md`.

## What I did

1. Read the whole Rust inference tree: `model.rs`, `vocoder.rs`, `weights.rs`,
   `tts.rs`, `crates/tiny-voca/src/bin/voca.rs`, `crates/tiny-voca/src/bin/parity.rs`, `wav.rs`, `main.rs`, and the
   `dizione` crate (`symbols.rs`, `engine.rs`, `normalize.rs`, `numbers.rs`,
   `morphology.rs`, `sentence.rs`, `g2p.rs`, `homographs.rs`, `tables.rs`).
2. Cloned the upstream Python repository to `/tmp/tiny-voca-upstream`
   (`n4ze3m/tiny-voca`, `e7f1bde`, 2026-11-18) and the phonemizer to
   `/tmp/dizione-upstream` (`n4ze3m/dizione`, `8cd889e`, 2026-11-18).
3. Read the upstream training stack: `config.py`, `model.py`, `train.py`,
   `data.py`, `prepare.py`, `annotate.py`, `infer.py`, `export.py`,
   `losses.py`, `align.py`, the README, model card/dataset card and docs.
4. Downloaded the released model config and checkpoint from
   `huggingface.co/n4ze3m/tiny-voca` and inspected the checkpoint without
   PyTorch installed (stub unpickler; see `docs/repo-map.md` §3.4).
5. Counted and verified every tensor name/shape and the parameter count.
6. Cross-checked the Rust code against the upstream Python model and the
   checkpoint; flagged differences and hard-coded assumptions.

I did **not** train, synthesise audio, or run the Rust tests. No source code was
changed; only the two owned documents were added.

## Key numbers (verified)

| Item | Value | Evidence |
|---|---:|---|
| Inference parameters | 4,537,291 | released checkpoint: 184 tensors; `upstream/docs/model.md:33-44` |
| Tensor count | 184 | checkpoint inspection, `docs/repo-map.md` Appendix A |
| Checkpoint size / SHA-256 | ~18 MB / `ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70` | downloaded `tiny-voca.pt` |
| Sample rate / hop | 24,000 Hz / 256 (10.667 ms) | `upstream/src/tiny_voca/config.py:16-18` |
| Transformer | 192 ch, 4 heads, FF 640, kernel 7, layers 4/2/5 | `upstream/src/tiny_voca/config.py:26-32` |
| Vocabulary | 76 symbols: 5 special + 12 punctuation + 26 consonants + 33 vowels | `upstream/src/tiny_voca/text.py:50-60`; `dizione/src/dizione/symbols.rs:3-14` |
| Max input | 512 tokens / 3000 frames = 32.0 s | `upstream/src/tiny_voca/config.py:45-46` |
| Vocoder | source-filter, not HiFi-GAN; 25,862 params | `upstream/src/tiny_voca/model.py:1-17,165-238`; `upstream/docs/model.md:28-31` |
| Upstream training | 104,737 steps, 7.7 h on one RTX 4090 | `upstream/docs/model.md:73-77` |
| Upstream data | 17,371 clips / 19.1 h; train 16,018 / 17.6 h; synthetic F2/M2 | `upstream/docs/model.md:48-64` |
| Licences | code/weights Apache-2.0; dataset CC BY 4.0; dizione Apache-2.0; CMUdict BSD-style | `upstream/README.md`, `upstream/docs/dataset-card/LICENSE.md`, `dizione/scripts/fetch_sources.sh` |

## Most important corrections to prior assumptions

1. **Weights are not compiled into the binary.** `weights.rs` is a safetensors
   reader. `voca` loads `$VOCA_HOME/tiny-voca.safetensors`; `install.sh`
   downloads `voca-data.tar.gz` (19 MB) separately from the binary.
2. **The upstream training data was not LJSpeech.** It is
   `n4ze3m/tiny-voca`: synthetic Qwen3-TTS audio, two voices, 19.1 h, with text
   from LibriTTS-R, an LLM, and Mozilla Common Voice, plus project templates.
3. **This is English TTS only.** No STT, no voice cloning, two fixed voices.
4. **The Rust repo has no training code.** Use upstream Python; the released
   checkpoint is format version 4 (`export_model`).
5. **The 76-symbol layout is compile-time in Rust**, including the 26-consonant
   / 33-vowel split. A Russian inventory with hard/soft consonant pairs likely
   needs a Rust change.
6. **The vocoder is reusable in architecture, not in weights.** It is a
   source-filter harmonic/noise head with only 25,862 parameters; retraining it
   is cheap, but its weights are tied to the English F2/M2 voices.

## Russian strategy summary (detail in `docs/repo-map.md` §6)

- **Recommended:** retrain the same 4.5M architecture from scratch on Russian,
  reusing the upstream training stack and keeping 24 kHz / hop 256, with a new
  Russian frontend and a Russian duration/alignment pipeline.
- **Why:** the network is language-agnostic (phone ids + stress + voice); the
  model is tiny; the community list already has German, Italian, Turkish,
  Korean, Polish, Spanish, French, Portuguese, Arabic, Bengali and Sanskrit
  derivatives, including CPU-trained Arabic and fine-tuned Sanskrit
  (`upstream/README.md:109-129`).
- **Why not fine-tune English by default:** the token embeddings and text
  encoder are phoneme-specific; the voice embeddings are F2/M2-specific. Use
  fine-tuning only for very small Russian data (<2 h), and expect an English
  accent/voice risk.
- **Why not replace the model:** loses the 30 ms / 4.5M / Rust single-binary
  product constraints; only justified if Russian quality dominates.
- **Critical path:** Russian phoneme inventory + Russian durations/alignment.
  Upstream’s aligner (`facebook/wav2vec2-lv-60-espeak-cv-ft`,
  `upstream/src/tiny_voca/align.py:38`) is English/eSpeak and cannot label
  Russian phones.
- **Vocoder:** keep the architecture; retrain its two linear heads. The Rust
  vocoder code is reusable unchanged if sample rate/hop and tensor names stay.

## How to reproduce

```sh
# This worktree
cd /Users/suheugene/projects/22-tiniest-voca-ru/worktrees/recon
git log -1 --oneline                 # 8cc7d11 + R1 commits
cargo test --workspace               # not run: cargo is not installed here
cargo build --release -p tiny-voca --bin voca

# Upstream
git clone https://github.com/n4ze3m/tiny-voca /tmp/tiny-voca-upstream
git clone https://github.com/n4ze3m/dizione /tmp/dizione-upstream
cd /tmp/tiny-voca-upstream && git log -1 --oneline    # e7f1bde
cd /tmp/dizione-upstream && git log -1 --oneline      # 8cd889e

# Released model
cd /tmp
curl -fsSL -O https://huggingface.co/n4ze3m/tiny-voca/resolve/main/config.json
curl -fsSL -O https://huggingface.co/n4ze3m/tiny-voca/resolve/main/tiny-voca.pt
shasum -a 256 tiny-voca.pt
# tensor inspection snippet: docs/repo-map.md §3.4
```

All file:line citations for the analysis are in `docs/repo-map.md`. The
tensor table (184 rows) is in Appendix A of that file.

## Open questions / blockers

1. **Inventory:** expand beyond 76 symbols, or keep 76 and encode palatalisation
   with a `ʲ` token? Expanding is cleaner but requires coordinated Rust +
   checkpoint changes (`model.rs` constants, `crates/dizione/src/symbols.rs`, config).
2. **Russian durations:** which aligner/model? MFA Russian, a multilingual CTC
   model, or a corpus with alignments? This is the main data blocker.
3. **Russian corpus and licence:** not yet surveyed (R2’s job). Need training
   and redistribution rights; do not assume CC BY 4.0.
4. **Compute:** MPS support for the upstream vocoder’s complex STFT/FFT ops is
   unverified. CPU training of a 17.6 h-equivalent corpus will be slow; rented
   CUDA may be cheaper. R4 owns this.
5. **Voice scope:** one Russian voice or two? Keep the English F2/M2 voices?
   A single shared 76-symbol inventory cannot cleanly hold both languages if
   Russian needs more consonants; two model files/CLIs may be simpler.
6. **Quality target:** upstream WER checks are small; decide a Russian
   acceptance metric (WER? MOS?) before training.
7. **Interfaces:** preserve `voca` flags, `$VOCA_HOME` layout, `tiny-voca.safetensors`
   filename, 24 kHz / 32-bit-float WAV, and the `Tts`/`Phonemizer` API. If the
   Russian model keeps these, the installer and CLI need no changes.

## Risks

- Scope creep: a Russian frontend is a full normalizer + stress-aware lexicon +
  G2P + homographs, not a table swap.
- Licence: eSpeak NG (GPL-3.0) was used for the first 66k steps of upstream
  labels; it is not a runtime dependency, but avoid GPL tools in a permissive
  Russian pipeline. CMUdict (BSD-style) is not needed for Russian unless the
  English frontend is kept.
- Model quality: 4.5M on a few hours of Russian may be intelligible but not
  natural; the upstream English voice is itself synthetic.
- The “one binary” claim is really “binary + 19 MB data”; do not design a
  Russian release that assumes weights are embedded.

## Files added

- `docs/repo-map.md` — full analysis, architecture, tensors, upstream repo,
  English hard-coding, Russian strategy, licences, interfaces, risks,
  Appendix A (all 184 tensors), Appendix B (reproduction).
- `reports/R1-recon.md` — this report.
