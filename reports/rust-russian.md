# master: Russian runtime for voca-ru

Date: 2026-11-25. Worktree `/Users/suheugene/projects/22-tiniest-voca-ru/worktrees/rust`,
branch `master` (identical to `main` before this work).

## What changed

- **Inventory** — `crates/tiny-voca/src/model.rs` no longer hard-codes the
  English layout. The constants come from `voca-ru-text::symbols`
  (`SYMBOL_COUNT = 76`, `FIRST_PHONE = 28`, `FIRST_VOWEL = 64`), and
  `TinyVoca::load` reads the checkpoint's `symbols` metadata and compares it
  element-by-element with the frontend inventory. Weights for another
  inventory are rejected instead of silently shifting ids.
  `is_phone` (`id >= 28 || id == 4`) and `is_vowel` (`id >= 64`) now describe
  the Russian partition: 5 specials, 23 punctuation (5–27), 36 consonants
  (28–63), 12 vowels (64–75).
- **Frontend** — `crates/tiny-voca/src/tts.rs` uses `voca-ru-text` instead of
  `dizione` (normalise → stress → G2P → ids). `Tts::load(weights, data)` keeps
  its signature; the data directory is accepted but unused because the stress
  dictionary and G2P lexicon are embedded in `voca-ru-text`.
  Sentence splitting is now Unicode-aware (`is_alphanumeric`, `is_uppercase`,
  `is_numeric`) and has a Russian abbreviation/initial list; `«»` are handled
  as quotes. A byte-slicing bug for Cyrillic initials was found and fixed
  (`lower.len() - 1` → `lower.len() - c.len_utf8()`).
- **CLI** — texts and usage are Russian; `voca -v f2|m2`, speed, pitch,
  `--seed`, `--output`, `--file` behave as before. Development CLI's `--data`
  default is now `$VOCA_RU_DATA`. Peak is still not normalised (float WAV).
- **Data** — `out/tiny-voca.safetensors` (17 MB, 184 tensors, 4,537,291
  parameters) and `out/voca-data.tar.gz` (17 MB) built with
  `scripts/convert_weights.py`; archive layout:
  `tiny-voca.safetensors`, `phonemizer/stress_dict.txt`,
  `phonemizer/g2p_lexicon.txt`.
- **Docs** — `README.md` rewritten for Russian (build, local install, voices,
  archive recipe, quality limits); `NOTICE` and `install.sh` usage text
  updated.
- **Tests** — `tts::tests::splits_russian_sentences` (Cyrillic sentence
  splitting, abbreviation, quotes, numbers) and
  `tts::tests::frontend_ids_fit_the_model_layout` (fixed sentence →
  bos/eos, ids < 76, per-symbol id round-trip, phone/vowel id bounds). All
  existing `voca-ru-text` and `dizione` tests still pass.

## Parity with PyTorch

Reference: `scripts/dump_reference_ru.py` (new) writes tokens, stress, the
exact noise torch drew, durations, prosody, F0 and waveform for ten cases
(3 sentences × 2 voices × 1–2 settings). Rust side:
`target/release/parity out/tiny-voca.safetensors out/reference-ru`.

Target sentence, F2, seed 0, speed 1, pitch 0:

| metric | value |
|---|---|
| tokens | 45 |
| frames / durations | 229 frames, **durations equal** |
| max prosody difference | 5.72e-6 |
| max F0 difference | 9.16e-5 Hz |
| max waveform difference (absolute) | 6.85e-4 |
| max waveform difference relative to peak | **6.9e-4 (0.069 %)** |
| SNR | 72.6 dB |

All ten cases: durations equal, SNR 66.1–82.8 dB, max absolute waveform
difference 1.10e-3, no failures. The Rust seeded RNG matches PyTorch's noise
stream to the same level (`own-rng` column equals the handed-noise column).

End-to-end check of the actual frontend, not just handed tokens: the same
sentence through `scripts/say.py` (Python) and `voca` (Rust) gives
**58,624 samples on both sides**, max absolute difference 6.71e-4, relative to
peak 6.71e-4, relative RMS 2.60e-4. `voca-ru-text` and Python `analyze` print
byte-identical stressed text, phonemes and token ids for the sentence.

## Speed (macOS, Apple-silicon release build)

2.44 s sentence (`Мой коллега уходит в отпуск на две недели.`), 5 runs each:

| threads | model time (median) | RTF | real-time multiple |
|---:|---:|---:|---:|
| 1 | 7.0 ms | 0.0029 | 349× |
| 4 | 8.3 ms | 0.0034 | 295× |

8.9 s sentence (one utterance):

| threads | model time | RTF | real-time multiple |
|---:|---:|---:|---:|
| 1 | 36.7 ms | 0.0041 | 242× |
| 4 | 22.9 ms | 0.0026 | 388× |

First audio (cold process, including model load and phonemisation):
**≈27–29 ms** (load 3–4 ms, phonemes 17 ms, model 7 ms). Total command with WAV
write 27–30 ms. The dev binary prints the same numbers;
`--threads 1` is faster for short sentences because thread setup dominates.

## Reproduction

```sh
cd /Users/suheugene/projects/22-tiniest-voca-ru/worktrees/rust
. "$(cd /Users/suheugene/projects/22-tiniest-voca-ru && git rev-parse --show-toplevel)/scripts/cache-env.sh"
export CARGO_HOME="$VOCA_CACHE/cargo" RUSTUP_HOME="$VOCA_CACHE/rustup" PATH="$VOCA_CACHE/cargo/bin:$PATH"

cargo test --workspace
cargo build --release -p tiny-voca --bin voca -p voca-ru-text

# weights + archive
mkdir -p out
/Users/suheugene/projects/22-tiniest-voca-ru/.cache/venvs/r4/bin/python \
  scripts/convert_weights.py \
  /Users/suheugene/projects/22-tiniest-voca-ru/.cache/models/voca-ru-run/inference.pt \
  out/tiny-voca.safetensors
rm -rf out/stage && mkdir -p out/stage/phonemizer
cp out/tiny-voca.safetensors out/stage/
cp crates/voca-ru-text/src/data/{stress_dict.txt,g2p_lexicon.txt} out/stage/phonemizer/
tar -czf out/voca-data.tar.gz -C out/stage .

# run
VOCA_HOME="$PWD/out" ./target/release/voca -o out/privet.wav \
  "Мой коллега уходит в отпуск на две недели." -v f2

# parity (Python first, then Rust)
PYTHONPATH=/Users/suheugene/projects/22-tiniest-voca-ru/training \
  /Users/suheugene/projects/22-tiniest-voca-ru/.cache/venvs/r4/bin/python \
  scripts/dump_reference_ru.py \
  /Users/suheugene/projects/22-tiniest-voca-ru/.cache/models/voca-ru-run/inference.pt \
  out/parity-sentences.txt out/reference-ru
./target/release/parity out/tiny-voca.safetensors out/reference-ru
```

## Open issues / not verified

- `install.sh` still downloads `voca-data.tar.gz` from the upstream
  `sultson/voca` release URL. The archive is built and documented in `out/`,
  but no Russian release is hosted yet, so a one-line `curl | sh` install of
  this fork is not live. Point `repo`/`release` at the Russian release when it
  exists (or host the archive elsewhere).
- Quality was not re-evaluated here: no ASR/WER run, no listening test. The
  quoted 4.76 % WER and the pitch/timbre notes come from the model card and
  `.cache/models/voca-ru-run/eval-*.json`; the 5-phrase set is small.
- Only macOS arm64 was built and measured. Linux/Windows and non-Accelerate
  BLAS paths were not exercised.
- `cargo fmt --check` fails on pre-existing files in `crates/dizione`; it is
  not a regression of this change and was not mass-reformatted.
- Output is float WAV and may peak slightly above 1.0 (observed 1.0057); 16-bit
  export needs normalisation first.
