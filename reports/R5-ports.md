# R5 — community ports: findings and Russian recommendation

Full survey with per-port citations: [`docs/ports.md`](../docs/ports.md).
Read from GitHub at the head commits below; no weights or datasets downloaded.

## Bottom line

1. **Copy the Spanish port (`stdevMac/tiny-voca`) as the pipeline template.** It is
   the only one that matches our locked plan exactly: Qwen3-TTS VoiceDesign refs →
   Qwen3-TTS 1.7B Base → Whisper large-v3-turbo screen → `wav2vec2-lv-60-espeak-cv-ft`
   align → from-scratch 4.5M → two voices → eSpeak frontend → browser demo. Its
   `languages/<lang>.py` profile shape and `scripts/run_pipeline.sh` are directly
   reusable.
2. **Copy Polish (`damaon/slyszeglosy`) for anything the Spanish port does not
   solve:** a rule-based frontend with lexical stress, frontend fingerprints, a
   data-driven aligner label map, and timing calibration. Its two calibration
   scripts are the F6 toolkit.
3. **Build an own Russian G2P; keep eSpeak only as the development oracle.** The
   plan's F7 says "no eSpeak at runtime", and voca is an Apache-2.0 Rust binary.
   eSpeak is GPL-3.0 and needs a C/WASM dependency. Polish proves a 39-phone
   rule-based frontend with stress can train a strong 4.5M model; Russian is more
   regular segmentally, with stress as the hard part.
4. **The aligner is probably fine.** I checked
   `facebook/wav2vec2-lv-60-espeak-cv-ft`'s `vocab.json` against a local eSpeak
   `ru` probe: it contains the Russian-specific labels (`nʲ ɭ ɭʲ tʲ dʲ sʲ ɕ ʑ ɨ ʐ
   ʂ ɵ ɡʲ pʲ kʲ bʲ mʲ vʲ fʲ tʃʲ rʲ xʲ tsʲ u"`). Only `zʲ`, `ɕː`, `u"` and the
   `#`/`@-` artefacts need mapping. This upgrades F6 from "unproven" to
   "likely go, measure accept rate".

## Size expectations

| | Spanish | Polish | pt-BR Kokoro | Our plan |
|---|---:|---:|---:|---:|
| Sentences | 11,050 | 10,020 | 8,945 | ~8,400 |
| Clips (2 voices) | 20,790 | 20,040 | 17,890 | ~16,800 |
| Hours after screen+align | 22.7 | 18.0 | 14.8 | ~18–20 |
| Steps | 142k | 104k | 100k | 100–142k |
| GPU hours (4090-class) | ~? (4090) | 8.9 (24 GB) | 5.2 | 6–9 |

Cost on a rented 4090: ~$3–10 (pt-BR reports ~$3 for 5 h; upstream's whole project
was $9.56). Local M5 is 2–4× slower *if* MPS covers the GAN and source-filter ops;
unverified (R4).

## Russian inventory, from a local probe

`espeak-ng 1.52.0 -v ru --ipa --sep=_` over 16 varied sentences → **49 distinct
phones**; ~44–48 after canonicalising eSpeak artefacts (`a#`, `u#`, `8`, `@-`,
`j_j`). Add 5 specials + 12 punctuation → **~61–65 symbols**, the same class as
Spanish 61 / Italian 66 / German 67. The token stream encodes vowel reduction
(`ʌ` vs `ɑ`, `ə`), which is a real cue; stress stays the parallel 0/1/2 feature
(`ˈ`/`ˌ` stripped), never a token.

## Frontend ranking

1. **Own rules + stress lexicon (Polish `wymowa`).** No GPL at runtime, exact
   control of the training token stream, matches F7. Main work: stress lexicon
   (top ~50–100k forms), homographs (`за́мок/замо́к`), `ё`, final devoicing,
   regressive voicing, `-тся/-ться`, `что/чтобы`, `-ого/-его`, OOV names.
2. **eSpeak NG `ru` (Spanish template).** Fastest to data and aligner-compatible;
   pin 1.52.0 (or 1.51), record the version in the fingerprint, ship it as a
   separate program/WASM. Cost: GPL + runtime dependency, F7 cannot pass.
3. **Grapheme.** Reject. Stress is unwritten, reduction is stress-dependent, and
   no port uses a pure grapheme frontend.

Whichever is chosen, **the same frontend must produce the training tokens and the
runtime tokens**; a mismatch invalidates the annotations. Freeze it before
generation (F4/D2), not after.

## Files upstream must change

Python: new `languages/russian.py`; `text.py` profile selection; `config.py`
`vocab_size`/speakers; `align.py` label map + onset offset; `annotate.py` QC/ASR
per profile; `evaluate.py` `ru` ASR; `scripts/generate_audio.py`
`language="Russian"`; a corpus builder; the browser demo.

Rust voca: `crates/tiny-voca/src/model.rs` hard-codes `SYMBOL_COUNT = 76` (line 91)
and validates `vocab_size` against it; `crates/tiny-voca/src/tts.rs` hard-wires
`dizione::Phonemizer`. Both need a language/frontend seam. A new
`crates/voca-ru-text` is cleaner than mutating `crates/dizione/src/symbols.rs`
(English `[&str; 76]`). Keep the English path byte-identical.

No true Rust port of tiny-voca exists. The closest is `ilyaizen/CopySpeech`
(Tauri): it bundles a Python wrapper + ONNX in a sidecar, not a Rust frontend.
Our Rust frontend is genuinely new work.

## Top pitfalls (details in docs/ports.md §10)

- **Pin eSpeak.** 1.51 vs 1.52 changes tokens (Italian: 68/5,594 texts; pt-BR:
  42/9,013; Spanish refuses to start on mismatch). eSpeak 1.51 also drops the last
  character on unterminated stdin — always append `\n`.
- **Aligner labels ≠ eSpeak output.** Polish needs 19 mappings, pt-BR 10, Turkish
  found `ɡ`/`ɯɯ` surprises. Calibrate the map on real audio.
- **The aligner marks onsets, not closures.** Italian's geminate durations came out
  *shorter* for long consonants; calibrate the offset (−10.3 ms Polish, −19 ms
  English default) and refine closures if needed.
- **QC thresholds are per-model.** Turkish lost 34 % with English thresholds;
  Korean needed 0.6/0.3/90 for a 1,205-way softmax. Measure, don't copy.
- **Listen before scaling.** pt-BR rejected Qwen3-TTS/Chatterbox clones by ear;
  a better-WER Magpie variant lost the blind A/B. Spanish/French both had to add
  inference-only excitation fixes for buzz at word edges. Budget a 200-clip
  listening gate (D1/F4).
- **Numbers/dates are the residual WER.** Build the Russian normaliser before the
  corpus and use it for screening, training and evaluation.
- **Preprocessing cost.** `librosa.pyin` is the bottleneck (Turkish: 1.9× real
  time; YIN 271×). Shard `prepare` across processes; 17k clips is hours otherwise.
- **Unicode:** store NFD like eSpeak (pt-BR lost nasality with composed `õ`); watch
  `ё`, combining stress marks, `й`.
- **Browser/ONNX:** Pyodide or an exact JS port + eSpeak WASM; cache-version every
  asset; ORT-web op gaps (Turkish had to quantise MatMul only). Always peak-scale
  to 0.99 before 16-bit.

## Unknown / could not verify

- Qwen3-TTS Russian voice quality and VoiceDesign descriptions (no port used ru).
- Actual Russian alignment accept rate (vocabulary coverage verified, not quality).
- Russian eSpeak inventory stability across 1.51/1.52 and on a large corpus.
- Licensing of a Russian stress lexicon (eSpeak data is GPL; Wiktionary/academic
  sources need review).
- Russian text sources (R2).
- M5/MPS training feasibility (R4).
- No formal MOS anywhere; only pt-BR's single listener (4/5).

## Suggested next actions

1. F6: run Polish `calibrate_aligner_labels.py` + `calibrate_aligner_timing.py` on
   200–500 Russian clips; publish accept rate, map, offset.
2. F3/F7: decide own-vs-eSpeak now, because it fixes the token stream; prototype
   the Russian G2P and measure parity on 1,000 sentences.
3. D1: generate 50–100 Qwen3-TTS Russian clips with the two candidate voices and
   listen before D2.
4. F4: copy Spanish `run_pipeline.sh` and Polish `bakeoff/` for the teacher test.
