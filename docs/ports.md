# Community ports of tiny-voca: what it actually takes to add a language

R5 survey, 2026-11-19. Scope: the non-English ports listed in the upstream README's
"Community" section. Goal: extract a Russian recipe for the synthetic-Qwen3-TTS
pipeline in [`plan-ru-qwen.md`](plan-ru-qwen.md), with the frontend, annotation and
integration parts first.

**Method.** Direct `git clone`/HTTPS from this machine timed out (GitHub egress is
blocked here), so the code was read through the GitHub API at the commits in the
table below; Hugging Face model cards were read over the web. No model weights or
datasets were downloaded. Numbers below are the ports' own published numbers, not
measurements made here, unless marked "R5 probe".

| Repo | Head commit | What it is |
|---|---|---|
| `stdevMac/tiny-voca` (Spanish) | `a4bf2bd979` | synthetic Qwen3-TTS, eSpeak `es`, 2 voices, full browser demo |
| `damaon/slyszeglosy` (Polish) | `e89222ee0c` | synthetic OmniVoice, own rule-based G2P + stress, 2 voices |
| `Haple/tiny-voca-ptbr-kokoro` (pt-BR) | `1e7ed99326` | synthetic Kokoro teacher, eSpeak `pt-br`, 2 voices |
| `ooodev/korean-tts` (Korean) | `c63fc0de00` | real KSS corpus, own Korean/English frontend, language-specific aligner |
| `carbogninalberto/tiny-voca-it` (Italian) | `99f3ab2ddc` | real CML-TTS recordings, eSpeak `it`, 12 voices, neural vocoder |
| `erkinemreta1/zerre-tts` (Turkish) | `cf3cdfaaa5` | real 2-speaker corpus, eSpeak `tr`, browser lexicon |
| `DanielSuncost/tiny-voca-fr` (French) | — | synthetic Qwen3-TTS, eSpeak `fr-fr` |
| `SemFence/tiny-voca-ptbr` (pt-BR) | — | synthetic Chatterbox, own `dizione-ptbr`, WebGPU |
| `ElMariones/mini-voz` (Spanish) | — | synthetic Qwen3-TTS, 4 voices |
| `wite-tech/clarvia-tts-tiny` (HF, Turkish) | — | synthetic, 190 h, 10 voices, own phonemizer |
| `apptor/apptor-tiny-tts` (HF, German) | — | synthetic Qwen3-TTS, 12 h |

---

## 1. Cross-port summary

| Port | Data | Teacher | Clips / hours | Voices | Frontend / G2P | Inventory | Alignment | Training |
|---|---|---|---|---|---|---|---|---|
| Spanish `stdevMac` | synthetic | Qwen3-TTS 1.7B Base (VoiceDesign refs) | 20,790 / 24.5 h; 19,238 / 22.7 h train | 2 | eSpeak NG 1.52.0 `es` | 42 phones, 61 symbols | `wav2vec2-lv-60-espeak-cv-ft`, zero mapping | 142k steps, RTX 4090 |
| Polish `damaon` | synthetic | OmniVoice 0.6B | 20,040 / ~20 h; 18,268 / 18.0 h train | 2 | own rules `wymowa`, no eSpeak | 39 phones, 56 symbols | same aligner, 19-entry label map, −10.3 ms | 104k steps, 8.9 h, 24 GB |
| pt-BR `Haple` | synthetic | Kokoro-82M built-in voices | 17,890 / 16.37 h; 15,539 / 14.78 h train | 2 | eSpeak NG 1.51 `pt-br` | 51 phones, 68 symbols | same aligner, 10-entry map | 100k steps, 5 h 10, rented GPU |
| Korean `ooodev` | real KSS | — | 12,854 / 12.86 h; 8,504 / 8.45 h aligned | 1 (v2: 3) | `g2pk` + NFD Jamo + `dizione` for English | 142 tokens | Korean XLSR, syllable-level + Jamo sub-allocation | 130k steps, 11 h 35, 4090 |
| Italian `carbogninalberto` | real CML-TTS | Qwen3-TTS for 6 emotional voices | 14 h cap → 11.52 h train | 12 | eSpeak NG 1.51 `it` + JS port | 66 symbols (42 consonants, 7 vowels) | same aligner, long→short map | 100k steps, 6 h 15, 5070 Ti |
| Turkish `zerre` | real 2 speakers | — | 8,650 / 16.01 h | 2 | eSpeak NG `tr` | 44 phones | same aligner, 2 maps | v1, 4090-class |
| French `DanielSuncost` | synthetic | Qwen3-TTS 1.7B Base | 17,913 clips; 14.1 h after alignment | 2 | eSpeak NG `fr-fr` | — | same aligner, 97.5 % accept | 100k steps, ~6 h |
| pt-BR `SemFence` | synthetic | Chatterbox | — | 2 | own `dizione-ptbr` (MFA lexicon) | 55 symbols | label map | 104k steps, 8 h, RTX 5080 |
| mini-voz | synthetic | Qwen3-TTS | 7,000 sentences / 27 h | 4 | eSpeak `es` | — | same aligner | 104k steps, ~8 h |
| clarvia (HF) | synthetic | pocket-tts + Trendyol | 154,000 aligned clips / ~190 h | 10 | own pure-Python phonemizer | 56 symbols | same aligner | 190k steps |
| apptor (HF) | synthetic | Qwen3-TTS 1.7B Base | ~15,000 / 12 h | 2 | eSpeak `de` + num2words | 67 symbols | same aligner | 82k steps, 10 h, 3090 |

Two facts stand out. First, **the model never changes**: every port keeps the 4.5M
architecture and only changes `vocab_size`, the speaker list and the token meanings.
Second, **the forced aligner is nearly always the same** `wav2vec2-lv-60-espeak-cv-ft`;
the language work is a label map plus calibration, not a new aligner (Korean is the
exception because its writing system is syllabic).

---

## 2. Spanish (`stdevMac/tiny-voca`) — closest analogue to our plan

This is the port to copy. It uses exactly our intended pipeline: Qwen3-TTS
VoiceDesign references, Qwen3-TTS 1.7B Base cloning, eSpeak phonemes, Whisper
screening, the upstream aligner, two voices, from-scratch training.

### Data

- Text: 11,050 sentences. 8,100 Mozilla Common Voice (CC0; 5,400 Wikipedia +
  2,700 sentence collector), 900 project templates (numbers, dates, money, names),
  2,050 questions/exclamations generated with Mistral Large
  (`README.md`, "Data and training"; `docs/dataset-card/README.md`).
- Audio: both voices read every sentence, 20,790 clips / 24.5 h passed Whisper
  screening at 15 % WER. Alignment accepted 19,238 utterances / 22.7 h for train
  (`README.md`; `docs/model-card/README-es.md`).
- Voices: F2/M2 designed from Spanish descriptions, cloned per clip. No real
  speaker (`README.md`, "Voices").
- Screening: Whisper large-v3-turbo; clips outside plausible rate/silence are
  regenerated. The card warns the screen measures intelligibility, not naturalness.
- The published dataset is `marcosmaceo/tiny-voca-es` (HF, Apache-2.0 weights).

### Frontend

- eSpeak NG **1.52.0**, voice `es`, 42 phones / 61 symbols
  (`src/tiny_voca/languages/spanish.py`).
- Inventory built empirically: union of eSpeak output over a 25,000-sentence
  Common Voice sample, minus foreign-language artefacts (`docs/spanish.md`,
  "Phoneme inventory"). Every symbol is also an aligner label, so **no alignment
  mapping was needed** — the cleanest case in this survey.
- Stress: eSpeak emits `ˈ`/`ˌ`; `parse_espeak` strips them into the parallel
  0/1/2 feature, never a token (`src/tiny_voca/text.py`). Spanish stress is
  predictable from spelling, so eSpeak's marks are reliable.
- Normalisation: `¿`/`¡` are tokens; `«»`→`"`; decimal comma vs thousands point
  handled by keeping digit punctuation inside the text piece. Numbers/dates/money
  are left to eSpeak, with a separate `numbers_es.py` only for WER scoring.
- Accents are front-end rewrites of one inventory (`es-AR` seseo/sheísmo, `es-CU`
  aspiration), not separate models (`docs/spanish.md`, "Accents"). Only `es-ES`
  was ear-checked.

### Annotations

- `facebook/wav2vec2-lv-60-espeak-cv-ft` out of the box; "all 42 Spanish phones are
  among them" (`docs/spanish.md`). No label map, no timing offset recalibration
  mentioned; upstream defaults were kept.
- The port found the model's source-filter excitation, not alignment, to be the
  quality bottleneck (see pitfalls): a word-edge roughness traced to pulses above
  the pitch harmonics, fixed at inference with `NOISE_ABOVE = (2, 33)`
  (`docs/spanish.md`, "Excitation at inference"; `docs/word-edge-artifact.md`).

### Training

- 122,000 steps + 20,000-step low-rate anneal = 142,000; selected on validation;
  one RTX 4090 (`README.md`; model card "Training"). Architecture unchanged; vocab
  61, speakers F2/M2.
- `docs/training-es.md`, `run-es.sh`, `scripts/run_pipeline.sh` are the runbook;
  `make pilot` runs the whole chain on macOS `say` without a GPU, which is a good
  smoke-test pattern for our F4.

### Integration

- Inference/ONNX inputs unchanged (`tokens`, `stress`, `voice`, `speed`,
  `pitch_shift`, `pitch_range`). The ONNX export gained the mixed-excitation
  setting, recorded in `tiny-voca-es.json` (`docs/model-card/README-es.md`).
- Browser: `site/` is a separate Spanish demo. `site/frontend.js` is a
  line-for-line JavaScript port of `text.py` including Python Unicode semantics;
  `site/worker.js` runs eSpeak NG 1.52.0 as WebAssembly, then ONNX Runtime Web.
- **This is the pattern for our Rust frontend**: they did not invent a second
  phonemizer; they ported the tokenizer/canonicaliser exactly and kept eSpeak as
  the G2P, with a golden test (`site/test/frontend_golden.mjs`).

### Artefacts

- Weights `marcosmaceo/tiny-voca-es` (Apache-2.0), dataset
  `marcosmaceo/tiny-voca-es`, demo <https://stdevmac.github.io/tiny-voca/>.
- WER: validation 2.00 %, test 1.79 %, new sentences 0.0 % (small sets; model card).
  No formal MOS.

---

## 3. Polish (`damaon/slyszeglosy`) — the template for an own frontend with stress

If Russian gets an own G2P (no eSpeak at runtime), this is the reference
implementation. It is also a synthetic two-voice 4.5M model at 18 h, the same
budget as ours.

### Data

- 10,020 sentences: 7,101 Common Voice PL sentence collector + 1,711 Europarl
  (both CC0), 900 templates, 308 project sentences (`docs/dataset-card/README.md`).
- OmniVoice 0.6B clones two voices designed from descriptions ("female, young
  adult"; "male, middle-aged"), 20,040 clips. Whisper large-v3-turbo screening:
  CER > 5 % (12 % when the sentence has digits) → regenerate once → drop; 244
  dropped. Alignment then dropped 386 more; 18,268 clips / 18.0 h train
  (`README.md`; dataset card; model card).
- License: weights/data CC-BY-NC 4.0 because OmniVoice weights are CC-BY-NC;
  code Apache-2.0. Important for our licence record: the teacher's licence can
  propagate to the synthetic corpus.

### Frontend

- `src/wymowa/` — rules only, no data files: letter→phone, digraphs, softening
  `i`, nasal vowel decomposition, `n`→`ŋ`, voicing assimilation across word
  boundaries, final devoicing, and **stress** (`g2p.py` docstring).
- Stress: penultimate by default; exceptions `-yka` nouns, `-liśmy/-byśmy`,
  conditional `-łbym`; secondary stress on the first syllable of 4+ syllable
  words; monosyllabic function words unstressed. A small exception lexicon
  (`lexicon.py`) handles loanwords and respellings.
- Inventory: 39 phones (8 vowels incl. `ɛ̃ ɔ̃`, 31 consonants), 56 symbols
  (`src/wymowa/symbols.py`). Stress is a parallel feature, never a token.
- The frontend is shared by annotation, training and inference through
  `src/tiny_voca/text.py`; a `frontend_identity()` hash invalidates stale
  annotations when rules change.

### Annotations

- Same `facebook/wav2vec2-lv-60-espeak-cv-ft`, but Polish needs work:
  - a 19-entry `ALIGNER_LABEL_FALLBACK` map (`ɨ→i`, `ɛ→e`, `ʂ→ʃ`, `ʐ→ʒ`,
    `ɕ→ʃ`, `tɕ→tʃ`, `dʑ→dʒ`, `ɔ̃→o`, ...) because the recogniser was trained on
    2021 eSpeak labels, not today's output;
  - the map was chosen from data, not docs, by maximising mean log-probability
    (`scripts/calibrate_aligner_labels.py`); mean phone probability rose 0.52 →
    0.63;
  - onset offset recalibrated to **−10.3 ms** (`scripts/calibrate_aligner_timing.py`);
  - QC thresholds set from the full distribution: mean token probability ≥ 0.45,
    weak-phone fraction ≤ 0.35, vowel voicing ≥ 0.45.
- Result: ~97 % of generated clips survived both stages (19,410 of 20,040;
  18,268 in train, 18.0 h); the rest are overlong/weak alignments.

### Training

- 104,000 steps, GAN from 8,000, one 24 GB GPU, 8.9 h, with a GPU thermostat
  (`--max-temp 76`). Same architecture; vocab 56, speakers F/M.
- Evaluation: Whisper large-v3-turbo, CER 0.37 % validation / 0.48 % test; teacher
  0.28 / 0.52 %. The student matches its teacher, which is the realistic target.

### Integration

- PyTorch + ONNX, same six inputs. Browser demo loads `wymowa` through **Pyodide**
  (`web/public/worker.js`) rather than porting the rules to JS. That is cheaper
  than a JS port but costs ~10 MB of runtime and is not an option for a Rust binary.
- `docs/tiny-voca/` keeps upstream docs; `NOTICE` lists every change.

### Artefacts

- `Damaon/slyszeglosy` (HF; ONNX + PyTorch), demo
  <https://slyszeglosy-demo.damaon.workers.dev>.

---

## 4. Brazilian Portuguese, Kokoro teacher (`Haple/tiny-voca-ptbr-kokoro`)

The closest analogue in **teacher choice**: Kokoro is a small open model with
built-in voices, and it phonemises with eSpeak, so teacher and student share a
front end.

### Data

- 8,945 sentences / 81,704 words: 3,600 Common Voice pt-BR (CC0), 900 templates
  (money, dates, times, phones, ordinals), 4,445 LLM-written; 1,897 questions.
  Split by sentence: 8,322 / 357 / 266 (`README.md`, "How it was made";
  `eval/corpus_summary.json`).
- Kokoro read all sentences in both voices: 17,890 clips, 17,433 accepted after
  Whisper large-v3-turbo screening (WER > 15 %, one wrong word always allowed,
  plus rate/silence), 16.37 h (`eval/teacher_generation.json`). Alignment left
  15,539 train clips / 14.78 h.
- Voices chosen **by ear**: Qwen3-TTS and Chatterbox clones "did not sound
  Brazilian"; Kokoro's own `pf_dora` (70 %) + `af_heart` (30 %) and `pm_santa`
  passed (`README.md`; `model/voices.json`).

### Frontend

- eSpeak NG 1.51 `pt-br`; 51 phones / 68 symbols (`model/tiny-voca-ptbr.json`).
- Symbols stored **NFD** like eSpeak output; storing composed `õ` silently turned
  nasal vowels oral — a concrete Unicode pitfall.
- Stress: upstream parallel feature; eSpeak marks stripped.
- Normaliser writes numbers, `R$`, dates, times, phone groups, ordinals and
  abbreviations out before eSpeak (`tiny_voca_ptbr/normalize_ptbr.py`).

### Annotations

- Same aligner; it labels Portuguese with European Portuguese phones, so 10
  Brazilian→EP substitutions are applied for alignment only (`æ→ɐ`, `y→ɨ`,
  `tʃ→t`, `dʒ→d`, `x→ʁ`, `r→ɾ`, `j→i`, ...). 15,539 + 671 + 491 clips accepted.
- QC: upstream thresholds, no retune reported.

### Training

- 100,000 steps, batch-frames 9000, max-batch 32, GAN 8000/0.7, teacher-final 0;
  5 h 10 on a rented RunPod GPU, one 12-minute interruption resumed from
  `last.pt` (`README.md`; `eval/recipe.json`). Vocab 68.

### Integration

- Ships a **self-contained package** (`tiny_voca_ptbr/`) with an ONNX-only runtime,
  no PyTorch: `Voice().speak()` (`README.md`, "Quick start").
- ONNX is 21.4 MB, same six inputs. No browser demo yet; they note a browser would
  need eSpeak WASM.
- Evaluation includes a **frontend parity check**: the package reproduces training
  tokens on 1,568 texts and all phoneme strings in the report
  (`eval/frontend_check.json`), and the eSpeak-version drift test on 9,013 texts
  (1.51 exact, 1.52 differs on 42, mostly names — `eval/espeak_versions.json`).

### Artefacts

- `Haple/tiny-voca-ptbr-kokoro` (HF, Apache-2.0); WER 1.44 % on 16 new sentences,
  0.0 % test, 0.78 % validation; one listener rated both voices 4/5
  (`eval/listening.json`). A Magpie fine-tune scored better on WER but lost the
  blind A/B — a useful warning that WER is not the product metric.

---

## 5. Korean (`ooodev/korean-tts`) — when the aligner must change

Not a template for Russian (different writing system), but the best evidence for
**fallback alignment** and for browser packaging pitfalls.

### Data

- Real `Bingsu/KSS_Dataset`, CC BY-NC-SA 4.0: 12,854 clips / 12.86 h, single
  speaker; 8,504 aligned / 8.45 h train (`DATASET.md`). v2 adds 3 speakers and
  18.13 h, 160k steps, CER 5.35 % (`README.md`, "Further training").
- No teacher, no synthetic speech. Numbers/units already expanded in KSS
  transcripts. Split by **sentence**, because KSS records 391 sentences twice;
  a row-wise split triggers `ValueError: sentence leakage between train and
  validation` (`DATASET.md`).

### Frontend

- 142-token unified vocabulary: `dizione` English ids 0–75 (stress intact) plus
  Korean NFD Jamo 76–141; `g2pk` phonology; silent `ㅇ` stripped
  (`FINAL_REPORT.md` §2).
- Code-mixed English is routed through `dizione`; loanwords have a 462-entry
  `data/loanwords.tsv` dictionary (Python → 파이썬) (`README.md`).
- No stress for Korean; stress is 0 for every Korean token.

### Annotations

- `kresnik/wav2vec2-large-xlsr-korean` (1,205 syllable labels), CTC Viterbi at
  **syllable** level, then each syllable span sub-allocated to onset/nucleus/coda
  Jamo using energy + voicing (`nucleus_drop_db = 6`) (`DATASET.md` §3).
- `facebook/mms-1b-all` was rejected: its `kor` vocabulary lists 1,330 symbols but
  the CTC head exposes only 154 classes (`FINAL_REPORT.md` §4).
- QC limits had to be loosened for a 1,205-way softmax: `--max-weak-phone-fraction
  0.6 --min-token-probability 0.3 --max-phone-frames 90`; defaults rejected 70 %
  (`DATASET.md`). This is the key lesson: **QC thresholds are model-specific, not
  language-specific**.
- Rejections: overlong phone 2,000, weak alignment 1,766, low confidence 1,258,
  vowels not voiced 134.

### Training

- 130,000 steps, batch-frames 12,000, max-batch 48, GAN 8000, 11 h 35 on a 4090
  (5.2 GB before GAN, 14.4 GB after), chunked with resume (`FINAL_REPORT.md` §5).
  Vocab 142, one speaker in v1.
- Result: CER 11.8 % vs recorded reference 2.6 % — much weaker than the
  synthetic-data ports, because 8.45 h of one speaker is a hard budget and the
  frontend/aligner were new.

### Integration

- Browser demo runs the actual Python frontend in **Pyodide** (`pyfront.zip`:
  g2pk + Jamo + kofront + dizione). MeCab has no WASM build, so they ship a pure
  Python shim plus a dictionary of the 20,649 training eojeols; the shim matches
  92.64 % of token streams / 99.80 % token accuracy on held-out data
  (`FINAL_REPORT.md` §8).
- ONNX export needed a special `--external-noise` graph for parity; quantisation
  bugs and stale caches are documented in commits (see pitfalls).

### Artefacts

- Demo <https://ooodev.github.io/korean-tts/>; weights in `runs/full/inference.pt`
  and browser assets in `web/public/`; KSS licence restricts use to
  non-commercial research.

---

## 6. Italian (`carbogninalberto/tiny-voca-it`) — real data, 12 voices, neural vocoder

Not our data path, but the most complete **productisation** and the clearest
example of eSpeak-version fragility.

### Data

- Real CML-TTS Italian (CC BY 4.0, LibriVox), 145 h, many speakers. Speaker 1595
  chosen by an objective audit (24.4 h, Whisper WER 0.108, SNR 38 dB); 14 h cap →
  11.52 h train / 0.65 h val / 0.69 h test (`docs/italian.md`, "Data").
- Six synthetic emotional voices (EF1–3, EM1–3) were later added with Qwen3-TTS
  VoiceDesign and Whisper screening, giving 12 voices (`README.md`).
- Filters: transcript/ASR similarity ≥ 0.93, 1–30 s, group by book section to
  avoid leakage. 2.1 % rejected at annotation.

### Frontend

- eSpeak NG **1.51** `it`, 66 symbols = 5 specials + 12 punctuation + 42
  consonants (24 short, 18 long) + 7 vowels (`src/tiny_voca/languages.py`).
- Geminates become **one long token** `Cː`; eSpeak writes them three different
  ways (`l_l`, `ss`, `kː`) and all are canonicalised. Vowel length dropped;
  `ɪ/ʊ→i/u`; open/closed `e/o` kept.
- Stress is a parallel 0/1/2 feature; eSpeak marks are stripped (`languages.py`
  docstring).
- Abbreviations expanded before eSpeak; no English fallback table (unknown phone
  → `<unk>` → clip rejected).
- Browser: `playground/web/frontend.js` is a JS port of the tokenizer; eSpeak NG
  1.51 runs as WebAssembly (npm `espeak-ng` 1.0.2). Parity on 5,594 texts: 0
  mismatches replaying 1.51 output; 98.8 % with 1.52 WASM (68 texts, mostly
  `avesse`).

### Annotations

- Same aligner. Every Italian phone has a label except `dz`, aligned as `dzː`;
  long consonants aligned with their short label. 150-clip pilot: short labels
  gave mean token probability 0.753 vs 0.729 for long labels.
- **Gemination timing failed**: aligned long consonants were not reliably longer
  (`l` 5.4 vs `lː` 3.8 frames) because the recogniser marks onsets and assigns the
  closure to the preceding vowel. They added closure-aware boundary refinement.
  This is directly relevant to Russian: **the aligner marks onsets, so any
  duration contrast that lives in a closure will be mis-assigned unless you
  refine boundaries.**
- Offset kept at English −19 ms; Italian recalibration still pending.

### Training

- 100,000 steps, batch-frames 9000, max-batch 32, GAN 8000/0.7, full precision:
  6 h 15 on an RTX 5070 Ti (9.7 steps/s before GAN, 4.3 after; 4.7 → 9.7 GB).
- WER 7.1 % test / 1.6 % CER; challenge sentences 4.5 %. Real errors were
  geminates and proper names.

### Integration

- Voice/emotion conditioning added to the model (emotion vector ~3k params), and
  the neural vocoder (2.23M) is exported **in the same ONNX graph** (43 MB preview;
  21.4 MB voice-only). Inputs gained `emotion`, `intensity` for emotional models.
- This is the only port that changed the architecture (extra conditioning + a
  vocoder). Everything else kept upstream.

### Artefacts

- First-round checkpoints in `models/italian/`; current 12-voice models not yet
  published (README, "Status"). Demo
  <https://carbogninalberto.github.io/tiny-voca-it/>.

---

## 7. Turkish (`erkinremreta1/zerre-tts`) and clarvia — real vs synthetic at scale

### zerre-tts

- Real data: two speakers, 8,650 clips / 16.01 h after filtering 48→24 kHz
  (`docs/data.md`). 44 phones; **43 are direct aligner labels**, only `g→ɡ` and
  `ɯː→ɯ` mapped (`docs/alignment.md`).
- eSpeak 1.52 produced two surprises on a 744-clip scan: `ɡ` (115×) and `ɯɯ`
  (132×), causing 23/100 clips to be rejected as `unknown_phoneme` until fallbacks
  were added. After the fix: 51 → 66 accepted of 100.
- Upstream QC rejected 34 % of real Turkish clips (`weak_alignment` 26,
  `vowels_not_voiced` 10, `low_confidence` 4 per 100) — again, recalibrate.
- Performance work worth copying: `librosa.pyin` ran at 1.9× real time; YIN at
  271× with a median 0.2 Hz difference on voiced frames, and autocorrelation
  voicing instead of frame energy (flip detection 0.247/0.015 → 0.843/0.884).
  Acoustic prep fell from ~36 h to ~13 min for 8,650 clips (`docs/data.md`).
- Browser: 25k-word lexicon + rules in `web/public/phonemize-tr.js`, int8 ONNX
  8.5 MB. Quantising all ops broke ONNX Runtime Web; they quantise **MatMul only**
  (commit "fix: quantise only MatMul so onnxruntime-web can open the session").
- WER 1.3 % held-out, 12.7 % new sentences; small sets.

### clarvia-tts-tiny (HF)

- 190 h synthetic, **10 voices**, 154,000 aligned clips, 190k steps. 41 h from
  pocket-tts (4 voices) + 150 h from Trendyol-TTS (6 designed voices).
- Own pure-Python phonemizer (`clarvia-phonemizer`), 56 symbols, with text rules
  for phone/account groups, dates, money, units, suffixes after numbers, and
  Turkish stress (final stress, Sezer rule, exception lists); it runs under
  Pyodide.
- WER 2.7 % validation / 2.0 % test / 4.7 % new customer-service sentences.
- Lesson: 190 h and 10 voices are possible with the same 4.5M model, but the
  card explicitly notes inherited teacher mispronunciations; they had to find and
  replace bad clips with Whisper and re-fine-tune the last 40k steps.

---

## 8. Smaller ports, in one paragraph each

- **French (`DanielSuncost/tiny-voca-fr`)**: Qwen3-TTS synthetic, 8,000 Common
  Voice + 1,200 templates, 17,913 clips, 14.1 h after 97.5 % alignment accept,
  100k steps ~6 h, eSpeak `fr-fr`. WER 0 % / 4.5 %. Added an `aperiodicity`
  config to replace part of the male voice's pulse power with noise above 1 kHz;
  eSpeak liaison errors (`comment‿allez`, `fils`→[fil]) forced them to drop
  sentences with `fils` from training. This is a clean example of "fix the data,
  not the model".
- **pt-BR (`SemFence/tiny-voca-ptbr`)**: synthetic Chatterbox, own
  `dizione-ptbr` (55 symbols, MFA-derived lexicon), 104k steps / 8 h on an RTX
  5080, weights CC BY 4.0, browser demo with a TypeScript phonemizer and WebGPU.
  Demonstrates a fully permissive frontend path (no eSpeak), at the cost of
  building a lexicon.
- **mini-voz (Spanish)**: synthetic Qwen3-TTS, 27 h, **4 voices**, 7,000 Common
  Voice sentences, 104k steps ~8 h, eSpeak `es`, WER 2.6 % on 32 unseen
  sentences. Shows that more voices are a data multiplier, not an architecture
  change.
- **apptor (German, HF)**: 12 h synthetic from Qwen3-TTS **built-in voices** (no
  VoiceDesign refs), 15k utterances, eSpeak `de`, 67 symbols, 82k steps / 10 h on
  a 3090, WER 5.0 %. Adds an optional Vocos vocoder fine-tune (13.5M, 54 MB) to
  remove the buzzy source-filter sound — relevant if Russian listeners reject the
  upstream waveform head.
- **Italian emotional voices / Korean v2 / clarvia** all show the same scaling
  law: 4.5M params absorb a few hours per voice; beyond that you need more data
  per speaker or a bigger model, not more steps.

---

## 9. Russian blueprint

### 9.1 Which port to copy

**Primary template: Spanish (`stdevMac/tiny-voca`).** It is the only port that
matches our locked pipeline end to end: Qwen3-TTS VoiceDesign references, Qwen3-TTS
1.7B Base cloning, Whisper large-v3-turbo screening, the same aligner, two voices,
from-scratch 4.5M training, eSpeak frontend, browser demo with an exact JS port.
Copy its `scripts/run_pipeline.sh` / `docs/training-es.md` structure and its
`languages/<lang>.py` profile shape.

**Secondary template: Polish (`damaon/slyszeglosy`)** for the parts Spanish does
not solve: a rule-based frontend with lexical stress, frontend identity
fingerprints, data-driven aligner label mapping, timing calibration, and a
non-eSpeak runtime. Use its `calibrate_aligner_labels.py` and
`calibrate_aligner_timing.py` almost verbatim (change the language profile).

**Teacher-selection lesson from pt-BR Kokoro**: choose the teacher by ear on
Russian test sentences before generating 17k clips. Qwen3-TTS may be the plan,
but the Brazilian experience shows a teacher can be intelligible and still not
sound like the target accent. Budget a 50–200-clip listening bake-off (Polish
`scripts/bakeoff/` is a ready harness).

### 9.2 Frontend, ranked

**Rank 1 — own Russian G2P with a stress lexicon (Polish `wymowa` model).**
Why: voca is a Rust binary; shipping eSpeak means GPL-3.0 code/data and a C
dependency, and the project's F7 acceptance explicitly says "no eSpeak dependency
at runtime". Russian is much more rule-governed than English: letter→phone is
almost regular, with a small set of rules (final devoicing, regressive voicing,
vowel reduction, `ё`, hard/soft pairs, `-тся/-ться`, `что/чтобы`, `-ого/-его`).
The hard part is **stress**, which is lexical and mobile. Plan:
- rules for segmental G2P, with an exception lexicon for loanwords/names;
- a stress lexicon for the top ~50–100k word forms plus a rule-based fallback
  (Russian stress has weak regularities: fixed stem, suffix patterns);
- homograph list (`за́мок/замо́к`, `му́ка/мука́`, `сто́ит/стои́т`);
- `ё` normalised to a stressed `о`; `е` after consonants may be soft;
- stress emitted as the parallel 0/1/2 feature, never a token.
This is 2–4 weeks of work, but it is the only path that keeps the runtime clean
and gives us control over the training token stream. Validate against eSpeak on
1,000 sentences (token parity target ≥ 99 %, F7). Do **not** copy eSpeak's Russian
dictionary into a permissive lexicon — its data is GPL.

**Rank 2 — eSpeak NG `ru` as the single source of truth (Spanish template).**
Fastest to data and exactly aligner-compatible. Use eSpeak 1.52.0 or 1.51, pin it,
record `espeak_version` in the frontend fingerprint, and ship it as a separate
program/ WASM (never link it into the Apache-2.0 binary). The cost is the runtime
dependency and GPL obligations; it also makes F7's "no eSpeak" acceptance
impossible. If chosen, the Rust frontend is a port of the *tokenizer and
canonical map*, with eSpeak still doing G2P (as Spanish `site/frontend.js` does).

**Rank 3 — grapheme input.** Not recommended. Russian has 33 letters, but stress
is unwritten (except `ё`), vowel reduction is stress-dependent, and `е/ё/э`,
`ь/ъ`, `тся` need lexical knowledge. No port uses a pure grapheme frontend;
Korean uses Jamo (a phonemic alphabet), not Hangul syllables, and still needs
`g2pk` phonology.

**Inventory estimate (R5 probe).** I ran the locally installed eSpeak NG 1.52.0
over 16 varied Russian sentences (`-v ru -q --ipa --sep=_`, stress marks
stripped): **49 distinct phone tokens**. The frequent ones were `ʌ i ɑ t n s j u
o r ɪ ɭ v k a y ɭʲ e tʲ p d ʃ m nʲ b ʒ z ts sʲ x ja f vʲ tʃʲ rʲ ɡ dʲ pʲ ɵ mʲ
kʲ ju ɛ ɕ ʑ u" bʲ ɡʲ ə`. After removing eSpeak artefacts (`u"`, `a#`, `u#`),
folding rare allophones, and adding punctuation/specials, expect **~44–48 phones,
~61–65 symbols** — the same size as Spanish (61), Italian (66), German (67).
Keep `ʌ/ə` vs `ɑ/a` distinct if you train on eSpeak output: they encode vowel
reduction, which is a real cue. Do **not** hand-write this list; build it from a
large corpus as Spanish did, because eSpeak leaks internal names (`a#`, `I#`,
`8` for `ё`, `@-`).

**Stress handling.** eSpeak emits `ˈ` (primary) and `ˌ` (secondary); strip both
into the parallel feature exactly as `text.py` does. Russian minimal pairs
(`за́мок/замо́к`) mean the feature must be correct, not just present. If using
eSpeak, trust its stress; if using an own G2P, the stress lexicon is the critical
asset and needs its own test set (≥ 500 homographs/disputed words).

### 9.3 Annotation

Do this before generating 17k clips (F6):

1. **Label coverage.** I checked the aligner's `vocab.json`
   (`facebook/wav2vec2-lv-60-espeak-cv-ft`) against the 49-token probe. It
   contains Russian-specific labels: `nʲ`, `ɭ`, `ɭʲ`, `tʲ`, `dʲ`, `sʲ`, `ɕ`,
   `ʑ`, `ɨ`, `ʐ`, `ʂ`, `ɵ`, `ɡʲ`, `pʲ`, `kʲ`, `bʲ`, `mʲ`, `vʲ`, `fʲ`, `tʃʲ`,
   `rʲ`, `xʲ`, `tsʲ`, `u"`. Missing or mismatched: `zʲ` (map to `ʑ`), `ɕː`
   (map to `ɕ`), `u"` (map to `u`), `a#/u#/I#` (strip). So the coverage is good
   enough to start; this is stronger evidence than the plan's "unproven".
2. **Calibrate the map from data**, not from IPA tables: run Polish
   `calibrate_aligner_labels.py` on 400–500 real Russian clips (or on the first
   teacher batch), then inspect the top-3 labels per phone.
3. **Calibrate the onset offset** (`calibrate_aligner_timing.py`); Polish got
   −10.3 ms, Italian kept −19 ms. Wrong offset shifts every boundary and hurts
   duration learning, especially stops.
4. **QC thresholds**: start from upstream, then set mean token probability and
   weak-phone fraction from the observed distribution. Turkish lost 34 % with
   English thresholds; Korean had to loosen for a 1,205-way softmax. With ~45
   labels Russian should be closer to Polish (0.45 / 0.35 / 0.45), but measure.
5. **Fallback** if accept rate < 70 % or vowel voicing is poor:
   - MFA with a Russian acoustic model + eSpeak dictionary (word/phone boundaries);
   - a Russian CTC/ASR model at character or word level + sub-allocation to
     phones (Korean's trick), if a phone-level model is unavailable;
   - only as a last resort, train a small CTC aligner on the accepted synthetic
     clips and iterate.
   Do not build a new aligner before trying the multilingual one; every eSpeak
   port except Korean succeeded with it.

### 9.4 Data and expected size

The plan's 4,905 + 3,479 sentences × 2 voices = ~16.8k clips. The closest ports:

| Port | Sentences | Clips generated | Accepted | Hours |
|---|---:|---:|---:|---:|
| Spanish | 11,050 | 20,790 | 19,238 train | 22.7 train / 24.5 total |
| Polish | 10,020 | 20,040 | 18,268 train | 18.0 train / 19.2 total |
| pt-BR Kokoro | 8,945 | 17,890 | 15,539 train | 14.78 train / 16.37 total |
| French | 9,200 | ~18,400 | 17,913 | 14.1 |

So **~17k clips / 18–20 h** is the expected outcome for our script count, with
~3–5 % dropped by screening and ~2–5 % by alignment. Two voices double the clip
count but not the sentence count. If the teacher is slower or the screen stricter,
plan for 15 h minimum, which is still above the 8.45 h Korean model and in the
range where the 4.5M model is comfortable.

Screening for Russian: faster-whisper `large-v3` (not `small.en`), `language="ru"`,
with a Russian normaliser for numbers/dates/money; WER ≤ 15 % (upstream) or CER
≤ 5 % (Polish) with a looser threshold on digit-heavy sentences (12 % in Polish);
regenerate once, then drop. Allow one wrong word (pt-BR) to avoid over-rejecting
short sentences. Also check duration (0.5–35 s), speaking rate (8–18 phones/s),
leading/trailing silence, and clipping.

### 9.5 Training

No architecture change. Set `vocab_size` to the Russian inventory, `speakers:
["F", "M"]` (or `F2`/`M2` to mirror the interface), and keep channels 192,
4/2/5 layers, hop 256. Recipe from the ports:

- 100,000–142,000 steps; GAN start 8,000, ratio 0.7, ramp 3,000,
  teacher-hold 3,000 / ramp 7,000 / final 0;
- `--batch-frames 9000 --max-batch 32` (Spanish/Italian/pt-BR) or 12,000/48 if
  VRAM allows (Korean);
- one 4090: 5–9 h; 5070 Ti: 6 h 15; 5080: 8 h; 24 GB card: 8.9 h.
- Expect 2–4× that on the M5 if MPS covers every op; the GAN and the source-filter
  head are the risky ops (plan §6). The R4 result decides.

Evaluation: Whisper large-v3-turbo `ru`, same normaliser for reference and
synthesis, CER and WER, plus a stress-error list and per-category breakdown
(numbers, names, questions). The ports' realistic target is **student ≈ teacher**,
not "better than teacher": Polish 0.37/0.48 vs 0.28/0.52; Spanish 2.00/1.79 vs
1.41/2.09. A 0 % on 20 new sentences is a sanity check, not a benchmark.

### 9.6 Files to change

Python upstream (mirroring Spanish/Polish layout):

| File | Change |
|---|---|
| `src/tiny_voca/languages/russian.py` (new) | inventory, fallback/canonical map, abbreviations, `frontend_version`, ASR language `ru`, TTS language `Russian` |
| `src/tiny_voca/text.py` | select the profile; no logic change if the profile carries the rules |
| `src/tiny_voca/config.py` | `vocab_size` follows the profile; speakers `F2/M2` |
| `src/tiny_voca/align.py` | `ALIGNER_LABEL_FALLBACK` for Russian, alignment version, onset offset |
| `src/tiny_voca/annotate.py` | ASR/QC per profile if thresholds differ |
| `src/tiny_voca/evaluate.py` | `ru` ASR model/language, Russian normaliser |
| `scripts/generate_audio.py` | `language="Russian"`; voice refs |
| `scripts/build_text_*.py` (new) | Russian corpus builder, licence columns |
| `web/` or `site/` | eSpeak WASM + JS tokenizer, or a compiled frontend |

Rust `voca` (this repo):

| File | Change |
|---|---|
| `crates/tiny-voca/src/model.rs` | `SYMBOL_COUNT` is a hard-coded `76`; make it config/profile-driven or add a Russian build feature. The `vocab_size` check must accept the Russian table. |
| `crates/tiny-voca/src/tts.rs` | `Tts` hard-wires `dizione::Phonemizer`; introduce a language frontend trait or a second crate, keep English path byte-identical. |
| `crates/dizione/src/symbols.rs` | English `SYMBOLS: [&str; 76]`; a Russian frontend should not mutate this. New crate `crates/voca-ru-text` is cleaner. |
| new `crates/voca-ru-text` | text → phones + stress, parity tests against the Python frontend (F7) |
| weights/config | new safetensors + config with Russian `vocab_size` and speakers; `scripts/convert_weights.py` conventions |
| CLI/config | `--lang ru`, voice ids `F2/M2` or `F/M`, error messages |

The two Rust hard-coded points (`SYMBOL_COUNT = 76`, `Tts` → `dizione`) are the
real integration cost; neither is large, but both must be done before the model
can ship.

---

## 10. Pitfalls, with sources

1. **eSpeak version drift changes tokens.** Italian trained on 1.51; 1.52 differs
   on 68 of 5,594 texts (mostly `avesse`), and 1.52 is arguably more correct
   (`docs/italian.md`). Spanish refuses to start unless 1.52.0 exactly
   (`README.md`; frontend fingerprint check). pt-BR measured 42/9,013 differences
   (`eval/espeak_versions.json`). → Pin the version, record it in the frontend
   fingerprint, and use the *same* version for annotation, training and inference.
2. **eSpeak 1.51 drops the last character on unterminated stdin.** Upstream's
   `text.py` documents versions 1 and 2 producing wrong final phonemes; the fix is
   to append `\n` (`src/tiny_voca/text.py` docstring). Short Russian utterances
   would hit this constantly.
3. **Aligner labels are not today's eSpeak output.** Polish needed 19 mappings and
   gained 0.52 → 0.63 phone probability (`scripts/calibrate_aligner_labels.py`);
   pt-BR needed 10 (`README.md`); Turkish found `ɡ` and `ɯɯ` surprises
   (`docs/alignment.md`). → Build the map from data.
4. **The aligner marks onsets, not closures.** Italian's geminate durations came
   out *shorter* for long consonants (`l` 5.4 vs `lː` 3.8) because the closure is
   assigned to the previous vowel (`docs/italian.md`, G3). → Calibrate the offset
   and add closure-aware refinement if Russian has contrasts that depend on it.
5. **QC thresholds are model-specific.** Turkish rejected 34 % of real clips with
   upstream thresholds (`docs/alignment.md`); Korean had to use 0.6/0.3/90 for a
   1,205-way softmax (`DATASET.md`). → Measure the distribution before fixing
   thresholds.
6. **Synthetic teachers have systematic flaws.** Spanish: flat prosody (2.0 st,
   ~3–4 expected) and word-edge roughness from the source-filter head, fixed at
   inference with mixed excitation (`docs/model-card/README-es.md`;
   `docs/word-edge-artifact.md`). French: male-voice buzz, fixed with
   `aperiodicity`, plus eSpeak liaison errors (`README.md`). Clariva: inherited
   teacher mispronunciations, requiring clip replacement and a 40k-step re-finetune
   (HF card). → Listen to 200 clips before full generation; screen, then re-screen.
7. **Teacher choice is not just WER.** pt-BR Kokoro rejected Qwen3-TTS/Chatterbox
   clones because they did not sound Brazilian, and a Magpie fine-tune with better
   WER lost the blind A/B (`README.md`; `eval/listening.json`). → Ear test first,
   WER second.
8. **The teacher and student may not share a G2P.** Kokoro uses eSpeak, so
   teacher/student pronunciation agrees; Qwen3-TTS does not. Spanish/Polish accepted
   this, but it means the student learns the teacher's phones, not eSpeak's
   intended ones. For Russian, compare eSpeak tokens with what the teacher actually
   says on 100 clips (ASR + listening) before scaling.
9. **Numerals and dates are the most common "errors".** Italian's remaining WER was
   mostly number/currency formatting; Spanish leaves numbers to eSpeak and keeps a
   separate reader only for scoring; French writes all numbers out in the corpus;
   pt-BR has a full normaliser. → Build the Russian normaliser before the corpus,
   and use the same one for screening and evaluation.
10. **Unicode decomposition matters.** pt-BR stored composed `õ` and silently lost
    nasality; storing NFD fixed it (`README.md`). Russian `ё`, combining acute
    stress marks and `й` need the same care.
11. **Browser frontends are a packaging project.** Polish loads `wymowa` through
    Pyodide; Korean ships `pyfront.zip` and hit missing `loanwords.tsv` /
    `kofront` modules, plus a PowerShell CP949 mojibake bug (commits). → Add a
    parity test and a cache-version bump to every asset change.
12. **ONNX Runtime Web has op gaps.** Turkish had to quantise MatMul only because
    ConvInteger broke the session, and stale caches served the old model (commits).
    Italian ships a 43 MB combined voice+vocoder graph; Spanish/French add
    inference-only excitation options. → Test the exact ORT-web build.
13. **Audio can peak above 1.0.** Spanish and Polish both scale to 0.99 before
    writing 16-bit; Spanish measured 1.02 on one clip. → Always peak-normalise.
14. **Sentence-level leakage.** All ports split by sentence; Korean's KSS has 391
    duplicate sentences and a row-wise split fails loudly (`DATASET.md`). → Split
    by sentence, both voices together.
15. **`pyin` is the preprocessing bottleneck.** Turkish measured 1.9× real time;
    YIN at 271× with ~0.2 Hz median difference, plus autocorrelation voicing, cut
    36 h to 13 min (`docs/data.md`). → For 17k clips on an M5 this matters; shard
    processes (Polish `--workers`, Korean 4 shards) and consider YIN.
16. **Long text needs sentence splitting and a token cap.** Every port reads one
    sentence per call (512 tokens / 32 s) and joins audio; `speak_text`/`speak`
    do this. → Reuse the splitter, don't feed paragraphs to the model.

---

## 11. Unknown / could not verify

- **Qwen3-TTS Russian quality and VoiceDesign Russian descriptions.** No port in
  this survey used Russian. The plan asserts official `ru` support; I did not run
  the teacher or listen to output. This is the project's main quality risk.
- **Aligner quality on Russian audio.** I verified that the aligner's vocabulary
  contains the Russian labels we saw, but not that CTC alignment accepts them at a
  useful rate. F6 must measure accept rate and vowel voicing on 200 clips.
- **Russian eSpeak inventory stability and full coverage.** My probe is 16
  sentences and one eSpeak build (1.52.0, Homebrew). The true inventory must come
  from a large corpus; eSpeak 1.51 may differ. I did not compare 1.51 vs 1.52 for
  Russian.
- **Stress lexicon licensing.** No permissively licensed Russian stress dictionary
  was evaluated here. Wiktionary-derived data, eSpeak's GPL data, and academic
  stress corpora all have different terms; this needs a licence review before F3/F7.
- **Russian text corpora.** Common Voice Russian sentence availability, PD authors,
  and LLM-written text were not surveyed here (R2 owns this).
- **M5/MPS training.** No port reports MPS numbers; all training was CUDA. The
  plan's 2–4× estimate is an assumption until R4 tests the GAN/source-filter ops.
- **A true Rust TTS port.** I found no Rust reimplementation of tiny-voca. The
  closest integration is `ilyaizen/CopySpeech` (Tauri), which ships a Python
  wrapper + ONNX and keeps the model in a sidecar (`scripts/tiny-voca/
  copyspeech-tiny-voca.py`); it does not port the frontend. This confirms that
  voca's Rust frontend is genuinely new work, not something to copy.
- **Hugging Face dataset cards for clarvia/apptor** were read as model cards only;
  dataset licences beyond the stated CC0 text / teacher licences were not opened.
- **Formal MOS / listening studies.** None of the ports ran one. The only
  listening number is pt-BR's single listener (4/5); treat all quality claims as
  ASR-based sanity checks.

---

## Appendix: R5 Russian eSpeak probe

Local `espeak-ng` 1.52.0, 16 sentences, command:

```sh
espeak-ng -v ru -q --ipa --sep=_ < ru_sample.txt
```

49 distinct phone tokens after stripping `ˈ`/`ˌ` and splitting on `_`/space:
`ʌ i ɑ t n s j u o r ɪ ɭ v k a y ɭʲ e tʲ p d ʃ m nʲ b ʒ z ts sʲ x ja f vʲ tʃʲ
rʲ ɡ dʲ pʲ ɵ mʲ kʲ ju ɛ ɕ ʑ u" bʲ ɡʲ ə`.

Observed artefacts to canonicalise: word-final `a#`/`u#`/`I#`, `@-` for a reduced
vowel + glottal-ish boundary, `8` for `ё`, doubled `j_j`/`d__` from spelling,
and `_` word boundaries. Aligner `vocab.json` covers all of these except `zʲ`
(use `ʑ`), `ɕː` (use `ɕ`), `u"` (use `u`) and the `#` variants (strip).
