# F6 — can the upstream aligner label Russian eSpeak phones?

**Verdict: usable with a label mapping.** `facebook/wav2vec2-lv-60-espeak-cv-ft`
aligns eSpeak-ru phone sequences against Russian audio at a 97.7 % accept rate
(128/131 clips) with no unalignable phones: every one of the 71 phones that
eSpeak 1.52 `ru` emitted over 775 sentences has a label in the model. Seven
Russian phones need a different label than eSpeak prints, two more (`zʲ`, `ɕː`)
are mapped defensively, and the foreign tail from Latin text is mapped too (all
decided below). Four phones align weakly (`ja ju j ɛ`) but only for the
glide/onset boundary, not for the token's total duration.

Fallbacks (Russian CTC model, MFA + Russian dictionary) are **not needed**.

- Mapping contract for F3/F7: [`docs/aligner-mapping.md`](../docs/aligner-mapping.md)
- Code: [`scripts/align_spike/`](../scripts/align_spike/) (rerunnable from a clean checkout)
- Raw results: `.cache/align-spike/alignment-{fleurs,silero,fleurs-v2,fleurs-v3}.json` (git-ignored)

## 1. What was measured

| | |
|---|---|
| Aligner | `facebook/wav2vec2-lv-60-espeak-cv-ft`, commit `ae45363b` |
| eSpeak | 1.52.0 (Homebrew), `-v ru -q --ipa --sep=_` |
| Audio set A | FLEURS `ru_ru` **test**, 131 clips / 1477 s (24.6 min), real human read speech, 16 kHz mono, CC BY 4.0 |
| Audio set B | 71 clips / 245 s from `out/silero/` (Russian TTS, 24 kHz) — **test material only**, Silero is excluded as a generator |
| Alignment | upstream CTC Viterbi (`tiny_voca.align` logic): 20 ms frames at 16 kHz, −19 ms onset offset, durations on the 24 kHz / hop-256 grid |
| Voicing | cheap autocorrelation detector (proxy for upstream's `librosa.pyin`), 1024-sample window, same grid |

Set A is the honest test: real speech, spoken transcripts. Set B is the domain
proxy for the production corpus (synthetic teacher audio).

## 2. Numbers

### FLEURS ru test — real speech, 131 clips

| Metric | mean | p10 | p50 | p90 |
|---|---:|---:|---:|---:|
| clips aligned | **131/131** (0 failures) | | | |
| token coverage (phone spans / audio) | 1.000 | 1.000 | 1.000 | 1.000 |
| mean token probability | **0.668** | 0.617 | 0.676 | 0.721 |
| weak phone fraction (<0.1) | 0.117 | 0.062 | 0.107 | 0.172 |
| one-frame phones | 0.017 | 0.000 | 0.015 | 0.031 |
| phones/second | 10.43 | 8.15 | 10.61 | 12.20 |
| vowel frames voiced | 0.627 | 0.476 | 0.673 | 0.768 |
| voiceless-consonant frames voiced | 0.317 | 0.239 | 0.311 | 0.412 |

- **Accept rate with the Polish port's calibrated thresholds** (mean token
  probability ≥ 0.45, weak fraction ≤ 0.35, vowel voicing ≥ 0.45):
  **128/131 = 97.7 %**. Rejections: 2 `vowels_not_voiced` (0.447, 0.416 — both
  just under the line), 1 `low_confidence` + `weak_alignment` (clip
  `12726640944874773003`, probability 0.157; a quote-heavy sentence whose audio
  does not match the eSpeak reading, not a mapping failure).
- 131 clips, 117.2 phones per clip on average (max 317).
- **Stressed vowels: 2098 occurrences, mean 11.1 frames (118 ms), p5 5 frames
  (53 ms), exactly 1 occurrence at 1 frame.** No zero-length stressed vowels.
  All phones average 9.0 frames.
- 210 of 2098 stressed vowels (10 %) fall below the 0.45 voicing proxy. Read
  this as an upper bound, not a verdict: the proxy scores voiceless consonants
  at 0.32, so its separation is weak. The real gate must use `librosa.pyin` (or
  the port's autocorrelation voicing) with a threshold re-measured on generated
  audio.

### Silero TTS — synthetic domain proxy, 71 clips

| Metric | value |
|---|---:|
| clips aligned | 71/71 |
| mean token probability | 0.706 |
| weak phone fraction | 0.082 |
| phones/second | 11.68 |
| vowel frames voiced | 0.714 |
| stressed vowels short (<2 frames) | 0 / 469 |
| accept rate (same thresholds) | 68/71 = 95.8 % (3 × `vowels_not_voiced`, 0.386–0.438) |

Synthetic speech aligns slightly better than human speech, which is the
direction that matters: the training corpus will be synthetic.

### Per-phone outcomes

63 of the 71 inventory phones occurred in the 131-clip run (full table with
counts, mean duration and probability: `docs/aligner-mapping.md` §3).

Consonants are strong: `p .92, nʲ .91, ʒ .91, r .90, k .89, rʲ .88, m .85,
v .85, n .82, s .82, z .83, t .81, ɭʲ .80, b .79, mʲ .79, d .75, ts .75, x .74,
ɡ .74, kʲ .74, sʲ .71, tʲ .68, tʃʲ .65, ɕ .66, vʲ .65, pʲ .63, ɭ .59, bʲ .58`.

Weak phones (mean probability over their spans):

| phone | prob | n | what the model does | decision |
|---|---:|---:|---|---|
| `ju` | 0.07 | 74 | blank on 33/62 spans | keep `ju` |
| `ja` | 0.12 | 156 | blank on 50/135 spans | keep `ja` |
| `ɛ` (э) | 0.16 | 114 | split `ʌ` 23 / `e` 21 / `ɛ` 17 | map to `e` (weak evidence) |
| `j` | 0.19 | 469 | blank on 330/398 spans | keep `j` |
| `fʲ` | 0.22 | 13 | mean span 379 ms — a few mis-matched spans | keep `fʲ` |
| `u"` | 0.26 | 25 | `u` is the better label | **map to `u`** |
| `ɡʲ` | 0.31 | 42 | `ɡʲ` is still top-1 | keep `ɡʲ` |
| `ʑ` | 0.32 | 43 | `ʑ` top-1 19/39 | keep `ʑ` |
| `y` (ы) | 0.37 | 538 | `y` top-1 215/453, `ʌ` 126 | keep `y` |

The four "adversarial" phones from the task all behaved:

- `lʲ` isn't emitted by eSpeak 1.52 ru at all — it prints `ɭʲ` (soft л), which
  aligns at 0.80.
- `zʲ` did not occur in 775 sentences (0 tokens); mapped to `ʑ`, unverified.
- `ɕː` did not occur either (щ is printed `ɕ`); mapped to `ɕ`, unverified.
- `ɨ` is not emitted: ы comes out as `y` (2905 tokens) and aligns to `y`.
- `ɵ` (ё) is emitted 145 times and the model does not have it as a usable
  phone: `o` scores −1.30 vs `ɵ` −6.38, `o` top-1 on 21/29 spans. Mapped.
- `ɡʲ` is a real label, top-1 on 19/33 spans.
- Stress marks are stripped into the 0/1/2 feature, never aligned.

### Mapping decisions (the contract)

Nine eSpeak phones are labels but are *not* what the model hears; each move was
chosen by mean log-probability over real spans, not by IPA reasoning:

| eSpeak | → label | evidence |
|---|---|---|
| `ɵ` | `o` | −1.30 vs −6.38, 21/29 top-1 |
| `ə` | `ʌ` | −0.86 vs −5.23, 31/39 |
| `ɪ` | `i` | −1.50 vs −2.42, 346/582 |
| `a` | `ʌ` | −0.92 vs −2.08, 322/393 |
| `ɛ` | `e` | −3.26 vs −3.46 (tie, low confidence) |
| `u"` | `u` | −2.01 vs −4.12 |
| `ɪ^` | `i` | −1.70 vs −10.07, but n=1 on audio |
| `zʲ`, `ɕː` | `ʑ`, `ɕ` | absent from today's output; IPA substitution (Polish precedent) |
| foreign tail (`θ ð ɹ ɻ ɒ ɔ …`) | nearest Russian phone | each mapped, never left unaligned |

Splitting `ja`/`ju` into `j`+vowel was **tested and rejected**: probability fell
from 0.119 to 0.036 and 0.065 to 0.038 (runs `alignment-fleurs-v2` vs
`alignment-fleurs`), because the model emits blank across the glide either way.
Consequence: я/е/ю/ё lose the observable j→vowel boundary inside the token; the
token's own duration is still bounded by its neighbours.

## 3. Timing and cost

| | MPS (Apple M5 Pro) | CPU |
|---|---:|---:|
| model load | 3.2 s | 2.0 s |
| per clip (11.3 s average) | 0.15 s | 0.41 s |
| real-time factor | 0.014 | 0.045 |

At CPU RTF 0.045, the full ~17 k-clip corpus aligns in **~2 h single-process**
(MPS: ~45 min); shard it like the ports do. Memory: 1.26 GB fp32 model.

## 4. Downloads made

All under `/Users/suheugene/projects/22-tiniest-voca-ru/.cache/models/`.

| File | Size | Integrity |
|---|---:|---|
| `wav2vec2-lv-60-espeak-cv-ft/pytorch_model.bin` | 1,263,535,127 B | SHA-256 `3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e` — matches the HF API LFS hash; fetched as 24 parallel ranges with `scripts/fetch-hf-file.sh` |
| `.../config.json`, `preprocessor_config.json`, `vocab.json`, `tokenizer_config.json`, `special_tokens_map.json` | 1.8–4.6 KB each | plain HTTPS |
| `fleurs-ru/test.tsv` | 735,258 B | plain HTTPS |
| `fleurs-ru/audio/test/*.wav` | 131 files | extracted from a **byte-range** fetch of `data/ru_ru/audio/test.tar.gz` (range 0–146,800,639 of 433,142,634 B; 72 MB had arrived when the spike was run). A range fetch has no whole-file checksum; gzip/tar decoded without error and every extracted WAV opens with `soundfile`. |

`espeak-ng` 1.52.0 was already installed (Homebrew); nothing was installed or
upgraded for this task. Nothing outside `.cache/` and this worktree was written;
no file owned by another task was touched or deleted.

## 5. Reproduce

```sh
. scripts/cache-env.sh
.cache/venvs/r4/bin/python scripts/align_spike/calibrate.py --clips 131 --device mps --tag fleurs
.cache/venvs/r4/bin/python scripts/align_spike/calibrate.py --silero --clips 71 --device mps --tag silero
```

(The cache root is found via `HF_HOME`, so the script works from a worktree.)

## 6. What I could not verify

- **`zʲ` and `ɕː` on audio** — eSpeak 1.52 never emitted them in 775 sentences.
  The `ʑ`/`ɕ` substitutions are IPA- and precedent-based, not measured.
- **`ɪ^`** — 28 occurrences in text, 1 on audio. Its `→ i` choice is thin.
- **The voicing check** is an autocorrelation proxy, not upstream's `librosa.pyin`
  (pyin is ~2× real time; not worth it for a spike). 10 % of stressed vowels
  look unvoiced by this proxy; whether that is a real QC problem must be
  re-measured with pyin on the generated corpus, where clip-level QC actually
  decides.
- **Alignment on Qwen3-TTS audio** — not generated yet. Silero TTS is the
  closest available proxy and scored better than real speech, which is
  encouraging but not a substitute.
- **Onset-offset calibration** — the upstream −19 ms was kept, as Italian did.
  Polish found −10.3 ms on its data. Re-run
  `calibrate_aligner_timing.py`-style calibration once the teacher corpus exists;
  a wrong offset shifts every boundary, though it cancels out of individual
  phone durations only if it is constant.
- **Punctuation/pause tokens** — the spike aligns phones only. Upstream's
  silence→pause-token logic is language-independent code but was not exercised
  here.

## 7. What this means for the plan

- Gate "F6 alignment accept rate measured" (§10 of `docs/contracts.md`) is
  satisfied: **97.7 %** on real Russian speech, **95.8 %** on Russian TTS.
- D3 can annotate with the upstream aligner as-is plus `mapping.py`; no Russian
  acoustic model and no MFA run is required. Build the mapping into
  `training/annotate/` behind a `ru` profile and keep the label map under test.
- QC thresholds should start at the Polish values and be re-fitted on the first
  1 000 generated clips, as the ports did. Expect `ja/ju/j` to be flagged by any
  weak-phone metric and decide whether that is acceptable or whether the
  frontend should drop those phones from the weak-phone denominator.
- The only fallback worth keeping in reserve is the one the ports describe:
  if accept rate on real teacher audio comes in under ~90 %, run MFA with a
  Russian dictionary, or sub-allocate phone spans from a Russian CTC model.
  Nothing in this measurement suggests that will be needed.
