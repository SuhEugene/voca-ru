---
library_name: voca
license: apache-2.0
language:
  - ru
tags:
  - text-to-speech
  - tts
  - russian
  - onnx
  - gguf
  - tiny-voca
---

# voca-ru

A 4.5M-parameter two-voice Russian text-to-speech model, trained from scratch with
the [tiny-voca](https://github.com/n4ze3m/tiny-voca) recipe.

**Current version (root of this repo) is v2**: a new warm female narrator voice and
a corpus regenerated with context-resolved stress homographs.

| voice | character | F0 in the model's output |
|---|---|---|
| `F2` | warm female narrator (`s3_lower_register`), 220 Hz clone | 209–235 Hz |
| `M2` | male, unchanged since v1 | ~126 Hz |

Weights only: the model takes phoneme ids and stress and returns a waveform. Text
has to be turned into ids first — see [Frontend](#frontend).

## Files

| file | size | what |
|---|---|---|
| `voca-ru.onnx` | 21.4 MB | ONNX graph, opset 17, one utterance in, waveform out |
| `voca-ru.onnx.json` | 1 KB | sample rate, hop, voices, 76-symbol inventory, input/output names |
| `gguf/voca-ru-js.gguf` | 27.3 MB | float32 GGUF in the `audiocpp` layout, tensors named `model/*`, Russian front-end tables embedded |
| `inference.pt` | 18.2 MB | PyTorch checkpoint the two exports came from, step 104 000 |
| `variants/104k/` | 63 MB | previous voice set, base run, 104 000 steps |
| `variants/134k/` | 63 MB | previous voice set, base + 30 000 fine-tune steps |
| `variants/164k/` | 63 MB | previous voice set, base + 60 000 fine-tune steps |
| `samples/` | | one phrase per voice, `plain`, `question`, `number`, `date`, `name` |
| `rust/voca-data.tar.gz` | 18.3 MB | runtime bundle for the Rust app (`voca`): `tiny-voca.safetensors` + `phonemizer/` with the same 150 k / 53 710 tables |

Every file carries the same 184 tensors / 4 537 291 parameters.

## Versions

| | v2 (root) | v1 (`variants/`) |
|---|---|---|
| female voice | `s3_lower_register`, warm narrator | `f_neutral` (bright, metallic — see limitations) |
| male voice | `m_neutral` | `m_neutral` |
| train clips / hours | 16 764 / 18.17 h | 18 057 / 20.35 h |
| stress homographs | resolved by context, corpus regenerated | single reading per form |
| test WER (80 clips) | 1.45 % | 0.45–0.68 % |
| test CER (80 clips) | 0.37 % | 0.29–0.34 % |

The v2 test split is disjoint from training (all validation/test sentences were
regenerated in the new voice and never appear in train), while the v1 test split
overlapped the general sentence pool, so the two WER numbers are not directly
comparable.

## ONNX

```python
import numpy as np, onnxruntime as ort

session = ort.InferenceSession("voca-ru.onnx", providers=["CPUExecutionProvider"])
feed = {
    "tokens": np.array([ids], dtype=np.int64),        # phoneme ids, <bos> … <eos>
    "stress": np.array([stress], dtype=np.int64),     # 0/1 per id, same length
    "voice": np.array([0], dtype=np.int64),           # 0 = F2, 1 = M2
    "speed": np.array([1.0], dtype=np.float32),
    "pitch_shift": np.array([0.0], dtype=np.float32), # log-F0 offset, 0.058 ≈ one semitone
    "pitch_range": np.array([1.0], dtype=np.float32),
}
audio, durations = session.run(None, feed)            # float32 [1, samples] at 24 kHz, frames per id
```

Verified against PyTorch for the same seeded noise: identical durations, maximum
relative waveform difference 0.0017 (float32 rounding).

## GGUF

The layout follows `n4ze3m/tiny-voca`: GGUF v3, float32 tensors under
`model/<attribute>`, an embedded `config.json`, and the Russian front-end tables
under `phonemizer/`. `js/tiny-voca.js` reads it with no adapter.

```sh
node js/say.mjs "Мой коллега уходит в отпуск на две недели." \
  --file gguf/voca-ru-js.gguf --output out.wav
```

The file carries `phonemizer/stress_dict.tsv` (150 000 forms) and
`phonemizer/g2p_lexicon.tsv` (53 710 entries), rebuilt for this release from the
product corpus plus the 46 091-sentence Common Voice pool.

### Stress homographs in the shipped dictionaries

RUAccent stores one reading per form (`accents.json`: `замок → з+амок`) and keeps
the second reading only in a separate omograph table. Earlier releases shipped the
first reading blindly. The dictionaries in this release carry the reading the
omograph model chose most often on the training corpus (765 homograph forms were
decided this way, e.g. `замок → замо́к`, `дела → де́ла`). The runtime dictionary is
flat, so it still has one reading per form: a phrase where the corpus majority is
the wrong sense (`за́мок` = castle) will be marked wrong. The training pipeline and
the Python frontend resolve per sentence; the browser/Rust frontends do not.

## Frontend

Normalisation (numbers, dates, money, «ё», Latin) → RUAccent stress with a
combining acute → eSpeak NG `ru` phonemes → 76-symbol ids. Python implementation in
`training/voca_ru/text`, Rust one in `crates/voca-ru-text`; both produce identical
token ids across the corpus.

## Training

Synthetic corpus from Qwen3-TTS-1.7B (VoiceDesign references + Base cloning) with a
Russian stress LoRA, screened by Whisper (WER ≤ 15 %) and a stress detector. v2:
19 822 clips generated (95.0 % passed screening), 16 764 clips / 18.17 h survived
forced alignment (93.1 %), trained 104 000 steps on one RTX 4090 in ~4 h, $4.57
total including corpus generation.

| metric (v2, step 104 000) | value |
|---|---|
| stft loss | 1.0808 |
| mel loss | 0.4259 |
| pitch loss (normalised, not semitones) | 0.084 |
| voicing loss | 0.084 |
| WER / CER, 40 held-out sentences × 2 voices | 1.45 % / 0.37 % |

Earlier releases quoted pitch error in semitones (1.36 at v1 164k); that number
came from the evaluation summary, not from these training losses, so the two are
not comparable.

## Known limitations

- **v2 female timbre has not been evaluated by ear yet.** Samples are in `samples/`.
  The previous female voice sat 1.8 dB above the teacher in 2–8 kHz (metallic);
  the new reference voice was chosen to fix that, but no listening test or LTAS
  measurement of v2 is published here yet.
- **Pitch mapping is untuned for v2.** The model's female output measures
  209–235 Hz against a 220 Hz clone, so `pitch_shift` may not be needed; the v1
  recommendation (−0.18 for F2, −0.13 for M2) belongs to the old voice.
- Homographs: one reading per form at runtime (see above).
- Unknown words fall back to letter rules. Measured coverage: on the product corpus
  the lexicon knows 99.8 % of word tokens and the stress dictionary 88.7 %; on
  held-out FLEURS Russian (never used to build them) 79.9 % and 70.7 %. The rest
  gets rule-based stress (penultimate vowel) and letter-to-sound phones, which is
  audible but intelligible.
- A dictionary covering **all** 3.19 M RUAccent forms would lift held-out stress
  coverage to 88 % but weighs 153 MB, so it is not shipped; the 150 k selection is
  ranked by frequency over real Russian text.
- No streaming, one utterance per call, 32 s maximum.

## Licence and provenance

- Teacher: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and `VoiceDesign` (Apache-2.0), stress
  adapter `siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf` (Apache-2.0).
- Corpus: synthetic, released as
  [`SuhEugene/voca-ru-corpus`](https://huggingface.co/datasets/SuhEugene/voca-ru-corpus)
  (CC0); source sentences from Mozilla Common Voice (CC0) and public-domain texts.
- Stress dictionary: RUAccent (MIT). The phoneme tables derive from eSpeak NG
  (GPL-3.0); no eSpeak code runs at inference, but the licence status of the
  derived lexicon is unresolved, which matters before commercial use.
- Architecture and training code: `n4ze3m/tiny-voca` (Apache-2.0).

Released for personal and research use.
