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
the [tiny-voca](https://github.com/n4ze3m/tiny-voca) recipe. Voices: `F2` (female)
and `M2` (male).

Weights only: the model takes phoneme ids and stress and returns a waveform. Text
has to be turned into ids first — see [Frontend](#frontend).

## Files

The main files are the +60k checkpoint (104 000 steps of the base run plus 60 000
of adversarial fine-tuning with a strengthened discriminator). The two earlier
checkpoints sit in `variants/` for comparison; on a listening test the difference
between them is barely audible, so pick whichever you like.

| file | size | what |
|---|---|---|
| `voca-ru.onnx` | 21.4 MB | ONNX graph, opset 17, one utterance in, waveform out |
| `voca-ru.onnx.json` | 1 KB | sample rate, hop, voices, 76-symbol inventory, input/output names |
| `gguf/voca-ru-js.gguf` | 23.5 MB | float32 GGUF in the `audiocpp` layout, tensors named `model/*` |
| `inference.pt` | 18.2 MB | PyTorch checkpoint the two exports came from |
| `variants/104k/` | 63 MB | the base run, 104 000 steps |
| `variants/134k/` | 63 MB | base + 30 000 fine-tune steps |
| `samples/` | | one phrase per voice |

Every file carries the same 184 tensors / 4 537 291 parameters.

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
relative waveform difference **0.0017** (float32 rounding). The shipped graph draws
its own noise; export with `--external-noise` only if you need reproducible audio.

## GGUF

The layout follows `n4ze3m/tiny-voca`: GGUF v3, float32 tensors under
`model/<attribute>`, an embedded `config.json` with `model` and `vocab`, and the
Russian front-end tables under `phonemizer/`. `js/tiny-voca.js` reads it with no
adapter; lengths and durations match PyTorch exactly.

```js
import { readGGUF, TinyVoca } from "./tiny-voca.js";
const file = readGGUF(await (await fetch(url)).arrayBuffer());
const voca = new TinyVoca(file);
const { audio } = voca.synthesize(ids, stress, { voice: "M2", seed: 0 });
```

Shortest end-to-end check, using the Russian frontend from the `original` branch:

```sh
node js/say.mjs "Мой коллега уходит в отпуск на две недели." \
  --file gguf/voca-ru-js.gguf --output out.wav
# 2.45 s of audio in 0.25 s, 9.9× real time
```

The file also carries `phonemizer/stress_dict.tsv` (100 000 forms) and
`phonemizer/g2p_lexicon.tsv` (14 492 entries) for a Russian frontend in the
browser. The English `dizione` engine shipped with tiny-voca cannot read this
model's symbols.

## Frontend

The model speaks a 76-symbol Russian inventory: normalisation (numbers, dates,
money, «ё», Latin) → RUAccent stress, marked as U+0301 → eSpeak NG `ru` phonemes →
ids. The Python implementation lives in the training repository
(`training/voca_ru/text`), the Rust one in `crates/voca-ru-text`; both produce
identical token ids across the 8 000-sentence corpus.

## Training

Synthetic corpus: 19 217 clips / 20.35 h from Qwen3-TTS-1.7B (VoiceDesign
references + Base cloning) with a Russian stress LoRA, screened by Whisper
(WER ≤ 15 %) and by a stress detector (meter2). 18 148 clips survived forced
alignment. The base model trained 104 000 steps (GAN from step 8 000) on one
RTX 4090 in 6 h 50 min.

The shipped checkpoint adds 60 000 steps with a stronger discriminator
(`gan_ratio` 0.7 → 1.1, `feature_matching` 30 → 45, adversarial from step 0).

| metric | 104k | 134k | 164k |
|---|---|---|---|
| stft loss | 1.0728 | 1.0591 | 1.0585 |
| mel loss | 0.4254 | 0.4175 | 0.4166 |
| pitch error, semitones | 1.486 | 1.380 | 1.358 |
| voicing accuracy | 0.9633 | 0.9642 | 0.9646 |

Round-trip WER, Whisper large-v3-turbo on CPU, 40 held-out test sentences in both
voices (80 clips):

| checkpoint | WER | CER |
|---|---|---|
| 104k | **0.45 %** | 0.29 % |
| 164k | 0.68 % | 0.34 % |

The two are within noise of each other (4 versus 6 word errors out of ~880 words),
so the fine-tune bought spectral accuracy, not transcription accuracy.

## Known limitations

- **The female voice is bright.** On the same text its 2–8 kHz energy sits
  1.8 dB above the teacher's (2–4 kHz +1.3, 4–8 kHz +1.8), which reads as a
  metallic timbre. The male voice sits 1.4–3.7 dB *below* the teacher and sounds
  closer to natural.
- **Both voices are pitched up** by 2–3 semitones relative to the teacher
  (F2 240 Hz vs 207 Hz, M2 126 Hz vs 108 Hz). `pitch_shift = -0.18` and `-0.13`
  bring them back exactly, and are the recommended defaults.
- **60 000 extra steps did not fix the timbre.** They moved the female 2–8 kHz
  excess by 0.7 dB, which is at the edge of audible. The next lever is model
  capacity (192 → 256–320 channels), not more adversarial training.
- Unknown words fall back to rules; the shipped lexicon covers 94.6 % of
  sentences in the training corpus.
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
