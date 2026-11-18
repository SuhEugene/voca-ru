# tiny-voca / voca repository map (R1)

Scope: the Rust inference repository in this worktree (`voca/`), the Python
training stack it was forked from (`n4ze3m/tiny-voca`, cloned as `upstream/`),
and the phonemizer (`n4ze3m/dizione`, cloned as `dizione/`).

Read at:

| Tree | Revision | Date | Notes |
|---|---|---|---|
| `voca/` (this worktree) | `8cc7d11` (`task/recon`) | 2026-11-19 | Rust rewrite; Python training code removed by `c7de4a6` |
| `upstream/` (`/tmp/tiny-voca-upstream`) | `e7f1bde` | 2026-11-18 | Python training + inference + JS port |
| `dizione/` (`/tmp/dizione-upstream`) | `8cd889e` | 2026-11-18 | Python phonemizer |

Citation convention: paths without a prefix are this repository; `upstream/...` and
`dizione/...` are the clones above. Paths such as `crates/tiny-voca/src/model.rs`
are relative to the repository root. Line numbers are from those exact
revisions.

---

## 0. Contradictions with the project plan (read this first)

The plan assumed this could be an STT project or an arbitrary “voice” project.
It cannot, as it stands:

1. **This is English TTS only, not STT.** There is no audio input anywhere in
   the model or CLI. `Tts::speak` accepts text and returns samples
   (`crates/tiny-voca/src/tts.rs:40-61`); `voca` reads text or stdin
   (`crates/tiny-voca/src/bin/voca.rs:83-92`). The model’s only inputs are
   phoneme ids, stress and a voice index (`crates/tiny-voca/src/model.rs:423-446`).
2. **No arbitrary voice / no voice cloning.** Two fixed speakers, `F2` and
   `M2`, are baked into the checkpoint config (`upstream/src/tiny_voca/config.py:22`;
   downloaded `config.json` has `"voices": ["F2", "M2"]`). The model has a
   2-row voice embedding (`crates/tiny-voca/src/model.rs:384-386`), not a
   speaker encoder.
3. **The weights are not compiled into the binary.** This corrects the prior
   context. `weights.rs` is only a safetensors reader
   (`crates/tiny-voca/src/weights.rs:1-67`); `voca` loads
   `$VOCA_HOME/tiny-voca.safetensors` from disk
   (`crates/tiny-voca/src/bin/voca.rs:97-106`), and `install.sh` downloads
   a 19 MB `voca-data.tar.gz` in addition to the binary
   (`install.sh:19-39`). The “one binary” claim is really “binary + data
   directory”.
4. **The Rust repository has no training code.** It was removed in
   `c7de4a6` and `NOTICE` says so. Any Russian model must be trained with the
   upstream Python stack (or a new stack), then converted back to safetensors.
5. **English is structural, not a flag.** The 76-symbol inventory, CMUdict
   lexicon, English G2P model, hand-written English function words, homograph
   rules, number words, morphology and text normalisation are all separate
   English-specific components (§5). The model itself is language-agnostic,
   but the frontend is not.
6. **The symbol layout is compile-time in Rust.** `voca` hard-codes
   `SYMBOL_COUNT = 76`, `UNKNOWN = 4`, `FIRST_PHONE = 17`, `FIRST_VOWEL = 43`
   (`crates/tiny-voca/src/model.rs:89-102`). A Russian inventory larger
   than 76 symbols requires a Rust change, not just a new config.
7. **The training data was not LJSpeech.** Upstream trained on
   `n4ze3m/tiny-voca`, 17,371 synthetic clips (19.1 h) in two voices, generated
   with Qwen3-TTS; see §4.
8. **This Mac cannot reproduce the upstream training run as-is.** Upstream’s
   default device is CUDA (`upstream/src/tiny_voca/train.py:273`) and the
   source-filter vocoder uses complex STFT/FFT operations
   (`upstream/src/tiny_voca/model.py:146-162,207-238`). MPS support for those
   ops was not verified here; plan for CPU or rented CUDA. No training was run
   in this reconnaissance.
9. **No streaming, no sampling, one utterance per call.** The ONNX/web model
   and the Rust port both synthesize a whole sentence at a time; long text is
   split by `Tts::fit` (`crates/tiny-voca/src/tts.rs:94-112`).

---

## 1. Exact architecture

The Python reference is `upstream/src/tiny_voca/model.py`; the Rust mirror is
`crates/tiny-voca/src/model.rs`. Both match tensor-for-tensor: the
downloaded checkpoint has **184 tensors and 4,537,291 inference parameters**
(verified from `tiny-voca.pt`, see §3/A).

### Top-level flow

```
phoneme ids + stress + voice
  -> token/stress/voice embeddings (sum)
  -> text encoder: 4 pre-norm transformer blocks, 192 ch, 4 heads, RoPE
  -> duration predictor: 2 conv1d + 2 LayerNorm + linear -> 2 outputs/token
  -> repeat each token by its frame duration; add 3 frame-position features
     and the frame voice embedding
  -> prosody stack: 2 transformer blocks -> 3 outputs/frame
     (log F0/200, voicing logit, normalised log energy)
  -> prosody projection: 3 -> 192, added to prosody hidden
  -> acoustic stack: 5 transformer blocks -> 192 ch/frame
  -> source-filter waveform head:
       harmonic: linear 192 -> 98 (80 causal cepstrum + 16 anti + 2 subframe gains)
       noise:    linear 192 -> 36 (32 cepstrum + 4 subframe gains)
       pulse train at predicted F0 + noise, filtered in STFT domain
  -> 24 kHz mono waveform
```

References: `upstream/src/tiny_voca/model.py:241-266` (module tree),
`crates/tiny-voca/src/model.rs:321-335` (Rust struct),
`upstream/docs/model.md:5-31`.

### Concrete dimensions

From `upstream/src/tiny_voca/config.py:13-51`, confirmed by the HF
`config.json` and the checkpoint:

| Field | Value |
|---|---:|
| `sample_rate` | 24,000 Hz |
| `hop` | 256 samples = 10.667 ms (93.75 frames/s) |
| `n_fft` / `n_mels` | 1024 / 80 (training features only) |
| `f0_min` / `f0_max` | 50 / 600 Hz |
| `speakers` | `("F2", "M2")` |
| `vocab_size` | 76 |
| `channels` | 192 |
| `heads` | 4 (head width 48) |
| `feedforward` | 640 |
| `kernel` | 7 (depthwise conv) |
| `encoder_layers` | 4 |
| `prosody_layers` | 2 |
| `acoustic_layers` | 5 |
| `predictor_channels` | 128 |
| `dropout` | 0.1 (training only) |
| `synth_fft` / `synth_hop` | 512 / 128 |
| `noise_fft` / `noise_hop` | 256 / 64 |
| `cep_causal` / `cep_anti` | 80 / 16 |
| `noise_cep` | 32 |
| `harmonic_limit` | 0.47 (highest harmonic 11.28 kHz) |
| `max_tokens` | 512 |
| `max_frames` | 3000 (= 32.0 s at 24 kHz) |
| `pitch_reference_hz` | 200.0 |

The Rust loader validates that `vocab_size == 76`, `channels % heads == 0`,
head width is even, kernel is odd, and both synthesis windows are 4× overlap
with a hop dividing 256 (`crates/tiny-voca/src/model.rs:74-84`).

### 1.1 Text encoder and transformer block

Input is the sum of three embeddings, no positional table:

```python
x = tokens(tokens) + stress(stress) + voices(speakers)[:, None]   # [B,N,192]
```

(`upstream/src/tiny_voca/model.py:268-275`; Rust
`crates/tiny-voca/src/model.rs:438-447`).

Each block is pre-norm:

- `norm1` (LayerNorm 192)
- self-attention: `qkv: 192 -> 576` (3×192), 4 heads, rotary position
  embeddings applied to q and k, scaled dot-product, `out: 192 -> 192`
  (`upstream/src/tiny_voca/model.py:48-63`;
  `crates/tiny-voca/src/model.rs:104-196`)
- residual add
- `norm2` (LayerNorm 192)
- depthwise `Conv1d(192, 192, kernel=7, groups=192, padding=3)`
  (`upstream/src/tiny_voca/model.py:66-83`;
  `crates/tiny-voca/src/model.rs:198-232`)
- `expand: 192 -> 640`, GELU, `project: 640 -> 192`
- residual add; stack ends with a final LayerNorm
  (`upstream/src/tiny_voca/model.py:86-97`;
  `crates/tiny-voca/src/model.rs:235-253`).

Rotary inverse frequencies use the standard `1 / 10000^(2i/width)`
(`upstream/src/tiny_voca/model.py:36-45`;
`crates/tiny-voca/src/model.rs:255-267`). There is no absolute position
table, deliberately, so utterances longer than training remain in range
(`upstream/src/tiny_voca/model.py:49-50`).

### 1.2 Duration predictor

Two 1-D convolutions, GELU, LayerNorm, then a linear to 2 outputs per token
(`upstream/src/tiny_voca/model.py:100-115`;
`crates/tiny-voca/src/model.rs:269-290`):

- `conv1: 192 -> 128`, kernel 3, padding 1
- `conv2: 128 -> 128`, kernel 3, padding 1
- `out: 128 -> 2`
  - output 0 = `log(1 + frames)`
  - output 1 = “has any frames at all” logit for pause-capable tokens

Duration is predicted only for “present” tokens. Every phone gets at least one
frame; punctuation/edge/word-boundary tokens may get zero
(`upstream/src/tiny_voca/model.py:317-324`;
`crates/tiny-voca/src/model.rs:448-455`). The predicted frame count is
`round(expm1(clamp(logit, 0, ln 401)) / speed).max(1)`, i.e. 1–400 frames per
token (Rust uses `round_ties_even`; Python uses `round`).

### 1.3 Frame expansion

For each token with duration `d`, `d` frames are emitted. Each frame gets:

```
[ (offset + 0.5) / d, 1 - that, log1p(d) / 4 ]
```

and `frame_position: 3 -> 192` is added to the encoder state and the frame
voice embedding (`upstream/src/tiny_voca/model.py:277-298`;
`crates/tiny-voca/src/model.rs:460-477`).

### 1.4 Prosody and acoustic stacks

- Prosody stack: 2 blocks, 192 ch, same block shape as the encoder
  (`upstream/src/tiny_voca/model.py:255,300-302`).
- `prosody_head: 192 -> 3`: `[log(F0/200), voicing logit, normalised log energy]`
  (`upstream/src/tiny_voca/model.py:256-257`).
- Vowel frames are forced voiced by taking `max(voicing, 2.0)` before the
  threshold, because the pitch tracker missed creaky vowels
  (`upstream/src/tiny_voca/model.py:359-368`;
  `crates/tiny-voca/src/model.rs:482-487`).
- Decoder conditioning clamps pitch to `[-2, 1.5]`, turns voicing into 0/1,
  clamps energy to `[-4.5, 3]` (`upstream/src/tiny_voca/model.py:304-308`;
  `crates/tiny-voca/src/model.rs:488-495`).
- `prosody_projection: 3 -> 192` is added to the prosody hidden state.
- Acoustic stack: 5 blocks, 192 ch (`upstream/src/tiny_voca/model.py:259`;
  `crates/tiny-voca/src/model.rs:403`).

### 1.5 Vocoder: source-filter, not HiFi-GAN

This is **not** a neural upsampling vocoder. It is a harmonic-plus-noise
source-filter synthesizer. The network predicts slowly varying spectral
envelopes (complex cepstra); the oscillator supplies periodicity
(`upstream/src/tiny_voca/model.py:1-17,165-238`;
`crates/tiny-voca/src/vocoder.rs:1-3`).

Harmonic branch:

- `harmonic: Linear(192, 98)`, where `98 = cep_causal(80) + cep_anti(16) + harmonic_steps(2)`
  (`crates/tiny-voca/src/model.rs:404-407`).
- 80 causal + 16 anticausal cepstral coefficients are placed in a 512-point
  buffer, lifted, FFT’d, exponentiated to a complex response
  (`upstream/src/tiny_voca/model.py:154-162`;
  `crates/tiny-voca/src/vocoder.rs:211-253`).
- A band-limited pulse train is generated from F0 with closed-form harmonics
  below `0.47 * sample_rate` (`upstream/src/tiny_voca/model.py:118-143`;
  `crates/tiny-voca/src/vocoder.rs:296-351`).
- The pulse train is STFT’d at 512/128, multiplied by the predicted response,
  and inverse-STFT’d with a periodic Hann window.

Noise branch:

- `noise: Linear(192, 36)`, where `36 = noise_cep(32) + noise_steps(4)`
  (`crates/tiny-voca/src/model.rs:407`).
- 32 cepstral coefficients → 256-point response; noise is shaped at 256/64
  (`upstream/src/tiny_voca/model.py:224-229`;
  `crates/tiny-voca/src/vocoder.rs:255-291`).

Outputs are summed and the result is `frames * hop` samples
(`upstream/src/tiny_voca/model.py:236-238`;
`crates/tiny-voca/src/vocoder.rs:290-292`). The head has **25,862
parameters** (`upstream/docs/model.md:40`).

### 1.6 Parameter count

Verified from the released checkpoint (184 tensors):

| Component | Parameters |
|---|---:|
| Embeddings (phoneme, stress, voice, frame voice, frame position) | 16,704 |
| Text encoder (4 blocks + final norm) | 1,588,864 |
| Duration + pause predictor | 123,906 |
| Frame transformer A + prosody head/projection | 795,971 |
| Frame transformer B | 1,985,984 |
| Source-filter waveform head | 25,862 |
| **Total at inference** | **4,537,291** |

Training-only: auxiliary mel head 15,440; discriminators 2,675,312
(`upstream/docs/model.md:33-44`).

### 1.7 Input/output representation and limits

- **Input:** `tokens: u32` / int64, ids into the 76-symbol inventory, including
  `<bos>` and `<eos>`; `stress: u8` 0/1/2; `voice: usize`.
  Not characters. The model never sees text (`crates/tiny-voca/src/model.rs:423-446`).
- **Output:** f32 mono waveform at 24 kHz; `durations` in 256-sample frames
  (`upstream/src/tiny_voca/model.py:350-379`;
  `crates/tiny-voca/src/model.rs:510-515`).
- **Max input:** 512 tokens, 3000 frames = 32.0 s
  (`upstream/src/tiny_voca/config.py:45-46`;
  `crates/tiny-voca/src/model.rs:431-459`). `Tts::fit` splits longer text
  at sentence/clause/word boundaries (`crates/tiny-voca/src/tts.rs:94-112`).

---

## 2. Phoneme inventory: 76 symbols, phoneme-conditioned

The inventory is defined in `upstream/src/tiny_voca/text.py:50-63` and mirrored
in Rust at `crates/dizione/src/symbols.rs:3-14`.

```
5 specials:     <pad> <bos> <eos> <sp> <unk>
12 punctuation: , . ? ! ; : " ' ( ) — …
26 consonants:  p b t d k ɡ f v θ ð s z ʃ ʒ h m n ŋ l ɹ j w tʃ dʒ ɾ ʔ
33 vowels:      ə ɐ ɚ ᵻ ɪ i iː ɛ æ ʌ ɑː ɔ ɔː ʊ uː ɜː
                eɪ aɪ ɔɪ oʊ aʊ iə oː
                ɑːɹ ɔːɹ oːɹ ɪɹ ɛɹ ʊɹ aɪɚ aɪə əl n̩
```

Layout constants that other code depends on:

- `FIRST_PUNCTUATION = 5`, `FIRST_PHONE = 17`, `FIRST_VOWEL = 43`
  (`crates/dizione/src/symbols.rs:12-14`).
- `model.rs` repeats `SYMBOL_COUNT = 76`, `UNKNOWN = 4`, `FIRST_PHONE = 17`,
  `FIRST_VOWEL = 43` (`crates/tiny-voca/src/model.rs:89-94`).
- `is_phone(token)` = `token >= 17 || token == 4`; `is_vowel(token)` =
  `token >= 43` (`crates/tiny-voca/src/model.rs:96-102`).

Consequences:

- The model is conditioned on **phonemes**, not characters. Characters are
  converted to phones by the frontend, and only phone ids enter the network.
- The inventory is **American English IPA** as emitted by eSpeak NG / dizione.
  It is not a generic IPA table.
- The 26 consonant slots and 33 vowel slots are a hard partition. A Russian
  inventory with hard/soft consonant pairs (roughly 36 consonant phonemes) does
  **not** fit in 26 consonant slots; see §6.

`crates/dizione/data/letters.txt` is not the model inventory. It contains English letter
*names* used only when the normalizer decides to spell an initialism or a
single letter (`crates/dizione/data/letters.txt:1-27`;
`crates/dizione/src/engine.rs:110-115,181-192`).

---

## 3. Weights: format, loading, tensors, conversion

### 3.1 Runtime format

The deployed file is **safetensors**, not an embedded Rust array:

- `Archive::parse` reads the 8-byte little-endian header length, the JSON
  header, then F32 or I32 tensor bodies; `__metadata__` becomes a string map
  (`crates/tiny-voca/src/weights.rs:23-58`).
- `TinyVoca::load` reads the `config` metadata key as JSON
  (`crates/tiny-voca/src/model.rs:371-375`), then removes each expected
  tensor by name and shape; any leftover tensor is an error
  (`crates/tiny-voca/src/model.rs:411-414`).
- The runtime path is `$VOCA_HOME/tiny-voca.safetensors`
  (`crates/tiny-voca/src/bin/voca.rs:97-106`); the development binary
  defaults to `weights/tiny-voca.safetensors` or `$TINY_VOCA_MODEL`
  (`crates/tiny-voca/src/main.rs:12-30,52-58`).

The `voca` binary does **not** contain the weights. `install.sh` downloads
`voca-data.tar.gz` (19 MB) into `~/.local/share/voca`
(`install.sh:19-22`).

### 3.2 `scripts/convert_weights.py`

`scripts/convert_weights.py:1-22` consumes a PyTorch checkpoint with:

- `state["format_version"] == 4`
- `state["model"]`: the inference `state_dict` (no optimizer, no mel head, no
  discriminators)
- `state["config"]`: the `Config` dict
- `state["frontend"]`: frontend identity, including `symbols`

It writes every tensor as `float().contiguous()` plus metadata
`config`, `step`, `frontend`, `symbols` (`scripts/convert_weights.py:17-21`).
The Rust port only reads `config`; the other metadata is informational.

Upstream produces this checkpoint with `train.export_model`
(`upstream/src/tiny_voca/train.py:180-183`) or
`upstream/scripts/export_checkpoint.py` from a full training checkpoint
(`upstream/scripts/export_checkpoint.py:1-32`).

### 3.3 Every tensor, verified from the released checkpoint

Method: `tiny-voca.pt` from
`https://huggingface.co/n4ze3m/tiny-voca/resolve/main/tiny-voca.pt`
(SHA-256 `ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70`),
unzipped, `data.pkl` parsed with a stub unpickler (no PyTorch installed). Count:
**184 tensors, 4,537,291 parameters**. Full list in Appendix A.

The per-module pattern (repeated for each block):

| Module | Tensor | Shape |
|---|---|---|
| embeddings | `tokens.weight` | `[76, 192]` |
| | `stress.weight` | `[3, 192]` |
| | `voices.weight` | `[2, 192]` |
| each block `P` | `{P}.norm1.weight`, `.bias` | `[192]` |
| | `{P}.attention.qkv.weight` | `[576, 192]` |
| | `{P}.attention.qkv.bias` | `[576]` |
| | `{P}.attention.out.weight` | `[192, 192]` |
| | `{P}.attention.out.bias` | `[192]` |
| | `{P}.norm2.weight`, `.bias` | `[192]` |
| | `{P}.depthwise.weight` | `[192, 1, 7]` |
| | `{P}.depthwise.bias` | `[192]` |
| | `{P}.expand.weight` | `[640, 192]` |
| | `{P}.expand.bias` | `[640]` |
| | `{P}.project.weight` | `[192, 640]` |
| | `{P}.project.bias` | `[192]` |
| encoder | `encoder.{blocks.0..3}`, `encoder.norm.weight`, `.bias` | as above |
| duration | `duration.conv1.weight` | `[128, 192, 3]` |
| | `duration.conv1.bias` | `[128]` |
| | `duration.conv2.weight` | `[128, 128, 3]` |
| | `duration.conv2.bias` | `[128]` |
| | `duration.norm1/2.weight`, `.bias` | `[128]` |
| | `duration.out.weight` | `[2, 128]` |
| | `duration.out.bias` | `[2]` |
| frame | `frame_voices.weight` | `[2, 192]` |
| | `frame_position.weight` | `[192, 3]` |
| | `frame_position.bias` | `[192]` |
| prosody | `prosody_stack.{blocks.0..1}`, `prosody_stack.norm.*` | block pattern |
| | `prosody_head.weight` | `[3, 192]` |
| | `prosody_head.bias` | `[3]` |
| | `prosody_projection.weight` | `[192, 3]` |
| | `prosody_projection.bias` | `[192]` |
| acoustic | `acoustic_stack.{blocks.0..4}`, `acoustic_stack.norm.*` | block pattern |
| vocoder | `vocoder.harmonic.weight` | `[98, 192]` |
| | `vocoder.harmonic.bias` | `[98]` |
| | `vocoder.noise.weight` | `[36, 192]` |
| | `vocoder.noise.bias` | `[36]` |

Tensor names and shapes are also asserted one-by-one in
`crates/tiny-voca/src/model.rs:379-408`.

### 3.4 Reproduce the tensor inspection

```sh
cd /tmp
curl -fsSL -O https://huggingface.co/n4ze3m/tiny-voca/resolve/main/tiny-voca.pt
curl -fsSL -O https://huggingface.co/n4ze3m/tiny-voca/resolve/main/config.json
unzip -o -q tiny-voca.pt -d tiny-voca-pt
python3 - <<'PY'
import sys, types, pickle, math
torch = types.ModuleType("torch")
torch.__getattr__ = lambda n: type(n, (), {"__init__": lambda s, *a, **k: None})
sys.modules["torch"] = torch
utils = types.ModuleType("torch._utils")
utils._rebuild_tensor_v2 = lambda storage, offset, size, stride, *a: tuple(size)
sys.modules["torch._utils"] = torch._utils = utils
storage = types.ModuleType("torch.storage")
storage._load_from_bytes = lambda b: None
sys.modules["torch.storage"] = torch.storage = storage
class U(pickle.Unpickler):
    def persistent_load(self, pid): return None
model = U(open("tiny-voca-pt/inference.pt/data.pkl", "rb")).load()["model"]
print(len(model), sum(math.prod(v) for v in model.values()))
PY
```

Expected: `184 4537291`.

---

## 4. Upstream training repository (`n4ze3m/tiny-voca`)

Clone: `git clone https://github.com/n4ze3m/tiny-voca /tmp/tiny-voca-upstream`
(revision `e7f1bde`, 2026-11-18).

### 4.1 Entrypoints

Declared in `upstream/pyproject.toml:24-30`:

| Command | Module | Purpose |
|---|---|---|
| `python -m tiny_voca.prepare` | `prepare.py` | audio → `wave`, log-mel, pitch/voicing/energy |
| `python -m tiny_voca.annotate` | `annotate.py` | text → phonemes + forced-aligned durations |
| `python -m tiny_voca.train` | `train.py` | from-scratch training |
| `python -m tiny_voca.evaluate` | `evaluate.py` | WER etc. |
| `python -m tiny_voca.infer` | `infer.py` | CLI inference |
| `python -m tiny_voca.export` | `export.py` | ONNX export |

The documented sequence is in `upstream/README.md:82-94` (the “Training”
section):

```bash
export TINY_VOCA_FRONTEND=own
python -m tiny_voca.prepare --inputs data/train-*.parquet --split train --output features
python -m tiny_voca.annotate --features features --output annotations
python -m tiny_voca.train --train annotations/train.jsonl --validation annotations/validation.jsonl \
  --output runs/NAME --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0
python -m tiny_voca.evaluate --checkpoint runs/NAME/inference.pt --annotations annotations --output eval/NAME
```

### 4.2 Config

`upstream/src/tiny_voca/config.py:13-51` is the single source of truth; the
`Config` dataclass is frozen and serialised into the checkpoint. It includes
sample rate, hop, n_fft, n_mels, F0 range, speakers, vocab size, all transformer
dimensions, vocoder FFT sizes, cepstrum orders, `max_tokens`, `max_frames`,
mel/energy normalisation constants and `pitch_reference_hz`.

Training arguments are in `upstream/src/tiny_voca/train.py:267-311`: default
`--device cuda`, 100k steps, 9000 padded frames/batch, max batch 32, AdamW
lr 3e-4, warmup 1000, GAN start 8000, teacher hold/ramp, GAN ratio, etc.

### 4.3 Data format

1. **Source:** HF dataset `n4ze3m/tiny-voca`, Parquet shards with columns `id`,
   `text`, `voice_id`, `audio`, provenance columns
   (`upstream/src/tiny_voca/prepare.py:118-158`;
   `upstream/docs/dataset-card/README.md`, “Columns”).
2. **Acoustic preparation:** `prepare.py` reads 24 kHz mono audio, pads to a
   whole frame, computes log-mel (80 bands, n_fft 1024, hop 256), `librosa.pyin`
   F0/voicing, RMS energy, and writes one `.pt` per clip with
   `{"wave", "mel", "prosody"}` plus a JSONL manifest
   (`upstream/src/tiny_voca/prepare.py:45-66,87-175`).
3. **Annotation:** `annotate.py` reads the acoustic manifest, phonemizes text,
   and writes `tokens`, `stress`, `durations`, `frames`, `qc`, provenance to
   `<split>.jsonl`; durations must sum to the frame count exactly
   (`upstream/src/tiny_voca/annotate.py:1-20,290-303`;
   `upstream/docs/annotations-v3.md:35-80`).
4. **Training dataset:** `AnnotatedDataset` joins the manifest with the feature
   files, validates frontend/acoustic fingerprints, and returns tensors
   (`upstream/src/tiny_voca/data.py:48-132`).

### 4.4 Tokenizer / frontend

`upstream/src/tiny_voca/text.py` has two possible frontends:

- `espeak` (frontend version 3, default): eSpeak NG `en-us`, `--ipa --sep=_`,
  one token per eSpeak phoneme, stress as a separate 0/1/2 feature, explicit
  punctuation/edge/word-boundary tokens (`upstream/src/tiny_voca/text.py:1-30,41-77,181-219`).
- `own` (frontend version 4): the `dizione` package, same 76 symbols
  (`upstream/src/tiny_voca/text.py:6-13,116-143`).

The released `config.json` says `phonemizer: {name: "dizione", version: 4}`, so
the released model’s intended inference frontend is dizione. Training history
was 66,000 steps with eSpeak labels then 38,737 steps with dizione labels
(`upstream/docs/model.md:73-77`).

Inventory: 5 specials + 12 punctuation + 26 consonants + 33 vowels = 76
(`upstream/src/tiny_voca/text.py:50-60`). `SYMBOL_TO_ID` is append-only;
removing/reordering invalidates checkpoints (`upstream/src/tiny_voca/text.py:59-61`).

### 4.5 Model source

`upstream/src/tiny_voca/model.py` is the reference `TinyVoca` (see §1). The
model is trained **from scratch**; no pretrained model weights are part of it
(`upstream/docs/model-card/README.md`, “Pretrained weights in the model |
None”). The only pretrained component is the forced aligner used to label
durations (`facebook/wav2vec2-lv-60-espeak-cv-ft`,
`upstream/src/tiny_voca/align.py:38`), and it is not part of inference.

### 4.6 Dataset

`n4ze3m/tiny-voca` (HF datasets), CC BY 4.0:

- 17,371 clips, 19.1 h total; train 16,018 clips / 17.6 h, validation 714,
  test 594 (`upstream/docs/model.md:54-64`;
  `upstream/docs/dataset-card/README.md`).
- Two voices, F2 and M2, American English, 24 kHz mono, 16-bit FLAC inside
  Parquet (`upstream/docs/dataset-card/README.md`).
- Synthetic audio: voices created with Qwen3-TTS VoiceDesign; clips generated
  with Qwen3-TTS 1.7B Base; screened with Whisper small.en
  (`upstream/docs/model-card/README.md`, “Training”;
  `upstream/docs/dataset-card/README.md`, “How it was made”).
- Text sources: LibriTTS-R transcripts (CC BY 4.0), LLM-written sentences,
  Mozilla Common Voice sentences (CC0), project templates
  (`upstream/docs/dataset-card/README.md`, “Text sources”).
- **Not LJSpeech.** The dataset card never mentions LJSpeech.

Training cost: 104,737 steps, 7.7 h on one RTX 4090 (`upstream/docs/model.md:73-77`;
`upstream/README.md:10-14`).

### 4.7 Checkpoint → voca conversion path

```
train.py / export_checkpoint.py
  -> inference.pt   {format_version: 4, config, step, frontend, annotation, model}
  -> voca/scripts/convert_weights.py
  -> tiny-voca.safetensors  {F32 tensors + metadata}
  -> TinyVoca::load (Rust)
```

Upstream `train.export_model` writes `inference.pt` with
`model.state_dict()` (`upstream/src/tiny_voca/train.py:180-183`).
`upstream/scripts/export_checkpoint.py` extracts `system.model.*` from a full
`last.pt` and calls the same function (`upstream/scripts/export_checkpoint.py:1-32`).
`scripts/convert_weights.py` then writes safetensors
(`scripts/convert_weights.py:13-22`).

### 4.8 `n4ze3m/dizione`

Clone: `git clone https://github.com/n4ze3m/dizione /tmp/dizione-upstream`
(revision `8cd889e`).

- Apache-2.0 code; data derived from CMUdict and Wikipedia Homograph Data
  (`dizione/README.md:71-74`; `dizione/pyproject.toml:5-13`).
- CMUdict pinned at commit `74790861f652b15e4ac49015a90074ad62a27690`
  (2025-10-24); homograph data at
  `8f008f021e88f8b71118a27ae655f1f3121162bc` (2021-10-14)
  (`dizione/scripts/fetch_sources.sh:12-22`).
- `scripts/build_lexicon.py` builds `lexicon.tsv`, `align.json` and
  `lexicon.json` from CMUdict; only CMUdict goes in
  (`dizione/scripts/build_lexicon.py:1-12,95-125`).
- Unknown-word model: letters → ARPAbet CTC, trained by
  `dizione/scripts/train_g2p.py`; runtime is NumPy only (`dizione/src/dizione/g2p.py:1-95`).
- Data files: `lexicon.tsv`, `align.json`, `g2p.json`, `g2p.npz`,
  `homographs.json` (`crates/dizione/src/engine.rs:99-109`).
- Limits: American English only (`dizione/README.md:63-67`).

---

## 5. Everything that assumes English

This is the part a Russian model must replace. It is not a small patch.

### 5.1 Inventory and model partitions

| Location | English assumption | Russian consequence |
|---|---|---|
| `crates/dizione/src/symbols.rs:3-14` | 76 American-English IPA symbols; 26 consonant slots (17–42), 33 vowel slots (43–75) | Russian needs ~36 consonant phonemes with palatalisation; does not fit 26 slots. Expand inventory and move `FIRST_VOWEL`, or encode softness as a separate token (costly). |
| `crates/tiny-voca/src/model.rs:89-102` | `SYMBOL_COUNT = 76`, `UNKNOWN = 4`, `FIRST_PHONE = 17`, `FIRST_VOWEL = 43`, `is_phone`/`is_vowel` | Compile-time Rust edit required for any other inventory. These functions drive duration presence and vowel voicing. |
| `crates/dizione/src/symbols.rs:62-101` | tokenization adds `<bos>`, `<sp>`, `<eos>`; punctuation collapsed | Reusable, but punctuation set and clustering rules are English prose conventions. |

### 5.2 Grapheme-to-phoneme

| Location | What is English | What Russia needs |
|---|---|---|
| `crates/dizione/src/engine.rs:99-121` | loads `lexicon.tsv` from CMUdict; loads `align.json`, `g2p.npz`, `homographs.json` | Russian pronunciation lexicon; Russian align/convert tables or a Russian G2P model. |
| `crates/dizione/src/engine.rs:139-178` | dictionary lookup, English regular endings `s/es/ed/ing`, compounds | Russian inflection, `ё`, `ь`/`ъ`, vowel reduction, final devoicing, assimilation. |
| `crates/dizione/src/g2p.rs` | letter embeddings for A–Z, ARPAbet output, CTC greedy | Cyrillic alphabet and Russian phoneme output (or a rule-based Russian G2P). |
| `crates/dizione/src/morphology.rs:1-60` | English suffixes (`-ness`, `-less`, `-ment`, `-able`, …), prefixes, rhotic vowels, British spellings | Russian case/number/verb morphology, not English suffixes. |
| `crates/dizione/src/sentence.rs:1-50` | English sentence stress, weak forms, linking r, flaps | Russian stress and clitic rules; different weak-form system. |
| `crates/dizione/src/homographs.rs:1-60` | English homographs (`read`, `live`, `bow`, …) and a 13-class context model | Russian stress homographs (`за́мок`/`замо́к`, `му́ка`/`мука́`) and grammatical homographs; new rules/classifier. |
| `crates/dizione/data/letters.txt` | 26 English letter names | Cyrillic letter names (and policy for Latin brand names). |

### 5.3 Text normalisation

| Location | English assumption |
|---|---|
| `crates/dizione/src/normalize.rs:66-86` | English month/day names, currency names (`dollar`, `pound`, `euro`), time zones, sizes, lengths, `version` words. |
| `crates/dizione/src/normalize.rs:1-20` | the whole rule order is a port of the Python English normalizer. |
| `crates/dizione/src/numbers.rs:3-7` | `zero`…`nineteen`, `twenty`…`ninety`, `thousand`…`quintillion`; English ordinals (`first`, `second`, `twelfth`, …). |
| `crates/dizione/src/normalize.rs:884-1100` | initialisms, letter naming, codes, addresses, file names, web URLs — all ASCII `[A-Za-z]`. |
| `crates/dizione/src/normalize.rs:1160-1265` | English abbreviations (`U.S.`, `e.g.`, `p.m.`) and sentence starters. |
| `crates/dizione/data/*.txt` | all hand-written English tables: `abbreviations.txt` (103 lines), `chat.txt` (107), `endings.txt` (130), `exceptions.txt` (229), `function_words.txt` (189), `homograph_rules.txt` (37), `letters.txt` (27), `phrases.txt` (63), `units.txt` (76). Counts include comments. |
| `crates/dizione/src/tables.rs:1-18` | embeds those tables with `include_str!`; replacing them is a rebuild, not a runtime config. |

### 5.4 Sentence splitting and the TTS wrapper

`crates/tiny-voca/src/tts.rs:133-188`:

- English abbreviation list (`mr`, `mrs`, `dr`, `prof`, `sr`, `jr`, `st`,
  `vs`, `etc`, `no`, `inc`, `ltd`, `co`, `e.g`, `i.e`, `a.m`, `p.m`).
- `is_word` is ASCII alphanumeric + `_`.
- `opens_sentence` is `c.is_ascii_uppercase() || c.is_ascii_digit()`
  (`tts.rs:157`). **A Russian sentence starting with `Привет` will not be
  recognised as a sentence start**, so long Russian input may not split and
  will hit `max_tokens`.
- `sentences()` strips whitespace and splits on `. ! ? …` only after ASCII
  uppercase/digits; Russian text needs Unicode case handling and Cyrillic
  punctuation conventions.
- `Tts::fit` splits at `, ; : —` and then spaces; those marks are usable but
  should be reviewed for Russian.

### 5.5 Other frontend constants

- `upstream/src/tiny_voca/text.py:41-42,70-77`: `ESPEAK_VOICE = "en-us"` and an
  English fallback map for foreign eSpeak phonemes.
- `upstream/src/tiny_voca/infer.py:12-17`: `ENGLISH_ONLY = "tiny-voca only
  reads English for now"`.
- `crates/dizione/src/lib.rs:1-32`: crate docs say “English text to
  phonemes”; `VERSION = 4` is the token convention.
- `skills/voca/SKILL.md`: speaks replies; no language parameter.
- `README.md` and `NOTICE`: “English only”.

### 5.6 Minimum Russian frontend

To make Russian work end-to-end you need, at minimum:

1. **A Russian phoneme inventory** (decide size and whether to expand beyond
   76) with IPA or an internal symbol set, stress marked as the existing
   separate 0/1/2 feature.
2. **A Russian text normaliser**: numbers with case/gender, dates, times,
   money, units, abbreviations, `№`, quotes `«»`, dashes, `ё`, Latin brand
   names.
3. **A Russian G2P/lexicon with stress**: dictionary + rules for `ь`/`ъ`,
   unstressed vowel reduction, final devoicing, assimilation, and lexical
   stress; plus a stress predictor for unknown words (Russian stress is not
   written).
4. **A Russian homograph/stress disambiguator** for pairs such as
   `за́мок`/`замо́к`, `му́ка`/`мука́`, and grammatical homographs.
5. **Unicode-aware sentence splitting and abbreviation handling** in
   `tts.rs` / the normalizer.
6. **A duration source**: a Russian phoneme recogniser/forced aligner (MFA with
   a Russian model, a multilingual CTC model, or a corpus with alignments).
   This is a data-pipeline change, not just a frontend change; see §6.

The model code itself does not need to know the language, only the token ids.

---

## 6. Cheapest path to a Russian voice at 4.5M parameters

### 6.1 The architecture is language-agnostic; the frontend is not

The network consumes integer phone ids, a stress level and a voice index
(`crates/tiny-voca/src/model.rs:423-446`). Nothing in the transformer or
vocoder is English-specific. The source-filter head models pitch, voicing,
energy and spectral envelopes, all of which transfer across languages
(`upstream/src/tiny_voca/model.py:1-17`). The release’s own community list
demonstrates this: German, Italian, Turkish, Korean, Polish, Spanish, French,
Brazilian Portuguese, Arabic, Bengali and Sanskrit derivatives exist, including
CPU-trained Arabic and fine-tuned Sanskrit (`upstream/README.md:109-129`).

### 6.2 Options

| Path | What it costs | What it buys | Recommendation |
|---|---|---|---|
| **(a) Retrain the same architecture from scratch on Russian** | Russian frontend + Russian corpus with durations + GPU/CPU time. Upstream did 17.6 h / 104,737 steps in 7.7 h on one 4090. | Cleanest accent and prosody; no English baggage; can choose inventory/voice. | **Primary choice** if you have roughly ≥5–10 h of Russian speech with phone durations (or can align it). |
| **(b) Fine-tune English weights** | Same data/alignment need; new phoneme embeddings/encoder; risk of English accent and voice identity. | Faster convergence on very small Russian data (<2 h); reuses acoustic/prosody/vocoder weights. | Fallback only for tiny data. The text encoder (1.59M) and token embeddings are phoneme-specific; the 2-row voice embeddings are F2/M2-specific. |
| **(c) Replace with another model** | New inference stack, loses 30 ms / 4.5M / Rust single-binary target; may be larger and slower. | Potentially better Russian quality with a pretrained Russian TTS. | Only if Russian quality dominates the product constraints; not the cheapest for this project. |

### 6.3 What is easy

- The model is tiny: 4.5M params, 184 tensors, 18 MB. Retraining is cheap
  relative to modern TTS.
- Explicit duration prediction means no learned aligner in the model; durations
  are supervision, so any aligner that produces phone boundaries works.
- Stress is already a first-class 0/1/2 feature, which fits Russian lexical
  stress.
- The vocoder is a source-filter design with only 25,862 parameters; it is
  language-independent in structure and cheap to retrain.
- Rotary positions mean long utterances do not need a position table; but the
  hard 512-token / 3000-frame limits remain.

### 6.4 What is hard

1. **Inventory size/partition.** `model.rs` hard-codes 76 symbols and the
   26-consonant / 33-vowel split. Russian likely needs more consonant symbols
   for hard/soft pairs. Options:
   - expand the inventory (e.g. ~90 symbols), update `SYMBOLS`,
     `SYMBOL_COUNT`, `FIRST_VOWEL`, `FIRST_PHONE` and the checkpoint config;
   - or keep 76 and encode palatalisation with a `ʲ` token (one extra slot,
     but doubles some phone sequences and complicates duration/voicing rules).
   Expanding is cleaner. This is a small Rust change, but it must be mirrored
   in `crates/dizione/src/symbols.rs` and any tests.
2. **Frontend is a full rewrite**, not a translation of a few tables: CMUdict,
   English G2P, English morphology, homographs, number words and normalisation
   are all English (§5). Budget this as the main engineering work.
3. **Durations / alignment.** Upstream’s aligner is
   `facebook/wav2vec2-lv-60-espeak-cv-ft` with an English/eSpeak label set
   (`upstream/src/tiny_voca/align.py:38`). It cannot label Russian phones. You
   need a Russian or multilingual phoneme recogniser, MFA with a Russian
   acoustic model, or a Russian corpus that ships alignments. Map its phones to
   your inventory and validate that durations sum to the frame count
   (`upstream/src/tiny_voca/annotate.py:290-303`).
4. **Training device.** Upstream defaults to CUDA
   (`upstream/src/tiny_voca/train.py:273`). The Mac is M5 Pro without CUDA.
   The vocoder uses complex `torch.fft`/`torch.stft`/`torch.istft`
   (`upstream/src/tiny_voca/model.py:146-162,207-238`); MPS support for these
   ops is not guaranteed. Plan CPU (slow) or rented CUDA.
5. **Data licence.** Russian corpus licences are unverified (STATUS risk 3).
   Any Russian derivative must clear training and redistribution rights.

### 6.5 Can the vocoder be reused?

- **Architecture: yes.** The harmonic/noise source-filter head is language-
  and voice-agnostic in form; sample rate 24 kHz and F0 range 50–600 Hz are
  fine for Russian. Keep it to preserve the Rust inference and the 4.5M budget.
- **Weights: not directly.** The two linear heads (`vocoder.harmonic`,
  `vocoder.noise`, 25,862 params) were trained to the F2/M2 voices and English
  prosody. Retrain them with the rest of the model; optionally initialise from
  English to speed convergence, but that keeps some English timbre.
- **Rust code: reusable unchanged** if you keep 24 kHz / hop 256 and the same
  tensor names/shapes. The vocoder code is independent of the phoneme
  inventory.

### 6.6 Practical recommendation

1. Freeze the Russian phoneme inventory first (decide 76 vs expanded; keep
   stress as 0/1/2).
2. Build the Russian frontend and a duration pipeline in parallel.
3. Retrain from scratch with `upstream` `Config` adjusted (speakers,
   vocab_size, maybe channels unchanged). Keep 24 kHz / 256 hop so the Rust
   runtime and WAV path stay valid.
4. Convert `inference.pt` → safetensors with `scripts/convert_weights.py`;
   update `SYMBOLS`/`SYMBOL_COUNT`/`FIRST_VOWEL` in Rust if the inventory
   changed.
5. Validate with the existing parity harness
   (`crates/tiny-voca/src/bin/parity.rs`,
   `scripts/dump_reference.py`) against PyTorch before shipping.

---

## 7. Licences

| Artifact | Licence | Source |
|---|---|---|
| `voca` Rust code | Apache-2.0 | `LICENSE`; `crates/*/Cargo.toml` |
| `tiny-voca` Python code | Apache-2.0 | `upstream/pyproject.toml:5-13`; `upstream/LICENSE` |
| `tiny-voca` weights | Apache-2.0 | `upstream/docs/model-card/README.md` frontmatter and “License” |
| `tiny-voca` dataset | CC BY 4.0, attribution required | `upstream/docs/dataset-card/README.md`; `upstream/docs/dataset-card/LICENSE.md` |
| `dizione` code | Apache-2.0 | `dizione/pyproject.toml:5-13`; `dizione/LICENSE` |
| CMUdict data | BSD-style (CMU copyright; retain notice in source and binary redistribution) | `dizione/scripts/fetch_sources.sh:7-18`; CMUdict `LICENSE` at pinned commit |
| Wikipedia Homograph Data | Apache-2.0 | `dizione/scripts/fetch_sources.sh:8-10,19-20` |
| Qwen3-TTS (training audio generator) | Apache-2.0 | `upstream/docs/dataset-card/LICENSE.md` |
| eSpeak NG | GPL-3.0 | `upstream/docs/model.md:91-98`; used for the first 66k steps of the released run’s labels, no runtime dependency |

Constraints for a Russian derivative:

- **Code/weights:** Apache-2.0 permits a Russian fork; keep `NOTICE`/attribution
  for upstream. New Russian code can be Apache-2.0.
- **Data:** a Russian corpus needs a licence that permits training and
  redistribution of derived weights. Do not assume CC BY 4.0; verify.
- **CMUdict:** not needed for Russian unless you keep an English frontend or
  reuse the English lexicon. If you do, retain the CMU copyright notice in
  source and binary distributions.
- **eSpeak NG:** avoid GPL-3.0 tools in the Russian pipeline if you want a
  permissively licensed distribution; using it as a separate process is less
  risky than linking, but the safest route is a permissive Russian G2P/aligner.
- **Attribution:** the dataset requires credit “tiny-voca dataset, Muhammed
  Nazeem” and a link; the model card/citation is in `upstream/CITATION.cff`.
- **Synthetic data:** upstream audio was generated with Qwen3-TTS (Apache-2.0),
  not real speakers; no personality rights issue for the F2/M2 voices, but a
  Russian voice generated from a real speaker needs consent.

---

## 8. Interfaces to preserve

### 8.1 `voca` CLI (user-facing)

`crates/tiny-voca/src/bin/voca.rs:14-30`:

```
voca [options] TEXT...
echo TEXT | voca [options]

-v, --voice NAME     F2 (default) or M2
-s, --speed X        0.25 to 4 (default 1)
-p, --pitch X        semitones (default 0)
-o, --output FILE    WAV instead of playing
-f, --file FILE      read text from a file
    --seed N         noise seed (default 0)
-h, --help
```

- Voice is uppercased (`voca.rs:65`), matched case-insensitively
  (`model.rs:417-419`).
- `-p` converts semitones to log-F0: `value * ln(2) / 12`
  (`voca.rs:67`).
- Data directory: `$VOCA_HOME`, default `~/.local/share/voca`
  (`voca.rs:32-38`); expects `tiny-voca.safetensors` and `phonemizer/`
  (`voca.rs:98-106`).
- Playback: `afplay` on macOS; `pw-play`, `paplay`, `aplay`, `play` elsewhere
  (`voca.rs:40-50`).

### 8.2 `tiny-voca` CLI (development/reference)

`crates/tiny-voca/src/main.rs:12-30`:

```
--text TEXT
--text-file FILE        ('-' for stdin)
--output FILE            required
--voice NAME             F2 default
--speed X
--pitch-shift X          log-F0, 0.058 ≈ semitone
--pitch-range X          0..3
--seed N
--model FILE             $TINY_VOCA_MODEL or weights/tiny-voca.safetensors
--data DIR               $DIZIONE_DATA or weights/phonemizer
--threads N              default 1
--no-split               one utterance, like the reference CLI
--pcm16                  16-bit PCM instead of 32-bit float
--quiet
```

It also prints the phoneme symbols, duration and timing summary unless
`--quiet` (`main.rs:100-115`).

### 8.3 WAV output

`crates/tiny-voca/src/wav.rs:1-33`:

- default: 32-bit IEEE float WAV (`format = 3`), which is what the reference
  writes.
- `--pcm16`: 16-bit PCM, clamps to `[-1, 1]`.
- `voca` always writes 32-bit float (`voca.rs:110-113`); it does not normalise
  the peak. The reference `speak_text` scales to 0.99
  (`upstream/src/tiny_voca/infer.py:6-8,128`); a Russian port should preserve
  or add that behaviour before 16-bit export.

### 8.4 Installer

`install.sh:1-57`:

- `VOCA_BIN` default `~/.local/bin`; `VOCA_HOME` default `~/.local/share/voca`.
- Downloads `voca-data.tar.gz` (model + phonemizer) from the latest release
  into `$VOCA_HOME`.
- macOS arm64: prebuilt binary; otherwise builds from a checkout with cargo or
  `cargo install --git`.
- Copies `skills/voca/SKILL.md` into `~/.claude/skills/voca` when
  `~/.claude` exists.

### 8.5 Model file / embedding strategy

- **No embedding.** Weights are an external safetensors file plus a
  `phonemizer/` directory of data files. `weights.rs` is a reader, not a
  compiled array.
- A Russian release can keep the same filenames and layout:
  `~/.local/share/voca/tiny-voca.safetensors` and
  `~/.local/share/voca/phonemizer/{lexicon.tsv,align.json,g2p.json,g2p.npz,homographs.json}`.
  The Rust loader only needs the safetensors `config` metadata and tensors.
- If you change the inventory, the `phonemizer/` contents change, but the
  installer and `VOCA_HOME` contract can stay.
- To keep the model/embedding story small, do not add a second model file;
  replace the same file in the data tarball.

### 8.6 Library API

- `tiny_voca::tts::Tts::load(weights: &Path, phonemizer_data: &Path)`
  (`crates/tiny-voca/src/tts.rs:30-36`).
- `Tts::speak(text, voice, options)` (sentence-split) and
  `Tts::speak_whole` (one utterance) (`tts.rs:40-61`).
- `tiny_voca::{Config, Options, Synthesis, TinyVoca}`
  (`crates/tiny-voca/src/lib.rs:1-13`).
- `dizione::Phonemizer::load(dir)` and `phonemize(text)`
  (`crates/dizione/src/engine.rs:99-121,357-363`).
- `dizione::SYMBOLS`, `Phonemized` (`crates/dizione/src/lib.rs:24-32`).

Keep these signatures; a Russian frontend should plug in behind
`Phonemizer::phonemize` and return the same `Phonemized { symbols, ids, stress }`
shape.

---

## 9. Risks / open questions for the orchestrator

1. **Inventory decision is blocking.** 26 consonant slots vs Russian
   hard/soft pairs. Decide before writing the model or data pipeline. Expanding
   beyond 76 requires coordinated Rust + checkpoint changes.
2. **Russian durations/alignment is the critical path**, not the model. Which
   aligner/model will produce Russian phone boundaries? Is there a Russian
   corpus with alignments?
3. **MPS training feasibility is unverified.** Complex STFT/FFT ops may not run
   on MPS. CPU training of 17.6 h-equivalent data will be slow; renting CUDA
   may be cheaper than engineering around it.
4. **Corpus licence is unknown.** Russian datasets often carry CC BY-NC-SA or
   unclear provenance; check before training.
5. **Frontend scope is large.** A Russian normalizer + stress-aware lexicon +
   G2P + homographs is comparable in size to the Rust `dizione` crate, not a
   table swap.
6. **Output quality target is unstated.** The upstream English voice is
   intelligible but synthetic; a 4.5M Russian model on a few hours may be
   intelligible but not natural. Decide the acceptance threshold (WER? MOS?)
   before training.
7. **Keep the 30 ms / 4.5M / Rust contract?** If yes, keep 24 kHz / hop 256 and
   the same tensor names. If Russian quality requires a larger model, the
   Rust inference and installer need rework.
8. **English voice retention.** If the product keeps English F2/M2 alongside
   Russian, the 76-symbol English inventory and the Russian inventory must
   coexist. The current model has one shared inventory and one `vocab_size`;
   either use a larger shared inventory or two model files/CLIs. One model with
   a union inventory is possible but wastes capacity and complicates the
   frontend.

---

## Appendix A: all 184 tensors in the released checkpoint

Verified from `tiny-voca.pt` (step 104,737). The order is the `state_dict`
order; shapes are exact. This is generated, not hand-written; see §3.4 for the
parser.

<details>
<summary>Full tensor table</summary>

| # | Tensor | Shape | Params |
|---:|---|---|---:|
| 1 | `tokens.weight` | [76, 192] | 14,592 |
| 2 | `stress.weight` | [3, 192] | 576 |
| 3 | `voices.weight` | [2, 192] | 384 |
| 4 | `encoder.blocks.0.norm1.weight` | [192] | 192 |
| 5 | `encoder.blocks.0.norm1.bias` | [192] | 192 |
| 6 | `encoder.blocks.0.attention.qkv.weight` | [576, 192] | 110,592 |
| 7 | `encoder.blocks.0.attention.qkv.bias` | [576] | 576 |
| 8 | `encoder.blocks.0.attention.out.weight` | [192, 192] | 36,864 |
| 9 | `encoder.blocks.0.attention.out.bias` | [192] | 192 |
| 10 | `encoder.blocks.0.norm2.weight` | [192] | 192 |
| 11 | `encoder.blocks.0.norm2.bias` | [192] | 192 |
| 12 | `encoder.blocks.0.depthwise.weight` | [192, 1, 7] | 1,344 |
| 13 | `encoder.blocks.0.depthwise.bias` | [192] | 192 |
| 14 | `encoder.blocks.0.expand.weight` | [640, 192] | 122,880 |
| 15 | `encoder.blocks.0.expand.bias` | [640] | 640 |
| 16 | `encoder.blocks.0.project.weight` | [192, 640] | 122,880 |
| 17 | `encoder.blocks.0.project.bias` | [192] | 192 |
| 18 | `encoder.blocks.1.norm1.weight` | [192] | 192 |
| 19 | `encoder.blocks.1.norm1.bias` | [192] | 192 |
| 20 | `encoder.blocks.1.attention.qkv.weight` | [576, 192] | 110,592 |
| 21 | `encoder.blocks.1.attention.qkv.bias` | [576] | 576 |
| 22 | `encoder.blocks.1.attention.out.weight` | [192, 192] | 36,864 |
| 23 | `encoder.blocks.1.attention.out.bias` | [192] | 192 |
| 24 | `encoder.blocks.1.norm2.weight` | [192] | 192 |
| 25 | `encoder.blocks.1.norm2.bias` | [192] | 192 |
| 26 | `encoder.blocks.1.depthwise.weight` | [192, 1, 7] | 1,344 |
| 27 | `encoder.blocks.1.depthwise.bias` | [192] | 192 |
| 28 | `encoder.blocks.1.expand.weight` | [640, 192] | 122,880 |
| 29 | `encoder.blocks.1.expand.bias` | [640] | 640 |
| 30 | `encoder.blocks.1.project.weight` | [192, 640] | 122,880 |
| 31 | `encoder.blocks.1.project.bias` | [192] | 192 |
| 32 | `encoder.blocks.2.norm1.weight` | [192] | 192 |
| 33 | `encoder.blocks.2.norm1.bias` | [192] | 192 |
| 34 | `encoder.blocks.2.attention.qkv.weight` | [576, 192] | 110,592 |
| 35 | `encoder.blocks.2.attention.qkv.bias` | [576] | 576 |
| 36 | `encoder.blocks.2.attention.out.weight` | [192, 192] | 36,864 |
| 37 | `encoder.blocks.2.attention.out.bias` | [192] | 192 |
| 38 | `encoder.blocks.2.norm2.weight` | [192] | 192 |
| 39 | `encoder.blocks.2.norm2.bias` | [192] | 192 |
| 40 | `encoder.blocks.2.depthwise.weight` | [192, 1, 7] | 1,344 |
| 41 | `encoder.blocks.2.depthwise.bias` | [192] | 192 |
| 42 | `encoder.blocks.2.expand.weight` | [640, 192] | 122,880 |
| 43 | `encoder.blocks.2.expand.bias` | [640] | 640 |
| 44 | `encoder.blocks.2.project.weight` | [192, 640] | 122,880 |
| 45 | `encoder.blocks.2.project.bias` | [192] | 192 |
| 46 | `encoder.blocks.3.norm1.weight` | [192] | 192 |
| 47 | `encoder.blocks.3.norm1.bias` | [192] | 192 |
| 48 | `encoder.blocks.3.attention.qkv.weight` | [576, 192] | 110,592 |
| 49 | `encoder.blocks.3.attention.qkv.bias` | [576] | 576 |
| 50 | `encoder.blocks.3.attention.out.weight` | [192, 192] | 36,864 |
| 51 | `encoder.blocks.3.attention.out.bias` | [192] | 192 |
| 52 | `encoder.blocks.3.norm2.weight` | [192] | 192 |
| 53 | `encoder.blocks.3.norm2.bias` | [192] | 192 |
| 54 | `encoder.blocks.3.depthwise.weight` | [192, 1, 7] | 1,344 |
| 55 | `encoder.blocks.3.depthwise.bias` | [192] | 192 |
| 56 | `encoder.blocks.3.expand.weight` | [640, 192] | 122,880 |
| 57 | `encoder.blocks.3.expand.bias` | [640] | 640 |
| 58 | `encoder.blocks.3.project.weight` | [192, 640] | 122,880 |
| 59 | `encoder.blocks.3.project.bias` | [192] | 192 |
| 60 | `encoder.norm.weight` | [192] | 192 |
| 61 | `encoder.norm.bias` | [192] | 192 |
| 62 | `duration.conv1.weight` | [128, 192, 3] | 73,728 |
| 63 | `duration.conv1.bias` | [128] | 128 |
| 64 | `duration.conv2.weight` | [128, 128, 3] | 49,152 |
| 65 | `duration.conv2.bias` | [128] | 128 |
| 66 | `duration.norm1.weight` | [128] | 128 |
| 67 | `duration.norm1.bias` | [128] | 128 |
| 68 | `duration.norm2.weight` | [128] | 128 |
| 69 | `duration.norm2.bias` | [128] | 128 |
| 70 | `duration.out.weight` | [2, 128] | 256 |
| 71 | `duration.out.bias` | [2] | 2 |
| 72 | `frame_voices.weight` | [2, 192] | 384 |
| 73 | `frame_position.weight` | [192, 3] | 576 |
| 74 | `frame_position.bias` | [192] | 192 |
| 75 | `prosody_stack.blocks.0.norm1.weight` | [192] | 192 |
| 76 | `prosody_stack.blocks.0.norm1.bias` | [192] | 192 |
| 77 | `prosody_stack.blocks.0.attention.qkv.weight` | [576, 192] | 110,592 |
| 78 | `prosody_stack.blocks.0.attention.qkv.bias` | [576] | 576 |
| 79 | `prosody_stack.blocks.0.attention.out.weight` | [192, 192] | 36,864 |
| 80 | `prosody_stack.blocks.0.attention.out.bias` | [192] | 192 |
| 81 | `prosody_stack.blocks.0.norm2.weight` | [192] | 192 |
| 82 | `prosody_stack.blocks.0.norm2.bias` | [192] | 192 |
| 83 | `prosody_stack.blocks.0.depthwise.weight` | [192, 1, 7] | 1,344 |
| 84 | `prosody_stack.blocks.0.depthwise.bias` | [192] | 192 |
| 85 | `prosody_stack.blocks.0.expand.weight` | [640, 192] | 122,880 |
| 86 | `prosody_stack.blocks.0.expand.bias` | [640] | 640 |
| 87 | `prosody_stack.blocks.0.project.weight` | [192, 640] | 122,880 |
| 88 | `prosody_stack.blocks.0.project.bias` | [192] | 192 |
| 89 | `prosody_stack.blocks.1.norm1.weight` | [192] | 192 |
| 90 | `prosody_stack.blocks.1.norm1.bias` | [192] | 192 |
| 91 | `prosody_stack.blocks.1.attention.qkv.weight` | [576, 192] | 110,592 |
| 92 | `prosody_stack.blocks.1.attention.qkv.bias` | [576] | 576 |
| 93 | `prosody_stack.blocks.1.attention.out.weight` | [192, 192] | 36,864 |
| 94 | `prosody_stack.blocks.1.attention.out.bias` | [192] | 192 |
| 95 | `prosody_stack.blocks.1.norm2.weight` | [192] | 192 |
| 96 | `prosody_stack.blocks.1.norm2.bias` | [192] | 192 |
| 97 | `prosody_stack.blocks.1.depthwise.weight` | [192, 1, 7] | 1,344 |
| 98 | `prosody_stack.blocks.1.depthwise.bias` | [192] | 192 |
| 99 | `prosody_stack.blocks.1.expand.weight` | [640, 192] | 122,880 |
| 100 | `prosody_stack.blocks.1.expand.bias` | [640] | 640 |
| 101 | `prosody_stack.blocks.1.project.weight` | [192, 640] | 122,880 |
| 102 | `prosody_stack.blocks.1.project.bias` | [192] | 192 |
| 103 | `prosody_stack.norm.weight` | [192] | 192 |
| 104 | `prosody_stack.norm.bias` | [192] | 192 |
| 105 | `prosody_head.weight` | [3, 192] | 576 |
| 106 | `prosody_head.bias` | [3] | 3 |
| 107 | `prosody_projection.weight` | [192, 3] | 576 |
| 108 | `prosody_projection.bias` | [192] | 192 |
| 109 | `acoustic_stack.blocks.0.norm1.weight` | [192] | 192 |
| 110 | `acoustic_stack.blocks.0.norm1.bias` | [192] | 192 |
| 111 | `acoustic_stack.blocks.0.attention.qkv.weight` | [576, 192] | 110,592 |
| 112 | `acoustic_stack.blocks.0.attention.qkv.bias` | [576] | 576 |
| 113 | `acoustic_stack.blocks.0.attention.out.weight` | [192, 192] | 36,864 |
| 114 | `acoustic_stack.blocks.0.attention.out.bias` | [192] | 192 |
| 115 | `acoustic_stack.blocks.0.norm2.weight` | [192] | 192 |
| 116 | `acoustic_stack.blocks.0.norm2.bias` | [192] | 192 |
| 117 | `acoustic_stack.blocks.0.depthwise.weight` | [192, 1, 7] | 1,344 |
| 118 | `acoustic_stack.blocks.0.depthwise.bias` | [192] | 192 |
| 119 | `acoustic_stack.blocks.0.expand.weight` | [640, 192] | 122,880 |
| 120 | `acoustic_stack.blocks.0.expand.bias` | [640] | 640 |
| 121 | `acoustic_stack.blocks.0.project.weight` | [192, 640] | 122,880 |
| 122 | `acoustic_stack.blocks.0.project.bias` | [192] | 192 |
| 123 | `acoustic_stack.blocks.1.norm1.weight` | [192] | 192 |
| 124 | `acoustic_stack.blocks.1.norm1.bias` | [192] | 192 |
| 125 | `acoustic_stack.blocks.1.attention.qkv.weight` | [576, 192] | 110,592 |
| 126 | `acoustic_stack.blocks.1.attention.qkv.bias` | [576] | 576 |
| 127 | `acoustic_stack.blocks.1.attention.out.weight` | [192, 192] | 36,864 |
| 128 | `acoustic_stack.blocks.1.attention.out.bias` | [192] | 192 |
| 129 | `acoustic_stack.blocks.1.norm2.weight` | [192] | 192 |
| 130 | `acoustic_stack.blocks.1.norm2.bias` | [192] | 192 |
| 131 | `acoustic_stack.blocks.1.depthwise.weight` | [192, 1, 7] | 1,344 |
| 132 | `acoustic_stack.blocks.1.depthwise.bias` | [192] | 192 |
| 133 | `acoustic_stack.blocks.1.expand.weight` | [640, 192] | 122,880 |
| 134 | `acoustic_stack.blocks.1.expand.bias` | [640] | 640 |
| 135 | `acoustic_stack.blocks.1.project.weight` | [192, 640] | 122,880 |
| 136 | `acoustic_stack.blocks.1.project.bias` | [192] | 192 |
| 137 | `acoustic_stack.blocks.2.norm1.weight` | [192] | 192 |
| 138 | `acoustic_stack.blocks.2.norm1.bias` | [192] | 192 |
| 139 | `acoustic_stack.blocks.2.attention.qkv.weight` | [576, 192] | 110,592 |
| 140 | `acoustic_stack.blocks.2.attention.qkv.bias` | [576] | 576 |
| 141 | `acoustic_stack.blocks.2.attention.out.weight` | [192, 192] | 36,864 |
| 142 | `acoustic_stack.blocks.2.attention.out.bias` | [192] | 192 |
| 143 | `acoustic_stack.blocks.2.norm2.weight` | [192] | 192 |
| 144 | `acoustic_stack.blocks.2.norm2.bias` | [192] | 192 |
| 145 | `acoustic_stack.blocks.2.depthwise.weight` | [192, 1, 7] | 1,344 |
| 146 | `acoustic_stack.blocks.2.depthwise.bias` | [192] | 192 |
| 147 | `acoustic_stack.blocks.2.expand.weight` | [640, 192] | 122,880 |
| 148 | `acoustic_stack.blocks.2.expand.bias` | [640] | 640 |
| 149 | `acoustic_stack.blocks.2.project.weight` | [192, 640] | 122,880 |
| 150 | `acoustic_stack.blocks.2.project.bias` | [192] | 192 |
| 151 | `acoustic_stack.blocks.3.norm1.weight` | [192] | 192 |
| 152 | `acoustic_stack.blocks.3.norm1.bias` | [192] | 192 |
| 153 | `acoustic_stack.blocks.3.attention.qkv.weight` | [576, 192] | 110,592 |
| 154 | `acoustic_stack.blocks.3.attention.qkv.bias` | [576] | 576 |
| 155 | `acoustic_stack.blocks.3.attention.out.weight` | [192, 192] | 36,864 |
| 156 | `acoustic_stack.blocks.3.attention.out.bias` | [192] | 192 |
| 157 | `acoustic_stack.blocks.3.norm2.weight` | [192] | 192 |
| 158 | `acoustic_stack.blocks.3.norm2.bias` | [192] | 192 |
| 159 | `acoustic_stack.blocks.3.depthwise.weight` | [192, 1, 7] | 1,344 |
| 160 | `acoustic_stack.blocks.3.depthwise.bias` | [192] | 192 |
| 161 | `acoustic_stack.blocks.3.expand.weight` | [640, 192] | 122,880 |
| 162 | `acoustic_stack.blocks.3.expand.bias` | [640] | 640 |
| 163 | `acoustic_stack.blocks.3.project.weight` | [192, 640] | 122,880 |
| 164 | `acoustic_stack.blocks.3.project.bias` | [192] | 192 |
| 165 | `acoustic_stack.blocks.4.norm1.weight` | [192] | 192 |
| 166 | `acoustic_stack.blocks.4.norm1.bias` | [192] | 192 |
| 167 | `acoustic_stack.blocks.4.attention.qkv.weight` | [576, 192] | 110,592 |
| 168 | `acoustic_stack.blocks.4.attention.qkv.bias` | [576] | 576 |
| 169 | `acoustic_stack.blocks.4.attention.out.weight` | [192, 192] | 36,864 |
| 170 | `acoustic_stack.blocks.4.attention.out.bias` | [192] | 192 |
| 171 | `acoustic_stack.blocks.4.norm2.weight` | [192] | 192 |
| 172 | `acoustic_stack.blocks.4.norm2.bias` | [192] | 192 |
| 173 | `acoustic_stack.blocks.4.depthwise.weight` | [192, 1, 7] | 1,344 |
| 174 | `acoustic_stack.blocks.4.depthwise.bias` | [192] | 192 |
| 175 | `acoustic_stack.blocks.4.expand.weight` | [640, 192] | 122,880 |
| 176 | `acoustic_stack.blocks.4.expand.bias` | [640] | 640 |
| 177 | `acoustic_stack.blocks.4.project.weight` | [192, 640] | 122,880 |
| 178 | `acoustic_stack.blocks.4.project.bias` | [192] | 192 |
| 179 | `acoustic_stack.norm.weight` | [192] | 192 |
| 180 | `acoustic_stack.norm.bias` | [192] | 192 |
| 181 | `vocoder.harmonic.weight` | [98, 192] | 18,816 |
| 182 | `vocoder.harmonic.bias` | [98] | 98 |
| 183 | `vocoder.noise.weight` | [36, 192] | 6,912 |
| 184 | `vocoder.noise.bias` | [36] | 36 |

</details>

## Appendix B: reproduction commands

```sh
# Upstream training stack
git clone https://github.com/n4ze3m/tiny-voca /tmp/tiny-voca-upstream
git clone https://github.com/n4ze3m/dizione /tmp/dizione-upstream
cd /tmp/tiny-voca-upstream
git log -1 --oneline          # e7f1bde
cd /tmp/dizione-upstream
git log -1 --oneline          # 8cd889e

# Released config + checkpoint (18 MB)
curl -fsSL -O https://huggingface.co/n4ze3m/tiny-voca/resolve/main/config.json
curl -fsSL -O https://huggingface.co/n4ze3m/tiny-voca/resolve/main/tiny-voca.pt
shasum -a 256 tiny-voca.pt   # ebb22871524c5732520fb1652e6aefcec708a7452e83bdd00adcf6181db5ef70

# Rust checks (from this worktree)
cd /Users/suheugene/projects/22-tiniest-voca-ru/worktrees/recon
cargo test --workspace
cargo build --release -p tiny-voca --bin voca

# Reference model parity harness (needs PyTorch + weights)
# DIZIONE_DATA=weights/phonemizer python scripts/dump_reference.py weights/tiny-voca.pt bench/sentences.txt out/reference
# cargo run --release -p tiny-voca --bin parity -- weights/tiny-voca.safetensors out/reference
```

Note: this reconnaissance did not run `cargo test`, `cargo build`, training, or
audio synthesis; it read code and inspected the released checkpoint. The
`docs/repo-map.md` claims above are code/checkpoint facts; runtime claims such
as MPS support are explicitly flagged as unverified.
