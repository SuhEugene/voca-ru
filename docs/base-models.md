# Base models for a Russian `voca`

**Status:** R3 decision record, revision 2 (scope updated to the synthetic-teacher recipe).
**Date checked:** 2026-11-19. All URLs below were fetched live on that date; no model weights were downloaded.
**Environment:** Apple M5 Pro, 24 GB unified memory, macOS, no CUDA.
**Worktree:** `worktrees/base-models`, branch `task/base-models`.

---

## 0. Decision in one page

**Default: retrain the 4.5M `tiny-voca` architecture from scratch on ~16–20 hours of synthetic Russian speech generated with Qwen3-TTS-12Hz-1.7B VoiceDesign + Base voice clone.** Keep the 24 kHz source-filter vocoder and the 76-token inventory shape, replace the frontend with Russian phonemes + stress, and train with upstream's recipe (100k steps, GAN from 8k). This is path **A**.

Why:

- It is the only path that preserves the product: 4.54M parameters, 18 MB weights, 24 kHz, ~7–30 ms model time on Apple Silicon, and a pure-Rust inference path with no ONNX/PyTorch runtime.
- The recipe is proven on this exact architecture by two independent 2026 ports: Brazilian Portuguese with Kokoro-82M as teacher (16.37 h accepted, 100k steps, 5 h 10 min on one rented GPU, 1.44% WER on 32 novel clips, listener 4/5) and Polish with OmniVoice (≈10k sentences, 104k steps, 8.9 h on one 24 GB GPU, 0.37–0.48% CER on 50 clips). The pt-BR port is Apache-2.0 code and weights; the Polish port is Apache-2.0 code but **CC BY-NC 4.0 weights/data**, so only its recipe is reusable here.
- Qwen3-TTS 1.7B officially supports Russian and is the best-documented permissively licensed Russian teacher found: 3.212% WER and 0.792 speaker similarity on the authors' multilingual test set, better than ElevenLabs (3.878 / 0.676) and MiniMax (4.281 / 0.761) in that table.
- The complete upstream project cost $9.56 of rented GPU time for 19.1 h of data plus training; a Russian repeat is cheap.

The **critical path is not the acoustic model**. It is the Russian frontend:

1. **Stress.** Russian stress is lexical and unwritten. Qwen3-TTS Base does **not** reliably follow stress marks: the model card for the Apache-2.0 Russian stress adapter reports 65% adherence on-dictionary and 17% against-dictionary for base Qwen; on ordinary text roughly 15% of words are stressed differently from RUAccent. With the adapter and U+0301 marks this becomes 100% / 96% adherence, leaving RUAccent's own ~3% error floor. A 20 h corpus generated without this mitigation would teach the student wrong stress.
2. **Phonemes + alignment.** `voca` consumes phonemes, not characters. A permissive Russian G2P and a phoneme→aligner label map must be built. eSpeak NG `ru` has a 48-phoneme inventory and is usable **only at annotation time** (GPL-3.0); the shipped Rust frontend must be new, permissive code.
3. **Size.** RUAccent is MIT and its on-disk data is modest (21 MB compressed dictionary + 0.8 MB accent model + 10.5 MB omograph model + tokenizers), but its Python loader expands the dictionary to ~1.5 GB RAM. A Rust loader must be written for a <300 MB RSS target. This, not the 18 MB TTS model, is the real binary-size/RAM regression.

**What would make me argue against the default:** a native-listener voice audition that finds the VoiceDesign voice accented; a stress gate above 3–5% word errors after RUAccent + adapter; a teacher WER above 5% after screening; or a frontend that cannot map eSpeak `ru` labels to the forced aligner for `lʲ`, `zʲ`, `ɕː`. In those cases use real Russian audiobook speech (SOVA RuAudiobooksDevices 298 h or Russian LibriSpeech 98 h, both CC BY 4.0 / public domain) or a permissive Russian TTS as teacher (Silero `v5_cis_base`, MIT; Piper `ru_RU-denis`, CC0 chain), before falling back to a second runtime.

**Fallback final model, if the 4.5M retrain misses the quality gates:** UtrobinTTS low multispeaker (`utrobinmv/tts_ru_free_hf_vits_low_multispeaker`, Apache-2.0, 15.1M, 50.8 MB ONNX, 16 kHz) first, then Piper `ru_RU-denis-medium` (CC0 voice chain, 15.8M, 63.2 MB ONNX, 22.05 kHz). Both require a VITS/ONNX runtime and break the dependency-free, 4.5M, 24 kHz identity.

---

## 1. What `voca` actually is (verified, with one correction to the brief)

| Fact | Value | Source |
|---|---|---|
| Parameters at inference | 4,537,291 | [tiny-voca `docs/model.md`](https://github.com/n4ze3m/tiny-voca/blob/main/docs/model.md) |
| Audio | 24 kHz mono, hop 256 (10.7 ms) | [HF `config.json`](https://huggingface.co/n4ze3m/tiny-voca/blob/main/config.json) |
| Token inventory | 76 symbols: 5 specials, 12 punctuation, 26 consonants, 33 vowels | upstream `src/tiny_voca/text.py`, Rust `crates/dizione/src/symbols.rs` |
| Architecture | 192 channels, 4 heads, 4 encoder blocks, 2 prosody blocks, 5 acoustic blocks, source-filter vocoder (25,862 params) | upstream `docs/model.md` |
| Weights | 18 MB safetensors, **loaded from `$VOCA_HOME` at runtime**, not compiled into the binary | `crates/tiny-voca/src/tts.rs::Tts::load`, `crates/tiny-voca/src/bin/voca.rs::home` |
| Licences | Code and weights Apache-2.0; dataset CC BY 4.0; dizione Apache-2.0, CMUdict-derived data | [HF model card](https://huggingface.co/n4ze3m/tiny-voca), [dataset card](https://huggingface.co/datasets/n4ze3m/tiny-voca), [dizione](https://github.com/n4ze3m/dizione) |
| English WER claim | 0.6–0.9% on 8-clip validation/test sets (sanity check, not a benchmark) | upstream `docs/model.md` |

**Correction to the brief:** the weights are not compiled into the binary. `Tts::load(weights, phonemizer_data)` reads a safetensors file and a phonemizer directory; the installer downloads a 19 MB `voca-data.tar.gz`. A Russian model is therefore a data-file swap plus a frontend change, not a rebuild of baked-in tensors.

### 1.1 The Rust integration surface

Three constants in `crates/tiny-voca/src/model.rs` tie the binary to the English inventory:

```rust
const SYMBOL_COUNT: usize = 76;
const FIRST_PHONE: u32 = 17;
const FIRST_VOWEL: u32 = 43;
```

`Config::from_json` rejects any weights file whose `vocab_size` is not 76. The actual symbol list and tokenizer live in `crates/dizione/src/symbols.rs` (`pub const SYMBOLS: [&str; 76]`). `TinyVoca::load` reads every tensor shape from the weights file; the network code itself has no English-specific shapes.

Consequences:

- **If the Russian inventory is also 76 symbols**, the acoustic model code needs no change. Replace the `SYMBOLS` table, adjust `FIRST_PHONE`/`FIRST_VOWEL`, replace the dizione phonemizer with a Russian one, and ship Russian weights. This is the recommended shape because it also lets path B (`--init-from`) work without patching upstream's config-equality check.
- **If the inventory size changes**, `SYMBOL_COUNT` changes and `Config::from_json` must accept the new size; the network code still does not change. A dual-language binary would need runtime symbol tables instead of `&'static str` constants.
- **Stress is a parallel 3-value feature** (`0 none, 1 primary, 2 secondary`), not a token. Russian stress fits this exactly.

The frontend is the only large new subsystem. dizione is Apache-2.0 and English/CMUdict-specific; it cannot be extended to Russian without replacing its data and rules.

---

## 2. Path A — retrain the architecture on synthetic Russian Qwen audio (default)

### 2.1 Qwen3-TTS facts the plan depends on

All checked 2026-11-19.

| Fact | Qwen3-TTS-12Hz-1.7B-VoiceDesign | Qwen3-TTS-12Hz-1.7B-Base |
|---|---|---|
| HF repo | [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) |
| Licence | Apache-2.0 (card frontmatter) | Apache-2.0 (card frontmatter) |
| Speech tokenizer | [`Qwen3-TTS-Tokenizer-12Hz`](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz), Apache-2.0, 682 MB | same |
| Russian officially supported? | Yes — listed in the VoiceDesign row of the README model table | Yes — listed in the Base row |
| On-disk size | 3.83 GB `model.safetensors` + 0.68 GB speech tokenizer | 3.86 GB `model.safetensors` + 0.68 GB speech tokenizer |
| Output | 24 kHz mono (tokenizer `output_sample_rate: 24000`) | 24 kHz mono |
| API | `generate_voice_design(text, language="Russian", instruct=...)`; batch via lists | `create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)` then `generate_voice_clone(text, language="Russian", voice_clone_prompt=...)`; batch via lists |
| Reference audio | n/a | 3 s is the advertised minimum ("3-second rapid voice clone"); 10–20 s is the practical recommendation. `x_vector_only_mode=True` drops `ref_text` but lowers quality. |
| Max output | `generation_config.json` default `max_new_tokens: 8192`; the evaluation uses 2048. At 12.5 Hz, 2048 ≈ 164 s, 8192 ≈ 655 s. Per-sentence clips are the intended use. | same |
| Attention / hardware | FlashAttention 2 recommended but optional; examples use `device_map="cuda:0"`, bf16. Public Russian benchmark used an RTX 3090 Ti 24 GB. | same |
| Apple Silicon | No official MPS path. PRs [#124](https://github.com/QwenLM/Qwen3-TTS/issues/124) and [#345](https://github.com/QwenLM/Qwen3-TTS/pull/345) add MPS/MLX device detection but are **open, not merged**. CPU works in PyTorch but is not practical for hours of generation. | same |
| Smaller alternative | 0.6B: roughly half the weights + same tokenizer; Russian WER 4.458, SIM 0.781 | 0.6B: same, but 1.7B is better Russian |

**Practical hardware conclusion:** rent one 24 GB NVIDIA GPU (3090 Ti / 4090 / A10) for teacher generation. Do not plan the 20 h generation pass on the M5's CPU; it is a multi-day CPU job at best. The tokenizer is 24 kHz, so generated audio drops straight into `tiny-voca`'s feature pipeline with no resampling.

**Batch/prompt details:** VoiceDesign is used once per voice to make a short reference clip from an English or Russian description. Base then builds a reusable `voice_clone_prompt` from that clip + its transcript and generates all sentences with the same prompt. This avoids recomputing the speaker embedding per sentence and is the workflow Qwen documents under "Voice Design then Clone". Text/language/instruct can be lists for batch generation.

### 2.2 Teacher quality for Russian: what is actually published

| Item | Number | Source |
|---|---|---|
| Qwen3-TTS-12Hz-1.7B-Base Russian WER | **3.212%** | [Qwen3-TTS README, multilingual table](https://github.com/QwenLM/Qwen3-TTS#evaluation) |
| Qwen3-TTS-12Hz-1.7B-Base Russian speaker similarity | **0.792** | same table |
| Qwen3-TTS-12Hz-0.6B-Base Russian WER / SIM | 4.458% / 0.781 | same |
| ElevenLabs Russian WER / SIM (same table) | 3.878% / 0.676 | same |
| MiniMax Russian WER / SIM | 4.281% / 0.761 | same |
| Qwen3-TTS Base adherence to a stress mark on the dictionary vowel | 65% | [siriusfreak stress adapter card](https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf) |
| Qwen3-TTS Base adherence to a stress mark **against** the dictionary | 17% | same |
| Adapter adherence, on-dictionary / against | **100% / 96%** | same |
| Adapter departure from RUAccent marks on ordinary text | 0.1% of words | same |
| Adapter CER through Whisper on ordinary text | 2.9% (base: 3.6%) | same |
| No Russian preset voice in CustomVoice | 9 built-in voices, none Russian | [Russian Qwen3-TTS guide](https://aipolezno.ru/posts/qwen3-tts-webui-russkiy), [review](https://ya.zerocoder.ru/kak-prevratit-tekst-v-zhivoj-golos-s-pomoshhju-qwen3-tts) |

**Stress is the known failure mode.** [Discussion #185](https://github.com/QwenLM/Qwen3-TTS/discussions/185) reports that Russian text is often stressed incorrectly, that the base model does not understand `+`, `'` or combining-acute marks, and that examples such as `за́мок`/`замо́к` are mangled. A Russian review of 38 phrases reports failures on numbers with spaces in digit groups, abbreviations next to Latin text, and unit abbreviations (`97 мс` → "97 муэмс"), plus no native Russian preset voice. A positive review still puts Qwen's Russian WER ahead of ElevenLabs and MiniMax, which is consistent with the authors' table.

**No published Russian MOS was found.** The numbers above are WER/SIM, not listener ratings. Treat "Qwen Russian sounds good" as a claim to audition, not a verified MOS.

**Voice-design risk, from the closest precedent.** The author of the Brazilian Portuguese Kokoro port writes that voices upstream's teacher (Qwen3-TTS) designed from English or Portuguese descriptions "did not sound Brazilian to a Brazilian listener", and that most Qwen/Chatterbox clones failed too; Kokoro's own Brazilian voices passed ([port README](https://github.com/Haple/tiny-voca-ptbr-kokoro#how-it-was-made)). Qwen does officially support Russian and Russian reviewers are positive, so this is not a rejection — but it is a direct warning that a VoiceDesign voice must be auditioned by a native speaker before 20 h are generated.

**Mitigations, in order:**

1. Design 2–3 candidate voices with VoiceDesign using explicit Russian descriptions; audition all of them on 10 sentences with a native listener. Keep the best two. If none sounds native, do not force it.
2. If VoiceDesign is accented, clone a **permissive synthetic Russian reference** instead: Silero `v5_cis_base` (MIT) or Piper `ru_RU-denis` (CC0 voice-data chain) rendered on a few sentences. This keeps the teacher's output licence clean and avoids cloning a real person.
3. Use the Apache-2.0 stress adapter (or retrain the same recipe on Russian LibriSpeech, CC BY 4.0) and mark every word with RUAccent U+0301 before generation.
4. Normalize numbers, dates, money, units and abbreviations **before** Qwen; do not rely on the model's text normalization.
5. Screen every clip with a Russian ASR (Whisper large-v3 / faster-whisper) and, on a sample, with a stress detector. Reject, do not "fix in post".

### 2.3 The data recipe

Target: **16–20 h accepted**, two voices, 24 kHz, per-sentence clips of 2–15 s.

| Step | Choice | Evidence / reason |
|---|---|---|
| Sentence corpus | 8,000–10,000 Russian sentences: Common Voice sentence-collector (CC0, now distributed via Mozilla Data Collective), Russian LibriSpeech text (CC BY 4.0 / public-domain LibriVox), LLM-written everyday/dialogue/narration, and templates for numbers, dates, money, names, questions | Kokoro port used 8,945 sentences from the same mix; Polish used ≈10k |
| Stress marking | RUAccent, MIT; convert `+` before the vowel to U+0301 after it | Adapter card's recommended workflow; RUAccent 0.972 non-homograph / 0.96 homograph |
| Teacher | Qwen3-TTS-12Hz-1.7B-VoiceDesign for 2 reference voices, then Base + stress adapter for all sentences | Official Russian support; 3.212% WER; Apache-2.0 |
| Screening | Whisper large-v3-turbo Russian; reject >5–8% WER (allow one wrong word), implausible duration/rate/silence; spot-check stress | Kokoro port: 17,890 targets → 17,433 accepted → 16.37 h, 457 rejected, 15% WER gate |
| Accepted target | ≈16–20 h after rejection | Upstream English 17.6 h; Kokoro 16.37 h |
| Cost | 30–70 GPU-hours of 1.7B generation, ≈$10–40 at 4090/3090 rates; upstream's whole English project was $9.56 | [tiny-voca cost note](https://github.com/n4ze3m/tiny-voca#cost); [Russian Qwen benchmark RTF 2.8–3.3](https://github.com/meshoklv/ru-speech-bench) |

**Text licensing.** Common Voice sentences are CC0; LibriVox/RuLS is public domain in the USA with the HF mirror declaring CC BY 4.0; SOVA is CC BY 4.0. Avoid modern copyrighted books. Qwen's model and code are Apache-2.0; the cards state no additional output restriction, but the usual impersonation/consent rules apply. The VoiceDesign route creates synthetic voices, so no real speaker's personality rights are involved.

**Why 16–20 h, not 100 h:** the model has 4.54M parameters and was designed for 17.6 h. More data mainly adds speaker/acoustic variety, which this product does not need (two fixed voices). The Kokoro and Polish ports both stop at 16–17 h and ≈100k steps.

### 2.4 Distillation precedent (this matters more than the Piper/Silero comparison)

| Port | Teacher | Accepted data | Training | Result |
|---|---|---|---|---|
| [`Haple/tiny-voca-ptbr-kokoro`](https://huggingface.co/Haple/tiny-voca-ptbr-kokoro) | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0 | 17,890 target clips → 17,433 accepted → **16.37 h**; 8,945 sentences; 2 voices | 100k steps, **5 h 10 min** on one rented GPU (RunPod), upstream recipe | **1.44% WER** on 16 novel sentences × 2 voices (32 clips); held-out test 0.0%; listener **4/5**; 4.54M params, 21 MB ONNX |
| [`damaon/slyszeglosy`](https://github.com/damaon/slyszeglosy) (Polish) | OmniVoice, two designed voices | ≈10k sentences from Common Voice PL CC0 + own + templates | 104k steps, **8.9 h** on one 24 GB GPU | **0.37–0.48% CER** on 50 test clips; 4.53M params; code Apache-2.0, **weights/data CC BY-NC 4.0** |
| Arabic Halabi port ([repo](https://github.com/BenkadourKhaled/tiny-voca/tree/arabic-halabi-f2)) | from scratch, Halabi CC BY 4.0 | 1,214 utterances, 2.85 h | 3,000 steps, CPU | 4.53M params, 54 symbols, 1 voice; works from a small corpus |
| [`77ethers/alpa`](https://huggingface.co/77ethers/alpa) (Sanskrit) | English weights transplanted, then fine-tuned | 3.5 h | 3,000 steps, 100 min on 4 CPU cores | ~3% phoneme error; starts with an English accent; proves weight transplant on this architecture |
| Upstream English | Qwen3-TTS 1.7B | 17.6 h | 104,737 steps, 7.7 h on a 4090 | 0.6–0.9% WER on 8-clip sanity sets |

The two synthetic-teacher ports are the real evidence: **15–17 h of teacher audio, two voices, 100k steps, 5–9 GPU-hours, and a 4.5M student that is intelligible and, in the pt-BR case, preferred over a Magpie-TTS variant in a blind A/B.** The Russian repeat fits this envelope.

### 2.5 Russian frontend: the actual critical path

#### 2.5.1 Inventory: make it 76 symbols

A 76-symbol Russian inventory keeps `vocab_size` unchanged, so `Config::from_json` accepts the Russian weights, the Rust network code is untouched, and path B can use upstream's `--init-from` without patching the config check.

Concrete budget:

- 5 specials (`<pad> <bos> <eos> <sp> <unk>`)
- 23 punctuation marks: the English 12 (`, . ? ! ; : " ' ( ) — …`) plus 11 typographic/utility marks (`« » „ “ ” – / № % ° ±`)
- **48 Russian phonemes**, matching eSpeak NG `ru`'s `phsource/ph_russian` inventory (`g p x k l l^ s s; n_ n# r ts dZ; tS; t t; d d; m; I^ i I# y I I2 I3 E E# E2 E3 E% o O u u# u" 8 e ju ja ja# V# Y A a V a# R4` — 48 definitions).

This is not an accident: 5 + 23 + 48 = 76. The exact IPA spellings can be cleaned up (eSpeak's `--ipa` output uses nonstandard `ɭ`, `ɵ`, `ʌ` and emits stray `^` artifacts); define the student's symbols as clean IPA and map the 48 eSpeak codes to them. The mapping table is the contract between annotation and inference.

#### 2.5.2 eSpeak NG `ru`: what it is and what it is not

- Inventory: **48 phoneme definitions** in [`phsource/ph_russian`](https://github.com/espeak-ng/espeak-ng/blob/master/phsource/ph_russian); stress is a separate `'` marker; palatalization is `;` (e.g. `l;`, `s;`, `t;`).
- Licence: **GPL-3.0-or-later** ([repo](https://github.com/espeak-ng/espeak-ng), `COPYING`). It can be called as an external program during **annotation only**, exactly as upstream did for its first 66k steps. It must not be linked, ported, or shipped in the Rust binary.
- Quality: rule-based; it does not reduce unstressed vowels in the way a careful Russian G2P does, and it writes `ы` as `y` in its internal alphabet. The `Russian-G2P` benchmark reports eSpeak-ng `ru` at 24/604 exact words and 45.8% sentence PER on a Wiktionary-derived gold set — but that comparison penalizes notational differences, so treat it as "not a drop-in final frontend", not as a pronunciation verdict.
- Homographs: `замок` is read `zamˈok` (lock) with no context; eSpeak does not resolve `за́мок`/`замо́к`. This is another reason the final frontend must take stress from RUAccent, not from eSpeak.

Verdict: use eSpeak only to produce the **annotation label sequence** during training. The shipped frontend is new code.

#### 2.5.3 Forced alignment

Upstream uses [`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) (Apache-2.0, 392 output labels) to align eSpeak phonemes to audio. The Kokoro pt-BR port used the same aligner and mapped 12 Brazilian symbols to the labels it expects; that is the proven pattern.

For Russian:

- The aligner vocabulary contains most Russian IPA: `p b t d k ɡ f v s z ʃ ʒ x h ts tʃ m n l r j`, palatalized `pʲ bʲ tʲ dʲ kʲ ɡʲ fʲ vʲ sʲ mʲ nʲ rʲ`, `ɕ ʑ ʂ ʐ`, `ɨ ɵ ɐ ɪ ʊ ə a e i o u ɛ ɔ`, `ʲ`, and stress is not a label (durations only).
- It does **not** contain `lʲ`, `zʲ`, or long `ɕː`/`ʑː` as single labels. Map those eSpeak phones to two aligner labels (`l` + `ʲ`, `z` + `ʲ`) or to the short label (`ɕ`, `ʑ`), then merge/split durations into the student's inventory.
- Build a 48-row `ESPEAK_TO_ALIGNER` table (each row 1–2 labels), align the mapped sequence, and merge spans back to the 48 student symbols. Every student phone must have a mapping; add a unit test that asserts coverage of all 48.
- Calibrate onset offset and silence thresholds on 50–100 Russian clips, as the Polish port did. Expect 1–3% alignment rejects; drop them.

This is a bounded 1–2 day task, not a research project. It is also the point where path A can fail visibly: if `lʲ`/`zʲ` alignment is poor, the student will merge `лук`/`люк`. Test minimal pairs before scaling.

#### 2.5.4 Stress: RUAccent is the best permissive option, with a size caveat

| Option | Licence | Quality | Size / runtime | Verdict |
|---|---|---|---|---|
| [RUAccent](https://github.com/Den4ikAI/ruaccent) / [`ruaccent/accentuator`](https://huggingface.co/ruaccent/accentuator) | **MIT** (repo and HF card) | 0.972 non-homograph, 0.96 homograph (paper); 0.947 word / 0.833 homograph on an independent benchmark | 21 MB gzip dictionary + 0.8 MB accent ONNX + 10.5 MB omograph ONNX + tokenizers; Python loader RSS up to ~1.5 GB (int8 models ~313 MB RSS) | **Best.** Port the ONNX models to Rust or run via `tract`; build a compact dictionary loader |
| [Silero Stress](https://github.com/snakers4/silero-stress) | **MIT** | 0.9629 independent; 4M words 100%, homographs F1 0.92 | ~50 MB, PyTorch | Good accuracy, but adds PyTorch to the build; no official ONNX |
| [`nsu-ai/russian_g2p`](https://github.com/nsu-ai/russian_g2p) | MIT **code**; dictionary provenance unclear | Unknown on modern text; old (2018) | 54 MB `Accents.json` + 64 MB forms; pure Python | Useful as a rules reference / MIT-licensed port source; replace its dictionary with RUAccent's |
| [`maxmelichov/Russian-G2P`](https://github.com/maxmelichov/Russian-G2P) | **No licence file** (GitHub API returns none) | Claims 604/604 words, 67/67 sentences on its own gold set | 375 MB lexicon, ~1.5 GB RAM, ONNX | **Do not depend on it.** Use its gold-set methodology as a reference only |
| eSpeak NG `ru` | GPL-3.0 | Weak on reduction/homographs | tiny | Annotation only; GPL |
| Wiktionary-derived stress | CC BY-SA | Good coverage, incomplete homographs | varies | Share-alike contaminates shipped data; avoid unless the project accepts CC BY-SA |
| Qwen stress adapter [`siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf`](https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf) | Apache-2.0 weights | 100% / 96% mark adherence; 0.1% departure from RUAccent | 3.86 GB merged or 35 MB LoRA | **Use for teacher generation.** Its card flags training-data caveats (`biggest-ru-book` has no licence; meter2 trained on Silero NC); retrain the adapter on RuLS CC BY 4.0 for a clean commercial release |

**RUAccent vs the student's size promise.** The compressed artefacts are ~33 MB, which is acceptable next to an 18 MB model. The Python loader's 1.5 GB RSS is not. A Rust implementation should:

- memory-map/decompress the 21 MB accent dictionary into a compact hash map or a sorted trie, not a Python dict of 3.8M forms;
- load the 0.8 MB accent ONNX and 10.5 MB omograph ONNX (or port the 4-layer/128-hidden RoFormer to Rust);
- target <300 MB RSS and <50 MB on disk; measure on the M5.

If that cannot be met, train a compact character-level stress predictor (1–5M params) on RuLS + RUAccent labels, quantize it, and compare against RUAccent on a 1,000-word gold set. Do not silently drop stress accuracy to save memory; wrong stress is the most audible Russian TTS defect.

#### 2.5.5 The Rust G2P

Russian orthography is mostly phonemic, so a rule-based Cyrillic→IPA transducer + RUAccent stress is viable:

- vowel reduction (akanye/ikanye) conditioned on stress;
- final devoicing and regressive voicing assimilation;
- `г` → `v` in `-ого`/`-его`; `ч` → `ʃ` in `что`, `конечно`; `щ` → `ɕː`;
- `ь`/`ъ` and iotated vowels (`е ё ю я`);
- silent letters (`солнце`, `сердце`, `здравствуйте`);
- `и`/`ы` after hard/soft consonants;
- a small exception lexicon for frequent names/loanwords.

Build a 200–500-word gold set from Wiktionary's Russian IPA (check the CC BY-SA implications for the test data only; do not ship Wiktionary-derived dictionaries) plus RUAccent's own test sentences. Gate: ≥98% phoneme exact, ≥97% stress on polysyllables, 100% coverage of the 48 symbols, and <0.5% token mismatch between the Python annotation path and the Rust inference path over 5,000 sentences.

`nsu-ai/russian_g2p` (MIT) can be used as a reference implementation to port or crib rules from, provided its dictionary is replaced with RUAccent's. Do not copy GPL eSpeak rules.

### 2.6 Training on the M5 Pro

What the upstream trainer supports (verified in `src/tiny_voca/train.py`):

- `--device cpu` works; `--max-seconds`, `--resume`, `--init-from`, `--new-data` are present.
- The waveform head uses `torch.stft(..., return_complex=True)`, `torch.fft.rfft`, and `torch.polar`. MPS does not support complex tensors reliably, and there is no MPS-specific path in the code. **Plan CPU, not MPS.** A GPU rent is the fast path.
- Batch defaults: 9,000 frames, max 32 utterances; the model is 4.5M, so <10 GB VRAM is expected. A 24 GB GPU is more than enough; the 4090 in upstream's run was not the constraint.

| Plan | Hardware | Estimate |
|---|---|---|
| Pilot | M5 Pro CPU, 1–2 h data, 1,000 steps | measure steps/s; expect minutes per 1k steps at this model size, but the vocoder/STFT dominates |
| Full local | M5 Pro CPU, 16–20 h data, 100k steps | **30–80 h** (extrapolated 5–10× the 4090's 5–9 h; must be measured) |
| Full rented | RTX 4090 / 3090 Ti 24 GB, 100k steps | **5–9 h**, ≈$10–30; matches pt-BR (5 h 10) and Polish (8.9 h) |
| Teacher generation | same rented GPU, Qwen 1.7B | 30–70 GPU-hours for 20 h raw audio, ≈$10–40 |

The M5 plan is viable but slow: run a 1k-step pilot first, then decide whether 30–80 h local is acceptable or a $20 rental is cheaper. Do not start a 100k-step run without the pilot number.

### 2.7 Quality ceiling and acceptance gates

Expected, based on the two ports and the teacher's 3.212% WER:

- **Clarity:** 2–6% WER/CER on held-out Russian sentences. That is in Piper denis territory (3.7% CER) and below Silero's best (0.7–1.0% CER), but with better prosody than Piper.
- **Naturalness:** likely below Piper/Silero/XTTS/F5; a 4.5M model with a 16–20 h synthetic corpus has a hard ceiling. The pt-BR port's single-listener 4/5 is the realistic target; do not promise a MOS.
- **Stress:** RUAccent's ~3% word error floor, plus the student's own errors. Expect audible stress mistakes in long sentences; curate the evaluation set around homographs and names.
- **Speed:** same architecture, so ~7–30 ms model time for 2–6 s of audio on M5 with Accelerate; whole-command ~24–45 ms. The frontend (stress dictionary + G2P) may add 1–10 ms per sentence.

Acceptance gates before calling it done:

1. Native listener MOS ≥3.5/5 on 30 sentences, two voices. One listener is a smoke test; three is a real gate.
2. ASR WER ≤5% on 50 held-out RuLS sentences (not in training), with text normalization applied to both sides.
3. Stress error ≤3% on 200 polysyllabic words with RUAccent marks; ≤1% on homographs in context.
4. No repeated/hallucinated words, no truncated endings, no more than 1% clips with clicks/dropouts.
5. `voca` inference stays under 50 ms for a 6 s sentence on one M5 thread (model + frontend).

### 2.8 What would make me argue against the default

- **VoiceDesign accent gate fails.** If 2–3 designed voices do not sound native to a Russian listener, do not generate 20 h. Clone a permissive synthetic reference (Silero `v5_cis_base` MIT or Piper denis CC0) or use real Russian audiobook speech.
- **Stress gate fails.** If RUAccent + adapter still produces >3–5% wrong stress on a 200-word audit, the teacher is not usable for a stress-sensitive product. Switch to real Russian audio (RuLS/SOVA) or a teacher with better stress (Piper denis; Silero `v5_cis_base` + RUAccent).
- **Teacher WER gate fails.** If screened Qwen audio is >5% WER after normalization, the student will inherit it. Switch teacher or use real data.
- **Frontend mapping fails.** If `lʲ`/`zʲ` alignment cannot be made reliable, switch to a character-based student (larger change) or a different aligner/dictionary. Do not ship a model that merges `лук`/`люк`.
- **Size/RAM budget fails.** If the Rust stress frontend cannot fit under ~50 MB disk / ~300 MB RSS, the product's "one small binary" promise is broken. A compact stress predictor is then required before shipping.
- **Cost/time gate fails.** If teacher generation is >100 GPU-hours or training >2 weeks local, re-scope to a 10 h corpus and 50k steps, or rent.

---

## 3. Comparison table: A vs B vs C

Columns are the ones the brief requires. "Integration" means work to put the model inside the dependency-free Rust binary, not work to evaluate it.

| Path | Model / artefact | Params | Licence | Fine-tune / train VRAM-RAM | Data hours | Russian quality ceiling | Inference on Apple Silicon | Integration into Rust binary | Verdict |
|---|---|---:|---|---|---|---|---|---|---|
| **A. Retrain tiny-voca on synthetic Russian** | new 4.54M checkpoint, same architecture, 24 kHz | 4.54M | Apache-2.0 (code/weights); teacher Qwen Apache-2.0; data CC0/CC BY | <10 GB VRAM on GPU; CPU training works, 24 GB unified is ample | **16–20 h** synthetic | 2–6% WER/CER; naturalness below Piper/Silero; pt-BR analogue 1.44% WER, 4/5 listener | ~7–30 ms model time; ~24–45 ms whole command; 18–21 MB | **Low–medium:** replace 76-symbol table + frontend crate; model code unchanged | **Recommended** |
| **B. Fine-tune English weights on Russian** | `n4ze3m/tiny-voca` + `--init-from` | 4.54M | Apache-2.0 | <10 GB; CPU 2–6 h for 3–10k steps | **3–10 h** | 4–10% WER/CER, audible English accent; good as a 1-day baseline | same as A | **Low:** same inventory, `--init-from`; transplant shared phone rows | Useful baseline, not final |
| **C1. Piper `ru_RU-denis-medium`** | [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices/tree/main/ru/ru_RU/denis/medium), VITS ONNX | ≈15.8M (63.2 MB) | Voice data **CC0**; base `rdh` CC0; engine GPL-3.0 (new) / MIT (old) | fine-tune 8–12 GB GPU; not local-friendly | 0 | CER 3.7%, UTMOS 3.06; clear, robotic | RTF ~0.2–0.65 on old Xeon; on M5 faster, but ~5–10× slower than voca; ~239 MB RSS | **High:** VITS/ONNX runtime + eSpeak `ru` (GPL) or a new G2P; 63 MB model | Best licence-clean Piper voice; fallback |
| **C2. UtrobinTTS low multispeaker** | [`utrobinmv/tts_ru_free_hf_vits_low_multispeaker`](https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_low_multispeaker), ONNX | 15.1M (50.8 MB) | **Apache-2.0** | 8–12 GB GPU | 0 | CER 2.1%, UTMOS 2.85–3.19; 2 voices; 16 kHz | RTF ~0.13 CPU on old Xeon; ~7.5× real time; low quality tier | **High:** ONNX/VITS runtime; char tokenizer; RUAccent; 16 kHz | Best permissive fallback model |
| **C3. Silero `v5_cis_base`** | [`silero-models`](https://github.com/snakers4/silero-models), MIT CIS base | ≈22.9M (91.7 MB) | **MIT** (`LICENSE_CIS`) | PyTorch CPU/GPU; fine-tune not local | 0 | Not separately benchmarked for Russian; same family as Silero, clarity good; needs external stress | RTF ~0.05–0.13 CPU (v5 class); ~300 MB RSS | **Very high:** PyTorch/JIT, no official ONNX; 92 MB + LibTorch | Best permissive teacher fallback |
| **C4. Silero `v5_ru`** | [`silero-models`](https://github.com/snakers4/silero-models), 5 Russian voices | ≈36.3M (145.4 MB) | **CC BY-NC 4.0** (main repo licence) | PyTorch; fine-tune not local | 0 | CER 0.7–1.0% (v3/v4), UTMOS 1.75–2.98; best clarity, noncommercial | RTF ~0.05–0.13 CPU; ~500 MB RSS | **Very high:** PyTorch | Research only; licence blocks shipping |
| **C5. XTTS-v2** | [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) | ≈520M total (467M GPT + 52M DVAE; 1.87 GB + 210 MB) | **CPML 1.0.0, noncommercial**; outputs also noncommercial | 12–24 GB GPU; not local | 0 | CER 2.7%, UTMOS 3.04, WER 6.4%; clone SIM 0.76; needs 6 s reference | RTF ~0.35 GPU, ~1.5 CPU; 2 GB weights | **Impossible** in a small Rust binary | Reject: licence + size |
| **C6. F5-TTS Russian** | [`Misha24-10/F5-TTS_RUSSIAN`](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN) | 336M (1.35 GB inference) | **CC BY-NC 4.0**; code MIT | 16–24 GB GPU | 0 (5,000 h pretraining) | WER 3.2%, best clone SIM 0.70–0.75; needs reference audio + transcript | RTF ~0.2–0.3 GPU; CPU unusable | **Impossible** | Reject: licence + size |
| **C7. MMS-TTS Russian** | [`facebook/mms-tts-rus`](https://huggingface.co/facebook/mms-tts-rus), VITS | ~36M (145 MB) | **CC BY-NC 4.0** | 8–12 GB GPU | 0 | Not benchmarked here; 16 kHz; likely below Piper | ~RTF 0.1–0.2 CPU | **High:** VITS/ONNX | Research only |
| **C8. Vosk TTS ru multi** | [`alphacep/vosk-tts-ru-multi`](https://huggingface.co/alphacep/vosk-tts-ru-multi) | ~? (0.8 GB checkpoints) | **Apache-2.0** | 8–12 GB GPU | 0 | CER 2.3%, UTMOS 3.28; 5 voices (3F/2M) | RTF ~0.06 CPU on old Xeon | **Very high:** PyTorch, nonstandard training code | Interesting quality, poor integration |
| **C9. StyleTTS2 Russian** | no Russian checkpoint found on HF/GitHub on 2026-11-19; code [`yl4579/StyleTTS2`](https://github.com/yl4579/StyleTTS2) MIT | ~? | code MIT; weights would be new | 12–24 GB GPU to train | 0 | none | n/a | **None** | Not a candidate |

**Format summary (offline, no Python at inference):**

- **A / B:** safetensors read directly by the Rust `Archive` loader; no ONNX, PyTorch, GGML, or C/C++ runtime. Training is PyTorch (CPU/GPU); inference is pure Rust. A Russian GGUF could be produced later for `llama.cpp`'s `llama-tts`, but `voca` does not need it.
- **C1 Piper:** ONNX VITS, 63 MB; needs ONNX Runtime, `tract`, or a VITS port, plus an espeak `ru` G2P (GPL) or a new frontend.
- **C2 Utrobin:** ONNX VITS, 50.8 MB; needs ONNX Runtime/`tract` or a VITS port; char tokenizer + RUAccent; 16 kHz.
- **C3 / C4 Silero:** PyTorch `.pt` / TorchScript `.jit`; no official ONNX. Needs LibTorch (C++) or a custom port; 92–145 MB.
- **C5 XTTS-v2 / C6 F5-TTS ru:** PyTorch checkpoints, 1.3–1.9 GB; no realistic Rust path.
- **C7 MMS-TTS:** PyTorch/safetensors VITS; ONNX export possible (Xenova ONNX exists for the base model) but the weights are CC BY-NC.
- **C8 Vosk TTS:** PyTorch checkpoints (~0.8 GB); no ONNX/small-runtime path found.
- **C9 StyleTTS2:** no Russian checkpoint; would be a new PyTorch training project.

**C table notes.**

- "Params" for VITS models are estimated from fp32 file size (63.2 MB ≈ 15.8M; 50.8 MB ≈ 15.1M; 145 MB ≈ 36M; 91.7 MB ≈ 22.9M). They are not author-declared except Utrobin (15.1M).
- Piper's `medium` is a size tier, not a quality tier. The four Russian voices: [denis MODEL_CARD](https://huggingface.co/rhasspy/piper-voices/blob/main/ru/ru_RU/denis/medium/MODEL_CARD) (CC0 data, fine-tuned from `rdh`, CC0), [dmitri](https://huggingface.co/rhasspy/piper-voices/blob/main/ru/ru_RU/dmitri/medium/MODEL_CARD) (CC0 data, fine-tuned from `hifi_tts_92`, CC BY 4.0), [irina](https://huggingface.co/rhasspy/piper-voices/blob/main/ru/ru_RU/irina/medium/MODEL_CARD) (RHVoice, licence **Unknown**, fine-tuned from Lessac), [ruslan](https://huggingface.co/rhasspy/piper-voices/blob/main/ru/ru_RU/ruslan/medium/MODEL_CARD) (CC BY-NC-SA 4.0, fine-tuned from Lessac). Piper's own licensing page warns that the `MODEL_CARD` licence covers only the voice's own dataset, not the base chain, and that Lessac-derived voices are research-only ([LICENSING.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/LICENSING.md)). Only **denis** and **dmitri** are clean enough for commercial consideration; irina and ruslan are not.
- Piper's engine is **GPL-3.0-or-later** in the maintained [`OHF-Voice/piper1-gpl`](https://github.com/OHF-Voice/piper1-gpl); the old [`rhasspy/piper`](https://github.com/rhasspy/piper) is MIT but archived. Using Piper voices does not require using Piper's engine, but any port must reimplement VITS and the espeak G2P to avoid GPL.
- Silero's main models are CC BY-NC 4.0 ([README licence section](https://github.com/snakers4/silero-models#licence), [LICENSE](https://github.com/snakers4/silero-models/blob/master/LICENSE)); the CIS **base** models are explicitly MIT ([README V5 CIS Base Models](https://github.com/snakers4/silero-models#v5-cis-base-models), [LICENSE_CIS](https://github.com/snakers4/silero-models/blob/master/LICENSE_CIS)). The MIT base models have **no auto-stress or homographs**, so RUAccent is mandatory with them.
- XTTS-v2 is CPML 1.0.0: "This license allows only non-commercial use of a machine learning model and its outputs"; using it to train another model for commercial use is explicitly not a non-commercial purpose ([LICENSE.txt](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt)). The canonical `coqui.ai/cpml` URL was dead when checked; use the HF copy.
- F5-TTS code is MIT; its **pretrained models** are CC BY-NC because of the Emilia training data ([F5 README](https://github.com/SWivid/F5-TTS#license)). The Russian fine-tune is CC BY-NC 4.0.
- MMS-TTS Russian is CC BY-NC 4.0 ([card](https://huggingface.co/facebook/mms-tts-rus)).
- Vosk TTS Russian is Apache-2.0 ([card](https://huggingface.co/alphacep/vosk-tts-ru-multi)), but the checkpoints are PyTorch and the training code is a separate stack.

### 3.1 Fallback ranking

**If Qwen is unusable as a teacher, ranked by licence + quality + effort:**

1. **Silero `v5_cis_base` (MIT)** — same-family clarity, 24/48 kHz, no auto-stress; synthesize 16–20 h with RUAccent marks. Integration of the teacher is a one-off Python job; only the final 4.5M student ships.
2. **Piper `ru_RU-denis-medium` (CC0 voice chain)** — 22.05 kHz, clear but robotic; 16–20 h synthesis; output licence is the cleanest of the Piper voices.
3. **Real Russian audiobook speech** — SOVA RuAudiobooksDevices (298 h, CC BY 4.0) or Russian LibriSpeech (98 h, public domain / CC BY 4.0); best stress and prosody, but one real speaker's voice and no studio consistency. Use for evaluation even if not for training.
4. **Silero `v5_ru` (CC BY-NC)** — best clarity, but noncommercial; research-only teacher.
5. **XTTS-v2 / F5-TTS ru** — **not** usable as teachers for a commercial model; their licences restrict outputs and model training.

**If the 4.5M retrain misses the quality gates, ranked by final-model fit:**

1. **UtrobinTTS low multispeaker** — Apache-2.0, 15.1M, 50.8 MB ONNX, 16 kHz, two voices, RUAccent recommended. Cleanest licence/size compromise; add an ONNX/VITS runtime.
2. **Piper `ru_RU-denis-medium`** — CC0 chain, 15.8M, 63.2 MB ONNX, 22.05 kHz. Better sample rate than Utrobin, but the engine is GPL and the espeak frontend is a second GPL problem.
3. **Silero `v5_cis_base`** — MIT, 22.9M, 91.7 MB, 8/24/48 kHz; best if clarity matters more than size, but PyTorch/LibTorch is a hard sell for a Rust CLI.
4. **Silero `v5_ru`** — best Russian clarity, noncommercial only.
5. **XTTS-v2 / F5-TTS ru / MMS-TTS-rus** — no, for licence and size.

---

## 4. Path B — fine-tuning the English weights (sensible as a baseline, not as the plan)

### 4.1 Is it sensible for a 4.5M model with a different phoneme inventory?

Yes, with caveats. The architecture has 182–184 tensors; only the token embedding and the voice/stress embeddings are inventory-specific. The rest — encoder, duration predictor, prosody stack, acoustic stack, source-filter vocoder — is language-agnostic and already learned English phonation, timing and prosody. The Sanskrit `alpa` port did exactly this: it copied 182 of 184 tensors, rewrote two tables (nearest English sound for each Sanskrit symbol; syllable weight in place of stress), and fine-tuned on 3.5 h for 100 minutes on four CPU cores. It started "reading Sanskrit with an English accent" and reached ~3% phoneme error. Russian is much closer to English phonologically than Sanskrit (shared consonants, stress system, similar vowel inventory), so the prior is stronger.

But the English weights also carry an English timbre and English prosodic habits. For a product whose selling point is a *Russian voice*, the ceiling is an intelligible but audibly foreign speaker. That is why B is a 1-day baseline, not the default.

### 4.2 What upstream supports, and the one gotcha

`src/tiny_voca/train.py` supports:

```
--init-from PATH   # start a NEW run from another checkpoint
--new-data         # checkpoint comes from a different annotation run
```

`load_compatible()` copies tensors with the same shape and can **grow** an embedding (same rank, every dimension at least as large), keeping the old rows in the leading block. But `compatible()` first checks:

```python
if state["config"] != c.to_dict():
    raise ValueError("checkpoint was trained with a different model configuration")
```

`vocab_size` is part of `c.to_dict()`. So a **larger** Russian inventory will be rejected before `load_compatible` runs, even though the loader could grow it. Options:

- **Recommended:** use a 76-symbol Russian inventory (5 specials + 23 punctuation + 48 phones). The config is identical, `--init-from` works unmodified, and the Rust `SYMBOL_COUNT` stays 76.
- Or patch the config check to allow a vocab-size change and let `load_compatible` grow the embedding. This is a small change but another divergence from upstream.

### 4.3 Recipe and expectations

**Row transplant.** Do not simply overwrite the English symbol table and keep row indices: that gives `θ`'s weights to whatever Russian phone lands in slot 25. Build a `RU_TO_EN` table that maps every shared Russian phone (`p b t d k f v s z h m n l j`, `tʃ`, the open vowels, etc.) to its English row, and initialize Russian-only phones from the nearest English phone (as alpa's `NEAREST` table does) or from the mean of their articulatory neighbours. The transplant is a ~50-line script, not a trainer change. With a 76-symbol inventory the config check passes and `load_compatible` copies all same-shape tensors; the token embedding is the only table that needs this custom initialization.

| Item | Value |
|---|---|
| Data | 3–10 h Russian; the same annotation pipeline as A (phonemes + durations + stress) |
| Steps | 3,000–10,000; start at lr 3e-4, warmup 50, teacher-hold 150, GAN off (`--gan-start 100000`) as in alpa |
| Hardware | CPU 2–6 h for 3–10k steps; GPU <1 h |
| Quality | 4–10% WER/CER, audible English accent; useful to validate the frontend and the data pipeline before spending on A |
| Licence | Apache-2.0 base; fine |

**Use B as a frontend/integration smoke test:** it proves the Russian symbols, stress marks, aligner mapping, and Rust weight loader on a real Russian model in a day. If B already sounds unacceptable, A will still be better (native data, no English prior) but the frontend work is de-risked.

---

## 5. Russian TTS reference numbers (for the fallback table)

Caveat: these are **CER and UTMOS**, not MOS. UTMOS is trained mostly on English and is unreliable across languages; CER depends on the ASR and text normalization. Numbers from different sources are not a single ranking.

### 5.1 Vosk / Alphacephei evaluation (July 2024, ~1k utterances, audiobooks)

| Engine | Voice | CER ↓ | UTMOS ↑ | RTF CPU | RTF GPU | Source |
|---|---|---:|---:|---:|---:|---|
| Silero v3_1 | Aidar | 0.7% | 2.544 | 0.126 | 0.018 | [alphacephei](https://alphacephei.com/nsh/2024/07/12/russian-tts.html) |
| Silero v3_1 | Baya | 0.7% | 2.978 | 0.126 | 0.018 | same |
| Silero v4 | Aidar | 1.0% | 1.755 | 0.054 | 0.015 | same |
| Silero v4 | Baya | 0.9% | 2.144 | 0.054 | 0.015 | same |
| Piper | Denis | 3.7% | 3.056 | 0.045 | — | same |
| Piper | Dmitry | 3.6% | 2.864 | 0.045 | — | same |
| Piper | Irina | 1.4% | 3.672 | 0.045 | — | same |
| Piper | Ruslan | 3.0% | 2.975 | 0.045 | — | same |
| UtrobinTTS | Female | 2.1% | 2.851 | 0.132 | 0.027 | same |
| UtrobinTTS | Male | 2.1% | 3.186 | 0.132 | 0.027 | same |
| Vosk-TTS 0.6 | Multi | 2.3% | 3.283 | 0.061 | — | same |
| XTTS2 | Multi | 2.7% | 3.035 | — | 0.346 | same |
| Vosk-TTS GPT | Multi | 2.1% | 3.381 | — | 0.269 | same |
| EdgeTTS | Dmitry | 0.7% | 3.565 | 0.076 (cloud) | — | same |
| Yandex | Alexander | 0.6% | 3.413 | 0.028 (cloud) | — | same |

Note: `xRT` in the original table is real-time factor; lower is faster. Piper's 0.045 on an old Xeon is ~22× real time, consistent with the ru-speech-bench "RTF 0.20–0.65" for a slower VM.

### 5.2 ru-speech-bench (September 2026, Xeon E5 / RTX 3090 Ti)

| Engine | Speed | Russian quality | Licence |
|---|---|---|---|
| Piper denis/dmitri/irina/ruslan | RTF 0.20–0.65 CPU | even pace, no clicks; numbers/dates need normalization | voice data CC0/unknown/CC BY-NC-SA; runtime GPL-3.0 |
| XTTS-v2 | RTF ~0.5 GPU; first audio 0.47–0.53 s | WER 6.4% (no numerals), clone sim 0.58–0.69 | CPML noncommercial |
| F5-TTS Russian fine-tune | RTF ~0.2–0.3 GPU; first audio 1.1–3.1 s | WER 3.2%, clone sim 0.70–0.75 | CC BY-NC 4.0 |
| Qwen3-TTS 1.7B | RTF 2.8–3.3 on RTX 3090 Ti | 4/5 phrases verbatim; clone leaks the reference | Apache-2.0 |
| Silero VAD (context) | — | — | MIT |

Source: [ru-speech-bench README](https://github.com/meshoklv/ru-speech-bench), [Piper engine card](https://github.com/meshoklv/ru-speech-bench/blob/master/docs/engines/piper.md).

### 5.3 Subjective scores (single reviewer, not MOS)

A Russian review of seven open models gave 1–5 scores on a 10–15 s sentence: Silero v5_ru naturalness 4, expressiveness 4; F5-TTS 4.5/5; Qwen3-TTS 1.7B VoiceDesign 4.5/5; XTTS-v2 3/3. Latency: Silero 2.62 s CPU, Qwen 40.5 s CPU / 15.0 s GPU, XTTS 20.7 s CPU / 18.5 s GPU, F5 128.8 s CPU / 2.3 s GPU ([Habr](https://habr.com/ru/companies/raft/articles/991844), [mirror](https://ai-news.ru/2026/02/open_source_protiv_proprietarshiny_obzor_tts_modelej_dlya_russkogo_yazyka.html)). This is one listener's impression, not a benchmark.

### 5.4 What the numbers mean for the decision

- The teacher's 3.212% WER is **worse than Silero v3/v4 (0.7–1.0%) and Piper Irina (1.4%)**, better than Piper Denis (3.7%) and XTTS (2.7–6.4%), and roughly at Utrobin/Vosk (2.1–2.3%). A 4.5M student trained on it will not beat the teacher; expect 2–6%.
- Naturalness is the differentiator. Qwen/F5/XTTS are the lively ones; Piper is clear but flat; Silero is clear and still. A 4.5M source-filter model sits below all of them in naturalness, which is the price of 30 ms and 18 MB.
- If the product's bar is "as natural as Piper", path A may not clear it. If the bar is "a small, fast, offline Russian voice that is clearly intelligible and pleasant enough", the pt-BR port's 4/5 suggests it can.

---

## 6. Russian G2P + stress: options, licences, feasibility

| Component | What it gives | Licence | Quality evidence | Size / runtime | Use in this project |
|---|---|---|---|---|---|
| [RUAccent](https://github.com/Den4ikAI/ruaccent) / [HF](https://huggingface.co/ruaccent/accentuator) | Stressed text (`+` or U+0301), homograph resolution, ё-fication | **MIT** | 0.972 non-homograph / 0.96 homograph (paper); 0.947 word / 0.833 homograph independent | 21 MB gz dictionary, 0.8 MB accent ONNX, 10.5 MB omograph ONNX; Python RSS up to 1.5 GB | **Yes** — stress source for data and (ported) inference |
| [Silero Stress](https://github.com/snakers4/silero-stress) | Stressed text | **MIT** | 0.9629 independent; 4M words 100%, homographs F1 0.92 | ~50 MB, PyTorch | Backup/QA; PyTorch is a build burden |
| [nsu-ai/russian_g2p](https://github.com/nsu-ai/russian_g2p) | Stress + phonemes (custom code) | MIT code; dictionary provenance unclear | Old (2018), tests exist; no modern benchmark found | 54 MB `Accents.json`, 64 MB forms; Python | **Rules reference/port source**, replace dictionary with RUAccent |
| [maxmelichov/Russian-G2P](https://github.com/maxmelichov/Russian-G2P) | Narrow IPA + stress | **No licence file** (GitHub API: none) | Claims 604/604 words, 67/67 sentences on own gold set | 375 MB lexicon, 1.5 GB RAM, ONNX | **Reference only; do not depend** |
| [eSpeak NG](https://github.com/espeak-ng/espeak-ng) `ru` | IPA phonemes + `'` stress | **GPL-3.0-or-later** | 48 phonemes; no vowel reduction; 24/604 exact words / 45.8% sentence PER on Russian-G2P's gold set | small | **Annotation only**; never ship/port |
| Wiktionary-derived stress | Stress dictionary | CC BY-SA | Good coverage; incomplete homographs | varies | Avoid for shipped data (share-alike); test data only with care |

**Feasibility verdict:** path A is feasible **if** the project accepts a Russian stress frontend of ~30–50 MB on disk and a Rust G2P of ~500–800 lines. The acoustic model is not the risk. RUAccent's MIT licence and compact compressed artefacts make it the right source; its Python memory footprint is the thing to re-engineer.

**Recommended frontend stack:**

1. Normalize Russian text (numbers, dates, money, units, abbreviations).
2. RUAccent int8 models + compressed dictionary → U+0301 marks.
3. Rule-based Cyrillic→IPA G2P in Rust → 48-symbol inventory + stress levels.
4. Tokenizer → `tokens` + `stress` arrays, same layout as dizione.
5. For training only: eSpeak NG `ru` + `ESPEAK_TO_ALIGNER` map → forced-aligner labels → merged durations for the 48 symbols.
6. Tests: 200–500 word gold set, minimal pairs (`лук`/`люк`, `за́мок`/`замо́к`), 5,000-sentence Python-vs-Rust token parity, all 48 symbols covered by the aligner map.

---

## 7. Verification log and open questions

### 7.1 Verified on 2026-11-19

- tiny-voca architecture, parameter table, data size and training recipe: [GitHub README](https://github.com/n4ze3m/tiny-voca), [`docs/model.md`](https://github.com/n4ze3m/tiny-voca/blob/main/docs/model.md), [HF config](https://huggingface.co/n4ze3m/tiny-voca/blob/main/config.json), [HF dataset card](https://huggingface.co/datasets/n4ze3m/tiny-voca).
- Rust integration surface: `crates/tiny-voca/src/model.rs` (`SYMBOL_COUNT`, `FIRST_PHONE`, `FIRST_VOWEL`), `crates/dizione/src/symbols.rs`, `crates/tiny-voca/src/tts.rs`, `crates/tiny-voca/src/bin/voca.rs`.
- Upstream training flags `--init-from` / `--new-data` and the config-equality check: `src/tiny_voca/train.py` on GitHub.
- Qwen3-TTS licences, Russian support, sizes, API, 24 kHz output, hardware notes: HF model cards and [GitHub README](https://github.com/QwenLM/Qwen3-TTS); Russian WER/SIM table; `generation_config.json` (`max_new_tokens: 8192`), tokenizer config (`output_sample_rate: 24000`).
- Qwen MPS status: open issues/PRs [#124](https://github.com/QwenLM/Qwen3-TTS/issues/124), [#345](https://github.com/QwenLM/Qwen3-TTS/pull/345); not merged.
- Qwen Russian stress failures: [Discussion #185](https://github.com/QwenLM/Qwen3-TTS/discussions/185); stress adapter card with 65%/17% base and 100%/96% adapter numbers.
- Kokoro pt-BR distillation precedent: [HF card](https://huggingface.co/Haple/tiny-voca-ptbr-kokoro), [GitHub README](https://github.com/Haple/tiny-voca-ptbr-kokoro), `eval/corpus_summary.json` (16.37 h, 17,433 accepted), `eval/recipe.json` (100k steps), `eval/report.json` (1.44% WER), `eval/listening.json` (4/5).
- Polish and Arabic/Sanskrit precedents: [slyszeglosy](https://github.com/damaon/slyszeglosy), [Arabic port](https://github.com/BenkadourKhaled/tiny-voca/tree/arabic-halabi-f2), [alpa](https://huggingface.co/77ethers/alpa) and [TRAINING.md](https://huggingface.co/77ethers/alpa/blob/main/TRAINING.md).
- Piper voices and licences: [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices), the four `MODEL_CARD`s, [Piper LICENSING.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/LICENSING.md), [COMMERCIAL_USE.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/COMMERCIAL_USE.md), [piper1-gpl licence](https://github.com/OHF-Voice/piper1-gpl) (GPL-3.0), old [rhasspy/piper](https://github.com/rhasspy/piper) (MIT).
- Silero licences: [README licence section](https://github.com/snakers4/silero-models#licence), [LICENSE](https://github.com/snakers4/silero-models/blob/master/LICENSE) (CC BY-NC 4.0), [LICENSE_CIS](https://github.com/snakers4/silero-models/blob/master/LICENSE_CIS) (MIT for base CIS models), `models.yml` for `v5_cis_base` / `v5_cis_base_nostress` / `v5_ru`.
- XTTS-v2 CPML: [HF card](https://huggingface.co/coqui/XTTS-v2), [LICENSE.txt](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt) ("only non-commercial use ... and its outputs").
- F5-TTS: [README licence](https://github.com/SWivid/F5-TTS#license) (code MIT, pretrained CC BY-NC), [Russian fine-tune card](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN) (CC BY-NC 4.0).
- UtrobinTTS: [card](https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_low_multispeaker) (Apache-2.0, 15.1M, 16 kHz, ONNX 50.8 MB), [config](https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_low_multispeaker/blob/main/config.json).
- MMS-TTS Russian: [card](https://huggingface.co/facebook/mms-tts-rus) (CC BY-NC 4.0, VITS, 16 kHz).
- Vosk TTS Russian: [card](https://huggingface.co/alphacep/vosk-tts-ru-multi) (Apache-2.0).
- RUAccent: [repo](https://github.com/Den4ikAI/ruaccent), [LICENSE](https://github.com/Den4ikAI/ruaccent/blob/main/LICENSE) (MIT), [HF card](https://huggingface.co/ruaccent/accentuator) (MIT), file sizes from the HF tree.
- eSpeak NG `ru`: [`phsource/ph_russian`](https://github.com/espeak-ng/espeak-ng/blob/master/phsource/ph_russian) (48 phoneme definitions), [`dictsource/ru_rules`](https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/ru_rules), [COPYING](https://github.com/espeak-ng/espeak-ng/blob/master/COPYING) (GPL-3.0). `espeak-ng 1.52.0` was installed locally and run on Russian text to confirm the IPA output and homograph behaviour.
- Aligner labels: [wav2vec2-lv-60-espeak-cv-ft card](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) and `vocab.json` (392 labels; contains `ɨ`, `ɵ`, `ɐ`, `ʲ`, but not `lʲ`/`zʲ`).
- Russian corpora: [SOVA RuAudiobooksDevices 298 h, RuDevices 101 h, CC BY 4.0](https://github.com/sovaai/sova-dataset), [Russian LibriSpeech 98 h, public domain / CC BY 4.0](https://huggingface.co/datasets/istupakov/russian_librispeech), [Common Voice 17 now via Mozilla Data Collective](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0).
- Russian TTS numbers: [Vosk evaluation](https://alphacephei.com/nsh/2024/07/12/russian-tts.html), [ru-speech-bench](https://github.com/meshoklv/ru-speech-bench), [Habr subjective review](https://habr.com/ru/companies/raft/articles/991844).

### 7.2 Could not verify

- **No published Russian MOS for Qwen3-TTS.** The authors publish WER/SIM; Russian reviews are subjective. We did not listen to samples (no weights downloaded).
- **No official Apple Silicon support for Qwen3-TTS.** MPS PRs are open; CPU is possible in principle but no throughput number was found. The plan assumes a rented NVIDIA GPU.
- **No per-language quality number for Silero `v5_cis_base`.** The MIT claim and inventory are verified from the README; Russian UTMOS/CER for the CIS base model specifically was not found.
- **RUAccent dictionary provenance.** The repo and HF card grant MIT, but the upstream sources of `accents.json.gz` are not fully documented. Legal review is advisable before commercial redistribution.
- **Piper voice weights licence.** Piper makes no licence grant over voice models; the `MODEL_CARD` covers the voice's own dataset only, and whether a model is a derivative of its base/corpus is unsettled. The denis/dmitri chain is the cleanest but still needs a legal answer.
- **No Russian StyleTTS2 checkpoint found.** HF/GitHub searches on 2026-11-19 returned only English/generic StyleTTS2 repos. A Russian StyleTTS2 would require training.
- **No independent Russian benchmark for UtrobinTTS's ONNX file vs its PyTorch checkpoint.** The Vosk numbers are for the model family, not necessarily this ONNX export.
- **Kokoro pt-BR's teacher licence chain** was not audited beyond the card's Apache-2.0 statement; it is cited as a training-recipe precedent, not as a legal template.
- **`docs/ports.md` was not present in this worktree** when checked; R5 had not landed it. This document is self-contained and should be reconciled with R5's inventory/frontend decisions when it lands.

---

## 8. Recommendation and next actions

### 8.1 Recommendation

**Proceed with A: retrain the 4.54M tiny-voca architecture on 16–20 h of synthetic Russian from Qwen3-TTS 1.7B VoiceDesign + Base clone, with RUAccent stress marks and the Apache-2.0 stress adapter.** Use the 76-symbol inventory (5 specials + 23 punctuation + 48 Russian phonemes) so the Rust model code and path B compatibility are preserved. Train 100k steps with upstream's recipe; use a rented 24 GB GPU for teacher generation and training (5–9 h training, 30–70 GPU-hours generation), or a measured 1k-step M5 CPU pilot if a fully local run is required.

**Do not** ship Piper/Silero/XTTS/F5 as the primary model. They either break the 4.5M/24 kHz/pure-Rust identity or carry noncommercial licences. Keep UtrobinTTS (Apache-2.0) and Piper denis (CC0 chain) as the final-model fallbacks.

### 8.2 Gates before spending on the full 20 h

1. **Voice audition:** 2–3 VoiceDesign Russian voices, 10 sentences each, native listener. If none is native, clone a permissive synthetic reference instead.
2. **Stress audit:** 200 polysyllabic words + 30 homographs, RUAccent + adapter. Require ≤3% word stress error and ≥95% homograph accuracy.
3. **Screening audit:** 200 generated clips through Whisper Russian. Require ≤5% WER after normalization; inspect 20 by ear.
4. **Frontend spike:** 50 clips through eSpeak + aligner map + RUAccent. Require 100% label coverage, <3% alignment rejects, and correct durations for `lʲ`/`zʲ` minimal pairs.
5. **Pilot training:** 1,000 steps on 1–2 h. Measure steps/s on M5 CPU and on the rented GPU; decide local vs rented.
6. **Rust frontend size:** prove <50 MB on disk and <300 MB RSS with the RUAccent int8 models + compact dictionary.

### 8.3 Hand-offs

- **R2 (data):** deliver 8,000–10,000 normalized Russian sentences with provenance (CC0/CC BY 4.0), plus 50–100 held-out RuLS sentences for evaluation. Include numbers, dates, money, names, questions, homographs.
- **R4 (compute/env):** measure the 1k-step M5 CPU pilot; price a 24 GB GPU for 30–70 h of Qwen generation + 5–9 h of training; confirm Qwen bf16 memory and batch settings.
- **R5 (frontend/Rust):** freeze the 76-symbol inventory; implement the Russian G2P + RUAccent int8 integration; build the `ESPEAK_TO_ALIGNER` table; prove token parity on 5,000 sentences; measure disk/RSS.
- **Orchestrator:** run the six gates before authorizing the full generation/training spend.

### 8.4 One-line answer

**Retrain the native 4.5M architecture on ~16–20 h of screened Qwen3-TTS Russian with RUAccent stress; the acoustic model is the easy part, the Russian stress/G2P frontend is the real project.**
