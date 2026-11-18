# R3 — base models for a Russian `voca`

**Date:** 2026-11-19
**Branch:** `task/base-models`
**Companion document:** [`docs/base-models.md`](../docs/base-models.md) (full tables, URLs, licence notes)
**Scope:** the human chose the upstream synthetic-teacher recipe. This report states the revised recommendation, the evidence, the gates, and what could change the decision.

---

## Verdict

**Retrain the native 4.54M `tiny-voca` architecture from scratch on ~16–20 hours of synthetic Russian generated with Qwen3-TTS-12Hz-1.7B VoiceDesign + Base voice clone.** Keep the 24 kHz source-filter vocoder and a 76-token inventory. Use RUAccent for stress, a new permissive Russian G2P in Rust, and eSpeak NG `ru` only at annotation time.

This is the only path that keeps the product: 4.5M parameters, 18 MB weights, 24 kHz, ~7–30 ms model time, pure Rust, no ONNX/PyTorch runtime.

Path B (fine-tune the English weights) is a useful one-day baseline, not the plan. Path C (Piper/Silero/XTTS/F5) is fallback/reference only: it either breaks the size/speed identity or carries a noncommercial licence.

**The acoustic model is not the critical path. The Russian stress/G2P frontend is.** Qwen3-TTS Base does not reliably follow Russian stress marks; the Apache-2.0 stress adapter fixes that, but RUAccent's ~3% word error floor and the frontend's disk/RAM cost are the real risks.

---

## Why this path, in five pieces of evidence

1. **The recipe is proven on this exact architecture.** The Brazilian Portuguese Kokoro port trained a 4.54M student from scratch on **16.37 h** of synthetic speech from 8,945 sentences, **100k steps in 5 h 10 min on one rented GPU**, and reached **1.44% WER** on 16 novel sentences × 2 voices, with a single listener rating both voices **4/5** ([card](https://huggingface.co/Haple/tiny-voca-ptbr-kokoro)). The Polish port did ≈10k sentences, 104k steps, 8.9 h on one 24 GB GPU, and 0.37–0.48% CER ([repo](https://github.com/damaon/slyszeglosy)). Russian fits this envelope.
2. **Qwen3-TTS is the best-documented permissively licensed Russian teacher found.** Apache-2.0, Russian officially supported by all five 12 Hz models, 24 kHz output, **3.212% Russian WER / 0.792 speaker similarity** on the authors' multilingual table — better than ElevenLabs (3.878 / 0.676) and MiniMax (4.281 / 0.761) in that table ([README](https://github.com/QwenLM/Qwen3-TTS#evaluation)). Cost is small: upstream's whole English project (19.1 h data + training) was **$9.56** of GPU time.
3. **Two fixed synthetic voices are enough.** Upstream English used 2 voices, 17.6 h, 104k steps. We do not need 100 h or many speakers; 16–20 h with 2 designed/cloned voices matches the data scale that already works.
4. **The 76-symbol shape is preserved.** 5 specials + 23 punctuation + **48 Russian phonemes** (the eSpeak NG `ru` inventory in `phsource/ph_russian`) = 76. `Config::from_json` accepts it, the Rust network code is unchanged, and path B's `--init-from` works without patching upstream's vocab-size check.
5. **The fallbacks are clearly worse.** Piper is 15.8M/63 MB ONNX and its engine is GPL-3.0; only denis/dmitri have clean voice-data chains. Silero's best Russian model is CC BY-NC; the MIT `v5_cis_base` is PyTorch-only. XTTS-v2 is CPML noncommercial (outputs too); F5-TTS Russian is CC BY-NC; both are 1.3–1.9 GB PyTorch models. None belongs in an 18 MB Rust binary.

---

## The numbers that matter

| Item | Number | Source |
|---|---|---|
| tiny-voca params / audio | 4,537,291 / 24 kHz | upstream `docs/model.md` |
| Upstream English data / training | 17.6 h / 104,737 steps, 7.7 h on 4090 | upstream `docs/model.md` |
| pt-BR Kokoro port | 16.37 h accepted, 100k steps, 5 h 10 GPU, 1.44% WER, 4/5 | [card](https://huggingface.co/Haple/tiny-voca-ptbr-kokoro) |
| Polish port | ≈10k sentences, 104k steps, 8.9 h GPU, 0.37–0.48% CER; code Apache-2.0, **weights/data CC BY-NC 4.0** | [repo](https://github.com/damaon/slyszeglosy) |
| Qwen3-TTS 1.7B Russian | WER 3.212%, SIM 0.792 | [README](https://github.com/QwenLM/Qwen3-TTS#evaluation) |
| Qwen3-TTS 0.6B Russian | WER 4.458%, SIM 0.781 | same |
| Qwen3-TTS Base stress adherence | 65% on-dictionary / 17% against | [adapter card](https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf) |
| Stress adapter adherence | 100% / 96%; 0.1% departure from RUAccent; CER 2.9% | same |
| RUAccent accuracy | 0.972 non-homograph / 0.96 homograph; 0.947 / 0.833 independent | [paper](https://aclanthology.org/2025.coling-main.444.pdf), [benchmark](https://github.com/TigreGotico/stressonnx/blob/dev/docs/models.md) |
| RUAccent on disk / Python RSS | 21 MB gz dictionary + 0.8 MB accent ONNX + 10.5 MB omograph ONNX; up to 1.5 GB RSS | [HF tree](https://huggingface.co/ruaccent/accentuator) |
| eSpeak NG `ru` inventory | 48 phoneme definitions | [`phsource/ph_russian`](https://github.com/espeak-ng/espeak-ng/blob/master/phsource/ph_russian) |
| Aligner labels | 392; contains `ɨ`, `ɵ`, `ɐ`, `ʲ`; not `lʲ`/`zʲ` | [`vocab.json`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/blob/main/vocab.json) |
| Piper denis | CER 3.7%, UTMOS 3.06; 15.8M/63.2 MB; CC0 chain | [Vosk eval](https://alphacephei.com/nsh/2024/07/12/russian-tts.html), [MODEL_CARD](https://huggingface.co/rhasspy/piper-voices/blob/main/ru/ru_RU/denis/medium/MODEL_CARD) |
| UtrobinTTS low | CER 2.1%, UTMOS 2.85–3.19; 15.1M/50.8 MB ONNX; Apache-2.0 | [card](https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_low_multispeaker), [Vosk eval](https://alphacephei.com/nsh/2024/07/12/russian-tts.html) |
| Silero v5_ru | CER 0.7–1.0% (v3/v4), UTMOS 1.75–2.98; CC BY-NC; ~36M/145 MB | [Vosk eval](https://alphacephei.com/nsh/2024/07/12/russian-tts.html) |
| XTTS-v2 | CER 2.7%, UTMOS 3.04, WER 6.4%; CPML noncommercial; 467M/1.87 GB | [Vosk eval](https://alphacephei.com/nsh/2024/07/12/russian-tts.html), [LICENSE](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt) |
| F5-TTS ru | WER 3.2%; CC BY-NC; 336M/1.35 GB | [card](https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN) |

---

## Qwen3-TTS facts the plan depends on (verified 2026-11-19)

- **Licence:** Apache-2.0 for `Qwen3-TTS-12Hz-1.7B-Base`, `...-VoiceDesign`, and `Qwen3-TTS-Tokenizer-12Hz` (model-card frontmatter; tokenizer card).
- **Russian:** officially listed for all five 12 Hz models, including Base and VoiceDesign. The 1.7B is better than 0.6B on Russian (3.212 vs 4.458 WER).
- **Sizes:** 1.7B = 3.83–3.86 GB `model.safetensors` + 0.68 GB speech tokenizer. Output **24 kHz**, matching tiny-voca.
- **Voice design:** `generate_voice_design(text, language="Russian", instruct=...)`. Instructions in English or Chinese are accepted; batch via lists.
- **Voice clone:** `create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)` then `generate_voice_clone(text, language="Russian", voice_clone_prompt=...)`. The official claim is **3-second** cloning; 10–20 s is the practical recommendation. `x_vector_only_mode=True` drops `ref_text` but lowers quality. Reuse the prompt for all sentences; batch generation is supported.
- **Limits:** `generation_config.json` defaults to `max_new_tokens: 8192`; the evaluation uses 2048. At 12.5 Hz that is ≈164 s (2048) to ≈655 s (8192). Generate per sentence/short paragraph, not one giant call.
- **Hardware:** examples assume CUDA bf16, FlashAttention 2 optional. A public Russian benchmark used an RTX 3090 Ti 24 GB. **No official Apple Silicon support**: MPS PRs [#124](https://github.com/QwenLM/Qwen3-TTS/issues/124) / [#345](https://github.com/QwenLM/Qwen3-TTS/pull/345) are open, not merged. CPU is possible in PyTorch but not practical for 20 h. Rent a 24 GB NVIDIA GPU.
- **Generation cost:** 30–70 GPU-hours for 20 h raw audio, ≈$10–40 at 4090/3090 rates (public RTF 2.8–3.3 on a 3090 Ti; upstream's whole project was $9.56).

---

## Teacher-quality caveats for Russian

1. **Stress is the failure mode.** Base Qwen does not reliably follow `+`, `'`, or combining-acute marks; [Discussion #185](https://github.com/QwenLM/Qwen3-TTS/discussions/185) reports widespread wrong stress and mangled homographs (`за́мок`/`замо́к`). The adapter card quantifies it: 65% adherence on-dictionary, 17% against, for base; 100% / 96% with the adapter. On ordinary text the adapter departs from RUAccent in 0.1% of words, but RUAccent itself is ~97% accurate — so **~3% of words can still be stressed wrongly**.
2. **No Russian MOS is published.** Only WER/SIM and subjective reviews. A positive review puts Qwen ahead of ElevenLabs/MiniMax on WER; a 38-phrase Russian test reports failures on numbers with spaces, abbreviations next to Latin text, and unit abbreviations. Normalize text before generation; do not trust Qwen's normalizer.
3. **VoiceDesign may not sound native.** The pt-BR Kokoro port author found Qwen-designed voices from English/Portuguese descriptions did not sound Brazilian, and most Qwen clones failed too. Qwen does officially support Russian, so this is a warning, not a verdict. **Audition 2–3 VoiceDesign Russian voices with a native listener before generating 20 h.** If they are accented, clone a permissive synthetic Russian reference (Silero `v5_cis_base` MIT or Piper denis CC0) instead.
4. **Adapter data caveats.** The Apache-2.0 stress adapter flags that some training text (`biggest-ru-book`) has no licence and its stress detector was trained on Silero (noncommercial). The weights are Apache-2.0, but for a clean commercial release retrain the same recipe on Russian LibriSpeech (CC BY 4.0) + RUAccent marks; the adapter card publishes the scripts.

---

## The frontend is the real project

- **Inventory:** 76 symbols = 5 specials + 23 punctuation + **48 Russian phonemes** (eSpeak NG `ru`'s inventory). This preserves `vocab_size`, the Rust model code, and path B compatibility.
- **eSpeak NG `ru`:** 48 phonemes, GPL-3.0. Use it **only as an external annotation program**, as upstream did. Do not link, port, or ship it. It does not reduce vowels well and reads `замок` as "lock" without context, so it is not the final G2P.
- **Aligner:** `facebook/wav2vec2-lv-60-espeak-cv-ft` (Apache-2.0). Map all 48 eSpeak phones to 1–2 aligner labels; `lʲ`, `zʲ`, `ɕː` need splitting/shortening. Calibrate on 50–100 clips; test `лук`/`люк` minimal pairs before scaling.
- **Stress:** RUAccent (MIT) is the best permissive option: 0.972/0.96; 21 MB gz dictionary + 0.8 MB accent ONNX + 10.5 MB omograph ONNX. The Python loader's up-to-1.5 GB RSS is the thing to re-engineer; a Rust loader/trie + int8 models should target <300 MB RSS and <50 MB disk.
- **Rust G2P:** a rule-based Cyrillic→IPA transducer + RUAccent stress, ~500–800 lines + tests. `nsu-ai/russian_g2p` (MIT code, unclear dictionary provenance) is a reference/port source; replace its dictionary with RUAccent's. Do not copy GPL eSpeak rules. `maxmelichov/Russian-G2P` has **no licence file** — reference only.
- **Size regression:** the stress frontend, not the 18 MB TTS model, is the product-size risk. If it cannot fit, train a compact 1–5M stress predictor on RuLS + RUAccent labels; do not silently drop stress accuracy.

---

## Training on the M5 Pro

- Upstream supports `--device cpu`, `--max-seconds`, `--resume`, `--init-from`, `--new-data`.
- The waveform head uses complex STFT/FFT (`torch.stft(return_complex=True)`, `torch.fft.rfft`, `torch.polar`). **Plan CPU, not MPS**; there is no MPS path and complex ops are not reliable there.
- Full local training estimate: **30–80 h** for 100k steps on 16–20 h data (extrapolated from the 4090's 5–9 h; must be measured with a 1k-step pilot).
- Rented 24 GB GPU: **5–9 h** for 100k steps, ≈$10–30; plus 30–70 GPU-hours for Qwen generation.
- VRAM: <10 GB expected; 24 GB is ample. RAM: features + RUAccent 2–4 GB (Python), less in Rust.

---

## Acceptance gates (run before the full 20 h spend)

1. **Voice audition:** 2–3 VoiceDesign Russian voices, 10 sentences, native listener. If none is native, clone a permissive synthetic reference.
2. **Stress audit:** 200 polysyllabic words + 30 homographs, RUAccent + adapter. Require ≤3% word stress error, ≥95% homograph accuracy.
3. **Teacher screening:** 200 clips through Whisper Russian; ≤5% WER after normalization; 20 by ear.
4. **Frontend spike:** 50 clips through eSpeak + aligner map; 100% label coverage, <3% alignment rejects, correct `lʲ`/`zʲ` durations.
5. **Pilot training:** 1,000 steps on 1–2 h; measure steps/s on M5 CPU and the rented GPU.
6. **Rust size:** <50 MB disk / <300 MB RSS with RUAccent int8 + compact dictionary.

Expected quality if the gates pass: **2–6% WER/CER**, naturalness below Piper/Silero but above a plain formant synth; pt-BR analogue 1.44% WER and 4/5 listener. Not a MOS promise.

---

## What would make me argue against the default

- **VoiceDesign accent gate fails** → do not generate 20 h; clone a permissive synthetic reference (Silero `v5_cis_base` MIT / Piper denis CC0) or use real Russian audiobook speech.
- **Stress gate >3–5%** → teacher unusable for a stress-sensitive product; use real Russian audio (RuLS/SOVA) or Piper denis/Silero + RUAccent as teacher.
- **Screened teacher WER >5%** → student inherits it; switch teacher/data.
- **`lʲ`/`zʲ` alignment unreliable** → character-based student or different aligner; do not ship a model that merges `лук`/`люк`.
- **Frontend cannot fit <50 MB / <300 MB RSS** → compact stress predictor before shipping; the product's small-binary promise is otherwise broken.
- **Teacher generation >100 GPU-hours or training >2 weeks local** → cut to 10 h / 50k steps or rent.

---

## Fallbacks (demoted, but concrete)

**If Qwen is unusable as a teacher, ranked:**

1. **Silero `v5_cis_base`** — MIT, ~22.9M/91.7 MB, 8/24/48 kHz, no auto-stress; synthesize with RUAccent marks. Best permissive teacher fallback.
2. **Piper `ru_RU-denis-medium`** — CC0 voice-data chain, 15.8M/63.2 MB, 22.05 kHz; clear but robotic.
3. **Real Russian audiobooks** — SOVA RuAudiobooksDevices 298 h, RuDevices 101 h (CC BY 4.0); Russian LibriSpeech 98 h (public domain / CC BY 4.0). Best stress/prosody, one real speaker; use at least for evaluation.
4. **Silero `v5_ru`** — CC BY-NC; best clarity; research only.

**If the 4.5M retrain misses the gates, ranked final models:**

1. **UtrobinTTS low multispeaker** — Apache-2.0, 15.1M/50.8 MB ONNX, 16 kHz, 2 voices, CER 2.1%, UTMOS 2.85–3.19; needs ONNX/VITS runtime + RUAccent.
2. **Piper `ru_RU-denis-medium`** — CC0 chain, 15.8M/63.2 MB ONNX, 22.05 kHz, CER 3.7%, UTMOS 3.06; engine GPL / espeak GPL.
3. **Silero `v5_cis_base`** — MIT, 22.9M/91.7 MB, 8/24/48 kHz; PyTorch/LibTorch is the cost.
4. **Silero `v5_ru`** — best Russian clarity, noncommercial only.

Excluded: XTTS-v2 (CPML noncommercial, outputs too), F5-TTS Russian (CC BY-NC), MMS-TTS Russian (CC BY-NC), Piper irina (unknown data licence, Lessac chain), Piper ruslan (CC BY-NC-SA, Lessac chain), StyleTTS2 (no Russian checkpoint found).

---

## Could not verify (stated plainly)

- No published Russian **MOS** for Qwen3-TTS; WER/SIM and reviews only. We did not listen to samples (no weights downloaded).
- No official **Apple Silicon/MPS** support for Qwen3-TTS; MPS PRs are open. No CPU throughput number found.
- No per-language quality number for Silero **`v5_cis_base`**; the MIT licence and inventory are verified, Russian UTMOS/CER are not.
- **RUAccent dictionary provenance** beyond the repo's MIT grant; legal review advisable.
- **Piper voice-weight licensing** is unsettled; the MODEL_CARD covers only the voice's own dataset, and Lessac-derived voices are research-only. Only denis/dmitri are clean enough to consider.
- No Russian **StyleTTS2** checkpoint found.
- `docs/ports.md` was not present in this worktree; R5 had not landed it. Reconcile the 76-symbol inventory and frontend design with R5 when it lands.

---

## Hand-offs

- **R2 (data):** 8,000–10,000 normalized Russian sentences with provenance (CC0/CC BY 4.0), 50–100 held-out RuLS sentences, numbers/dates/money/names/questions/homographs.
- **R4 (compute):** 1k-step M5 CPU pilot; price 30–70 GPU-hours Qwen generation + 5–9 h training on a 24 GB GPU.
- **R5 (frontend):** freeze the 76-symbol inventory; Rust G2P + RUAccent int8; `ESPEAK_TO_ALIGNER` table; 5,000-sentence token parity; disk/RSS measurement.
- **Orchestrator:** run the six gates before authorizing the full generation/training spend.

---

## One-line answer

**Retrain the native 4.5M architecture on ~16–20 h of screened Qwen3-TTS Russian with RUAccent stress; the acoustic model is the easy part, the Russian stress/G2P frontend is the real project.**
