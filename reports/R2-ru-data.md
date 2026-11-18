# R2 — Russian data and text survey (second pass)

**Owner:** R2 · **Branch:** `task/ru-data` · **Date:** 2026-11-19
**Deliverable:** [`docs/datasets-ru.md`](../docs/datasets-ru.md)

**Scope correction:** the project now reproduces upstream's synthetic-data
recipe (Qwen3-TTS VoiceDesign + Base clone). The recorded-corpus survey is kept
as **fallback only** and is explicitly labelled so in the doc. The primary path
is text + stress + ASR judge + teacher quality, covered below.

## TL;DR

- **Text is solved without a corpus download.** Russian Common Voice's
  `server/data/ru/sentence-collector.txt` has **46,091 sentences**, of which
  **39,138 are 5–30 words**; the file is 5.25 MB and CV's own rules require
  **CC0/public-domain** text. That is the direct Russian equivalent of
  upstream's `sentence-collector.txt`.
- **Recommended script mix (8,000 sentences):**
  - batch 1 (4,500): 3,000 Common Voice CC0 + 900 Wikisource PD prose +
    600 LLM-written; split 3,600/450/450 by sentence.
  - batch 2 (3,500): 2,600 project templates + 900 LLM-written; train-only.
  - Release as **CC0 1.0**. Keep Wikipedia (CC BY-SA 4.0) and
    OPUS/OpenSubtitles (no clear licence) out of the released list.
- **Stress:** `ruaccent` is **both** a dictionary and a model: **3,194,879
  stressed forms** (21 MB gz), **19,741 omographs**, a ~0.8 MB OOV model, and a
  **10.5 MB tiny omograph model** + 0.88 MB tokenizer. Shippable in Rust at
  ~33 MB compressed; MIT, no share-alike. `russian_g2p` (MIT) emits stress
  (`+`) and phonemes (`0` = stress on vowels, palatalisation on consonants).
  eSpeak NG `ru` is **48 phoneme definitions** but **GPL-3.0**; do not port it
  into the Apache-2.0 Rust binary.
- **ASR judge:** use **Whisper large-v3-turbo** (MIT, 809M) for the 15% screen
  and final eval. On FLEURS-ru n=775 it scores **4.44%**, effectively tied with
  **GigaAM-v3** (**4.50%**, MIT, 240M); GigaAM wins audiobooks/spontaneous
  (6.57/6.95 vs 9.89/7.73). Use GigaAM as the mandatory fallback/cross-check.
  large-v3 is the accuracy reference but costs ~2× the compute for at most a
  small, harness-confounded read-speech gain.
- **Qwen3-TTS Russian is usable but stress-weak.** Official Russian WER is
  **3.212%** (12Hz-1.7B-Base) on Qwen's multilingual test, but two Apache-2.0
  Russian fine-tunes document that the base model follows a stress mark on the
  dictionary vowel only **65%** of the time and marks against the dictionary
  **17%**; on ordinary text its stress differs from RUAccent in **~14%** of
  words. No official number/stress eval and no Russian demo were found; numbers
  are untested. A stress-following fine-tune (96–100% mark-following) is the
  practical teacher.

## Primary-path evidence

### 1. Text sources

| source | licence | verdict |
|---|---|---|
| Common Voice ru `sentence-collector.txt` | **CC0** (CV rules quote: “must be in the public domain, with a CC0 license”) | **Use, 3,000 batch 1** |
| Wikisource PD (Pushkin, Gogol, Dostoevsky, Turgenev, Chekhov, Tolstoy) | **PD**; site contributions CC BY-SA 4.0, but PD text itself is not | **Use, 900 batch 1**; only literary text, no annotations/translations |
| Russian Wikipedia | **CC BY-SA 4.0** | **Do not use**; verbatim extraction forces the script list to BY-SA, breaking CC0 release |
| OPUS/OpenSubtitles ru | **Unclear**; attribution request, no explicit licence field | **Do not use** |
| News corpora | No permissive source verified; Taiga 404/unlicensed | **Do not use** |
| LLM-written + templates | **Project-own** | **Use, 600 + 900 + 2,600** |

Wikisource PD selection rule (from the Russian Wikisource copyright page):
authors who died ≤1940; 1941–1955 has war-era exceptions; pre-1917 publications
are PD. To stay US-safe, use works published **≤1930** and record author, work,
year, URL.

### 2. Templates

The doc contains **25 concrete Russian patterns** (greeting, phone, meeting,
money, shopping, travel, distance, reading, errand, date, quantity, question,
list, score, surprise, phone menu, discount, cash, year, train, ticket, address,
duration, invoice, abbreviations). They deliberately exercise:

- cardinal government (`рубль/рубля/рублей`, `копейка/копейки/копеек`),
- dates (`пя́того ма́я`, `в … году́` vs `… го́да`),
- `полтора́`/`полторы́`,
- names in dative/genitive/instrumental,
- abbreviations (`т. е.`, `см.`, `с.`, `ИНН`) and digit-by-digit phone numbers.

Common Voice sentences cannot supply these: its rules explicitly exclude
numbers and special characters. That is why batch 2 is templates + LLM.

### 3. Stress

- **`ruaccent` format:** gzip JSON, key = lowercase word, value = word with `+`
  before the stressed vowel. Inflected forms are covered (verified:
  `стол/стола/столу/столом/столе/столы`); omographs map to variants
  (`замок` → `з+амок`, `зам+ок`; `году` → `г+оду`, `год+у`).
- **Offline query:** load the gz JSON; look up the word; if it is an omograph,
  score variants with the ONNX model + tokenizer in sentence context. Python:
  `RUAccent().load(use_dictionary=True, omograph_model_size="tiny")`, then
  `process_all(text)`.
- **Ship in Rust:** ~33 MB compressed for dictionary + omographs + tiny model +
  tokenizer + accent model; use `ort` + `tokenizers`, or convert the dictionary
  to an FST/trie. MIT: include the notice; no SA.
- **Critical test case:** `в 2026 году` must be `год+у`, not the dictionary's
  default `г+оду`. The omograph model is required for dates/years.
- **`russian_g2p`:** MIT; `Accentor` emits `+`; `Grapheme2Phoneme` has 57
  non-syllabic symbols (29 base × stressed/unstressed or hard/soft) plus `sil`;
  `0` is overloaded (vowel stress vs consonant palatalisation). Use it to build
  the phoneme lexicon, not as a second stress oracle.
- **eSpeak NG `ru`:** 48 phoneme definitions in `phsource/ph_russian`, separate
  `'` stress; **GPL-3.0**. Porting its tables/rules into Rust makes a GPL
  derivative. Use only as a cross-check/offline oracle; keep the runtime MIT.

### 4. ASR judge

| model | id | params | licence | FLEURS-ru | audiobooks | spontaneous | memory |
|---|---|---|---:|---|---:|---:|---:|
| Whisper large-v3 | `openai/whisper-large-v3` | 1.55B | Apache-2.0 | 3.1% (GigaAM eval) | 9.4% RuLS | — | ~6 GB VRAM |
| **Whisper large-v3-turbo** | `openai/whisper-large-v3-turbo` | 809M | **MIT** | **4.44%** (n=775) | 9.89% | 7.73% | ~2–3 GB |
| **GigaAM-v3** | `ai-sage/GigaAM-v3` | 220–240M | **MIT** | **4.50%** (n=775) | **6.57%** | **6.95%** | <1 GB |

Use turbo; GigaAM is the fallback/cross-check. No M5-specific RTF is published;
`faster-whisper` large-v2 int8 does 13 min in 59 s on an 8-thread desktop CPU
(RTF ≈ 0.076), turbo is faster. Benchmark on the M5 in F1/F5.

### 5. Qwen3-TTS Russian

- Official: Apache-2.0; Russian supported; paper Table 6 gives Russian WER
  **3.212%** (12Hz-1.7B), 4.535% (25Hz-1.7B), 4.458% (12Hz-0.6B), 5.957%
  (25Hz-0.6B); MiniMax 4.281%, ElevenLabs 3.878%. Model card shows only
  Chinese/English Seed-TTS WER.
- Third-party: base follows a stress mark on the dictionary vowel **65%**, and
  against the dictionary **17%**; ~14% of ordinary-text stress differs from
  RUAccent; CER 3.6%; English terms in Cyrillic are the worst. Fine-tunes
  (`sknyazev/...-ru-stress-gguf`, `siriusfreak/...-ru-stress-cf`) reach 96–100%.
- No official Russian stress/number eval, no Russian demo, no GitHub issue about
  Russian quality found. **Numbers are untested** — use the templates and
  normalise digits to words before TTS.
- **Teacher decision:** either use a stress-following fine-tune, or keep Base
  and accept ~15% stress errors (which is exactly the screen threshold, so WER
  will not catch them). Test both on 200 clips before 17k generation.

## Fallback recorded corpora (unchanged, not the primary path)

The earlier survey remains in the doc: RUSLAN (31.5 h, 1 speaker, CC BY-NC-SA —
best quality, NC), RuLS (98.24 h, 13 speakers, PD-US), M-AILABS (46.78 h,
3 speakers, contradictory licence), Golos (1,240 h, custom BY-SA-like), SOVA
(298 h audiobooks, CC BY 4.0), Common Voice (220 h, CC0), FLEURS (11.63 h,
CC BY 4.0). RUSLAN/M-AILABS/Golos licence flags are unchanged. **No recorded
corpus is needed for the synthetic path.**

## Verification / acceptance

- [x] Live fetches on 2026-11-19 for every new claim; URLs in doc §11.
- [x] CV Russian sentence count (46,091; 39,138 at 5–30 words) and CC0 rule
      verified from the live file and CV docs.
- [x] Wikisource PD rule, site licence, and six author death dates verified.
- [x] ruaccent dictionary count, omographs, format, model sizes, MIT licence
      verified; inflected-form coverage spot-checked.
- [x] russian_g2p `0` semantics and 57/115 symbol counts verified from source.
- [x] eSpeak `ph_russian` 48 definitions and GPL-3.0 verified.
- [x] ASR WER numbers from GigaAM evaluation and the independent FLEURS-ru
      benchmark; licences/params/ids verified.
- [x] Qwen official Russian WER from the paper; third-party stress reports and
      licences verified.
- [x] No full datasets/models downloaded; no training run started.
- [x] Rebased on `origin/main` before reporting; commit prefix `R2:`;
      `git status` clean.

## Questions for the human

1. **Script mix:** approve CC0 release using CV + Wikisource PD + LLM/templates,
   and exclude Wikipedia/OPUS? (Recommended.)
2. **Qwen teacher:** use a stress-following Russian fine-tune (Apache-2.0,
   trained on RuLS) instead of base Qwen3-TTS, or keep Base and accept ~15%
   stress errors? (Recommended: fine-tune or mark+verify.)
3. **ASR:** approve turbo as primary and GigaAM as fallback/cross-check?
4. **Fallback corpora:** keep on the shelf; no download unless the synthetic
   path fails. RUSLAN/M-AILABS licences remain unresolved but are not needed.

## Next actions

1. F2: build the 8,000-sentence JSONL with per-row provenance/licence; freeze
   val/test by hash.
2. F3: normalise numbers/dates/money to Russian words, apply ruaccent, and add
   the 25-template regression suite.
3. F4/F5: evaluate base vs stress-fine-tuned Qwen on 200 clips; wire turbo +
   GigaAM cross-check into the WER harness.
