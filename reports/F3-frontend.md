# F3 — Russian text frontend (normalise → stress → phonemes)

Written 2026-11-19 by F3. Branch `task/f3-frontend`, rebased on `origin/main`
(which contains F6's `docs/aligner-mapping.md` and `scripts/align_spike/`).

## 1. What was built

Everything lives in `training/voca_ru/text/`:

| file | job |
|---|---|
| `symbols_ru.py` | the 76-symbol inventory (data), eSpeak→token map, F6 aligner map |
| `numbers.py` | cardinals/ordinals with case + gender, scale agreement, decimals |
| `normalize.py` | deterministic normaliser: numbers, dates, times, money, units, abbreviations, Latin, punctuation |
| `stress.py` | RUAccent dictionary + nn_accent + omograph list (MIT subset), rule fallback, OOV flag |
| `g2p.py` | eSpeak NG 1.52 `ru` through `espeakng-loader`'s dylib, stress transfer, token stream |
| `pipeline.py` | `analyze(text) -> {plain_text, stressed_text, norm_text, phonemes, stress, tokens, oov}` |
| `__main__.py` | `python -m voca_ru.text --in file.jsonl --out file.jsonl` |
| `measure.py` | corpus token/OOV metrics (numbers below) |
| `requirements.txt` | pinned runtime deps |

`tests/text/` holds 59 test methods with >300 table/property sub-cases; all
pass (`python -m unittest discover -s tests/text`, 0.30 s).

Reproduce:

```sh
cd worktrees/f3-frontend
. ../../scripts/cache-env.sh
uv venv ../../.cache/venvs/f3 --python 3.12
uv pip install --python ../../.cache/venvs/f3/bin/python -r training/voca_ru/text/requirements.txt
PYTHONPATH=training ../../.cache/venvs/f3/bin/python -m unittest discover -s tests/text
PYTHONPATH=training ../../.cache/venvs/f3/bin/python -m voca_ru.text.measure \
    data/scripts-ru/batch1.jsonl data/scripts-ru/batch2.jsonl
```

The RUAccent data is in `.cache/models/ruaccent/dictionary/` (3.19 M forms,
21 MB gz) and the ONNX model in the venv (`ruaccent/nn/nn_accent`). Set
`VOCA_RUACCENT_WORKDIR` if they live elsewhere. Loading is 2.2 s and peaks at
**1.53 GB RSS** (the full Python dict); per-sentence work is ~0.3 ms.

## 2. Token inventory (contract §4)

`SYMBOLS` is exactly 76 and is the single source of truth for F7/Rust:

```
0–4     5 specials   <pad> <bos> <eos> <sp> <unk>
5–27    23 punctuation
28–63   36 consonants
64–75   12 vowels
```

- Ids 0–16 are byte-identical to upstream (`<pad>..<unk>` + `, . ? ! ; : " ' ( ) — …`).
  `FIRST_PHONE = 28`, `FIRST_VOWEL = 64`.
- Punctuation (23): the upstream 12, then `« » - – № % / [ ] { }`.
- Consonants (36): hard/soft pairs as separate symbols
  (`b bʲ v vʲ ɡ ɡʲ d dʲ ʒ z ʑ j k kʲ ɭ ɭʲ m mʲ n nʲ p pʲ r rʲ s sʲ t tʲ ts tʃʲ f fʲ x ʃ ɕ dʒ`).
- Vowels (12): `ʌ a ɑ ɪ i ɛ e ə o u y ɵ`. Reduction and `ё` (`ɵ`) are kept
  as distinct tokens; this is a real cue and fits the 48-phone budget.
- All **48 phones occur** in the 8k corpus; no phone is padding.
- Palatalisation is part of the consonant token, not a separate `ʲ` token:
  eSpeak and F6's aligner both emit `tʲ` as one label, and 36+12 fits.

### Consistency with F6 (`docs/aligner-mapping.md`)

Every training phone has a row in `ALIGNER_LABEL` (the F6 label table).
`ALIGNER_LABEL` follows F6 exactly for the canonical merges
(`a→ʌ`, `ɪ→i`, `ɛ→e`, `ə→ʌ`, `ɵ→o`, `u"→u`, `ɪ^→i`, `tʃ→tʃʲ`, `dʒʲ→dʒ`,
`zʲ→ʑ`, `ɕː→ɕ`, and the foreign/allophone tails at the nearest Russian label).
Keeping `ʌ/a/ɑ/ɪ/i/ɛ/e/ə/ɵ` distinct in the *token* inventory while both map
to the same aligner label is explicitly allowed by F6 §5.4.

Two additions / one divergence, all recorded for F6:

1. `ŋ → n` — observed in the 8k corpus (F6's FLEURS list has no `ŋ`). Added to
   `PHONE_MAP`; no aligner row was needed because it maps to `n`.
2. `ɐ → ʌ` — defensive; not emitted by eSpeak 1.52 in this corpus.
3. F6's artefact table maps eSpeak `8` to `o`. F3 maps the token to **`ɵ`**
   (it is the internal name of the ё vowel) and maps `ɵ→o` at the aligner
   layer, so alignment still sees `o`. `8` never appeared with 1.52 here.
   This is a token-layer difference, not a label-layer fork.

## 3. Normalisation policy

- Lowercase; `ё` is preserved and never folded to `е`. Existing `U+0301` is
  stripped first, so a second pass is identical (`normalize(normalize(x)) == normalize(x)`).
- Numbers → words with preposition-driven case (`до пяти`, `к пяти`,
  `в двадцати километрах`, `на пять пунктов`), 2–4/5+/11–14 agreement,
  gender for 1/2, thousands/millions, compound tens, `полтора/полторы`,
  decimals (`2,5 → два целых пять десятых`), ordinals (`5-й`, `1-го`, `20-е`).
- Dates: `19 ноября 2026` → `девятнадцатого ноября две тысячи двадцать шестого года`;
  `19.11.2026` same. Years: `в 2026 году` → `… шестом году`
  (hard override `году → год+у`; F6/R2 flagged this).
- Times: `15:30 → пятнадцать тридцать`, `09:05 → девять ноль пять`.
- Money: `1 500 ₽ → тысяча пятьсот рублей`, `12 рублей 50 копеек` unchanged
  except the numerals. Units (`км`, `кг`, `л`, …) get the correct noun form.
- Abbreviations: `т.е./т.д./т.к./см./с./г./ул./кв./тыс./млн/млрд/№/руб./коп.`.
- Long digit runs (≥8 digits or after `ИНН/номер/телефон/счёт`) are read
  digit-by-digit; `495 123 45 67` after `номеру` likewise.
- Latin: **letter-by-letter** Russian letter names (`abc → эй би си`), frozen
  policy so no Latin reaches the phoneme path.
- Punctuation: inventory marks are kept; `„ “ ”` → `«`/`»`, `– −` → `—`/`-`,
  `...` → `…`, ASCII `"` alternates `«`/`»`, everything else is dropped.
- `norm_text` in the CLI equals `stressed_text` (contract §5: the manifest
  string carries the marks). `plain_text` is the same without marks.

Known limits of the rule layer (stated, not hidden): case is inferred from the
preceding preposition and a simple next-word heuristic, not a morphological
analyser; `на с. 12` becomes `на страница двенадцать` (case of `страница` is
not changed); animate accusative is not distinguished; `с 5` uses the
instrumental reading.

## 4. Stress

Primary source is the RUAccent **runtime subset** from contract §9:
`accents.json.gz` (3,194,879 forms), `omographs.json.gz` (19,741), `yo_words`
(85,568), and `nn/nn_accent` for OOV. The omograph *transformer*, stress-usage
predictor and koziev morphology are deliberately not loaded (they add ~300 MB
of downloads and are not in the shipped subset). Context overrides:
`году → год+у`, `года → г+ода` (the dictionary's statistical default is the
wrong reading for years).

Calibration choices:

- Clitics (prepositions, conjunctions, particles in `stress.CLITICS`) stay
  **unmarked**; RUAccent also leaves them unstressed and marking them would
  teach the wrong prosody.
- Every other word with a vowel gets exactly one `U+0301`. If neither the
  dictionary, omographs nor the ONNX model supply a mark, `fallback_stress`
  uses ё → a small suffix table → penultimate vowel, and the sentence is
  flagged OOV.
- `ё` carries the acute too (`ё́`), so every marked word has one mark.

## 5. Measured numbers (8,000 sentences: batch1 + batch2)

Token count per sentence (BOS/EOS/punctuation/`<sp>` included, as the model sees it):

| min | p50 | p90 | p95 | p99 | max | mean | >512 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 46 | 96 | 110 | 149 | 219 | 56.50 | **0** |

Per category (mean tokens / max): conversation 62.9/152, reading 77.8/219,
date 47.4/70, money 45.5/65, name 42.2/76, number 40.1/59,
question 35.1/52, template 50.0/81.

OOV (not in the 3.19 M dictionary or omograph list):

| measure | value |
|---|---|
| sentences with ≥1 OOV content word | **704 / 8000 = 8.80 %** |
| word tokens served by nn_accent or fallback | **1,362 / 66,408 = 2.05 %** |
| OOV over content words only (66,408 − 11,825 clitics) | **1,362 / 54,583 = 2.50 %** |
| per-category sentence OOV | conversation 12.6 %, reading 18.4 %, name 5.9 %, question 1.7 %, date 1.0 %, number 0.21 %, money 0.28 %, template 0 % |

G2P (eSpeak 1.52.0 via `espeakng-loader`): **0 dropped units**, **0 stress-transfer
misses** (vowel-letter index vs vowel-phone index agreed for all 66,408 words),
all 48 phones used. Normalisation left no digits/Latin; token ids all in 0–75;
0 sentences over the 512-token limit.

## 6. 20 hand-checked stress examples

`✔` = correct stress on all content words; `✘` = wrong.

| # | source | output | verdict |
|---:|---|---|---|
| 1 | Договор подписан. | `догово́р подпи́сан.` | ✔ |
| 2 | Каталог обновили. | `катало́г обнови́ли.` | ✔ |
| 3 | Она красивее сестры. | `она́ краси́вее сестры́.` | ✔ |
| 4 | Звонит телефон. | `звони́т телефо́н.` | ✔ |
| 5 | Позвонишь вечером? | `позвони́шь ве́чером?` | ✔ |
| 6 | Банты лежат на полке. | `ба́нты лежа́т на полке́.` | ✔ |
| 7 | Свёкла выросла на грядке. | `свё́кла вы́росла на гря́дке.` | ✔ |
| 8 | Жалюзи повесили вчера. | `жалюзи́ пове́сили вчера́.` | ✔ |
| 9 | Шофёр приехал вовремя. | `шофё́р прие́хал во́время.` | ✔ |
| 10 | В 2026 году мы поедем в Сочи. | `в две́ ты́сячи два́дцать шесто́м году́ мы́ пое́дем в со́чи.` | ✔ (`году́` override) |
| 11 | До 2026 года оставалось два месяца. | `до две́ ты́сячи два́дцать шесто́го го́да остава́лось два́ ме́сяца.` | ✔ (`го́да`) |
| 12 | Пятое мая — праздник. | `пя́тое ма́я — пра́здник.` | ✔ |
| 13 | Это стоит двести сорок три рубля пятьдесят копеек. | `э́то сто́ит две́сти со́рок три́ рубля́ пятьдеся́т копе́ек.` | ✔ |
| 14 | Доставка займёт полтора часа, а дорога — полторы минуты. | `доста́вка займё́т полтора́ ча́са, а доро́га — полторы́ мину́ты.` | ✔ |
| 15 | Замок на двери сломался. | `за́мок на две́ри слома́лся.` | ✔ (castle) |
| 16 | Он повесил новый замок. | `о́н пове́сил но́вый за́мок.` | **✘** should be `замо́к` |
| 17 | Мука рассыпалась по столу. | `му́ка рассы́палась по столу́.` | **✘** should be `мука́` (flour) |
| 18 | Я живу в двадцати километрах от Москвы. | `я́ живу́ в двадцати́ киломе́трах от москвы́.` | ✔ |
| 19 | Скидка двадцать процентов действует до тридцатого июня. | `ски́дка два́дцать проце́нтов де́йствует до тридца́того ию́ня.` | ✔ |
| 20 | Магазин закрывается в девять вечера. | `магази́н закрыва́ется в де́вять вечера́.` | ✔ |

**18/20 correct; 2/20 wrong, both homographs** (`за́мок/замо́к`, `му́ка/мука́`).
This is the expected cost of not loading RUAccent's omograph transformer: the
dictionary has one dominant reading and the context is not scored. Adding
`nn/nn_omograph/tiny` (10.5 MB) would address it; it is already listed as part
of the shippable subset in contract §9 and should be wired in before the
teacher generates 17k clips. All non-homograph probes and every number/date
template in the 8k corpus were marked without transfer misses.

## 7. Open issues / what could not be verified

1. **Homographs** (2/20 above). `замок`, `мука`, `года/году` (the last is
   hard-overridden) remain context-blind. The omograph transformer is not
   loaded in this task.
2. **Full case agreement** is heuristic, not morphological. `на с. 12` gets
   `страница` instead of `странице`; animate accusative is not modelled;
   `с 5` is instrumental. Tests encode this contract.
3. **Stress model quality** outside the 20 examples was not audited with
   `meter2`; that is F5/screening work. The 8.8 % sentence OOV is the input to
   that audit, not a quality claim.
4. **eSpeak is GPL-3.0 and offline-only.** `g2p.py` uses the dylib for training
   labels; nothing from eSpeak is shipped in the Rust binary. F7 must use the
   MIT rules/lexicon instead (R5 option 1).
5. **RUAccent memory**: the dictionary costs 1.53 GB RSS in Python. Fine for
   annotation; the Rust plan already replaces it with a trie/FST.
6. **F6 doc**: `docs/aligner-mapping.md` needs the `ŋ→n` row and a note that
   F3 keeps `ɵ/a/ɪ/ɛ/ə` distinct at the token layer while mapping them to
   F6's canonical labels. No other divergence found.
7. **Speed** was measured only on this M5 (8k sentences in 2.6 s after the
   2.2 s load, one process, word cache). Not benchmarked on other hardware.
