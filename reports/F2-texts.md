# F2 — Russian script corpus

Status: **done, frozen** (2026-11-19). Branch `task/f2-texts`.

## Deliverables

| path | what |
|---|---|
| `scripts/text/build_ru_scripts.py` | seeded, idempotent builder (sources, filters, templates, numerals, splits) |
| `scripts/text/own/batch1.txt` | 600 hand-written conversational sentences |
| `scripts/text/own/batch2.txt` | 900 hand-written sentences with numbers/dates/money/names/questions |
| `data/scripts-ru/batch1.jsonl` | 4500 sentences, train/val/test |
| `data/scripts-ru/batch2.jsonl` | 3500 sentences, train only |
| `data/scripts-ru/SPLITS.sha256` | frozen file hashes + split-manifest hash |
| `data/scripts-ru/sources.md` | provenance, pinned revisions, hashes, licences |
| `tests/text/test_scripts_ru.py` | 8 contract tests (stdlib `unittest`, no deps) |

Also changed: `.gitignore` now re-includes `data/scripts-ru/` (the contract says
these lists may be committed; all other `data/` stays ignored).

## Numbers

```
sentences 8000, words 66 876, average 8.4 words
batch1 4500 = train 3870 / val 315 / test 315   (86/7/7, seed 20261120)
batch2 3500 = train 3500
```

| origin | batch | count | licence |
|---|---:|---:|---|
| Common Voice ru sentence-collector | 1 | 3000 | CC0-1.0 |
| Wikisource PD prose (5 authors, 7 works) | 1 | 900 | public domain |
| written for this project | 1 | 600 | CC0-1.0 (project waiver) |
| templates in the builder | 2 | 2600 | CC0-1.0 (project waiver) |
| written for this project | 2 | 900 | CC0-1.0 (project waiver) |

Wikisource selection per work: Ася 236, Белые ночи 217, Пиковая дама 140,
Муму 109, Ионыч 80, После бала 61, Дама с собачкой 57.

Categories (the single `category` field): conversation 3936, number 945,
date 578, question 541, name 441, money 359, template 300, reading 900.

Content-based hard-case counts (tests enforce the thresholds):

| phenomenon | rows | required |
|---|---:|---:|
| contains a number word | 1975 | ≥400 |
| month/date | 310 | ≥200 (with time) |
| time (`час`) | 245 | — |
| money (`рубл/копе/…`) | 368 | ≥150 |
| question mark | 835 | ≥200 |
| Russian personal name | 851 | ≥100 |

## Reproduce

```sh
. "$(git rev-parse --show-toplevel)/scripts/cache-env.sh"
python3 scripts/text/build_ru_scripts.py     # prints counts by category
python3 -m unittest discover -s tests/text -v
```

The builder reads pinned files from `$VOCA_RU_CACHE/text/` and verifies their
SHA-256 (Common Voice file, seven Wikisource API responses). Two consecutive
runs produced byte-identical `batch1.jsonl`, `batch2.jsonl`, `SPLITS.sha256`.
If a cache file is missing it downloads it once; Wikimedia rate-limited rapid
API calls during this build, so a fresh fetch of all pages may need pauses
(2 s between requests is used).

Splits: sorted ids shuffled with `random.Random(20261120)`, first 315 val,
next 315 test. `SPLITS.sha256` stores both file hashes and the hash of the
sorted `id split` manifest, so a later task can prove val/test did not change.

## Filtering

Applied to every source: 5–30 words, final `.`/`?`/`!`, Cyrillic + ordinary
punctuation only, no digits, no Latin (catches CV homoglyphs), no URLs, no
repeated punctuation, balanced `«»`, unique under
`lowercase → ё→е → strip punctuation → collapse spaces`. Numbers in templates
are generated as Russian words by a cardinal/ordinal writer with noun gender
and 11–14 agreement; dates use genitive ordinals and genitive months; money
uses рубль/копейка agreement; times are spoken-style words.

## Open issues / deviations

- **Question share is 10.4 %, not the ~25 % soft suggestion in
  `docs/datasets-ru.md`.** All hard thresholds are met; raising questions to
  25 % would mean rewriting ~1200 rows and was not in the acceptance criteria.
- **Gogol was dropped.** The Wikisource pages «Шинель (Гоголь)» and
  «Капитанская дочка (Пушкин)» are edition-index pages, not prose; the build
  uses only pages that render paragraph text (`<p>`), so five authors remain.
- **Homograph ambiguity is not removed**, only avoided by wording where easy.
  Stress and omograph resolution are F3's job (`docs/contracts.md` §5).
- **Wikisource licence layer**: the literary text is PD; the site's CC BY-SA
  layer (footnotes, prefaces) is never taken. Wiki licence/footer paragraphs
  are filtered out explicitly.
- The builder hard-codes the default cache root for this machine
  (`/Users/suheugene/projects/22-tiniest-voca-ru/.cache`), overridable with
  `VOCA_RU_CACHE`, as AGENTS.md requires one shared cache.
- I could not verify every 19th-century sentence for modern stress ambiguity;
  spot checks of 40 random reading rows showed complete, grammatical sentences.
- `pytest` is not installed in the base interpreter; tests use stdlib
  `unittest` and run in 0.3 s.
