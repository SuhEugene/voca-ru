# Sources and licences — Russian script corpus (F2)

Frozen 2026-11-19. Regenerate with:

```sh
. "$(git rev-parse --show-toplevel)/scripts/cache-env.sh"
python3 scripts/text/build_ru_scripts.py
```

The build reads pinned copies from `$VOCA_RU_CACHE/text/` (default
`/Users/suheugene/projects/22-tiniest-voca-ru/.cache/text`), verifies every
SHA-256 below, and writes byte-identical `batch1.jsonl`, `batch2.jsonl` and
`SPLITS.sha256`. Nothing is downloaded unless the cache file is missing; a
digest mismatch aborts the build.

## Summary

| batch | source | sentences | licence | `origin` |
|---|---|---:|---|---|
| 1 | Mozilla Common Voice ru `sentence-collector.txt` | 3000 | CC0-1.0 | `common_voice_sentence_collector` |
| 1 | Russian Wikisource, public-domain prose | 900 | public domain (site layer CC BY-SA 4.0, not used) | `wikisource_pd` |
| 1 | Written for this project | 600 | CC0-1.0 (project waiver) | `project_llm` |
| 2 | Sentence templates in `build_ru_scripts.py` | 2600 | CC0-1.0 (project waiver) | `project_templates` |
| 2 | Written for this project | 900 | CC0-1.0 (project waiver) | `project_llm` |

Total 8000 sentences. Batch 1 is split by sentence 3870 train / 315 val /
315 test (86/7/7, seed `20261120`); batch 2 is train only. Val/test ids are
frozen in `data/scripts-ru/SPLITS.sha256` together with the hashes of both
JSONL files.

The released list is CC0-1.0 except the Wikisource sentences, which stay
public-domain. No Wikipedia, OPUS/OpenSubtitles, news corpus, LLM prompt text
or other third-party text is included.

## Common Voice — CC0-1.0

- URL: <https://raw.githubusercontent.com/common-voice/common-voice/main/server/data/ru/sentence-collector.txt>
- SHA-256: `e0869a008310c452f63fd02e4d52f8abc58df2de73e558c486719161e3ad1b87`
- 46 090 lines. Common Voice bulk-submission rules require the sentences to be
  public domain / CC0 and ask contributors to avoid numbers and special
  characters. The file is distributed by the Common Voice project on GitHub;
  the corpus itself is CC0-1.0.
- Selection: normalise whitespace, strip list markers and wrapping quotes,
  reject anything outside Cyrillic + ordinary punctuation (Latin homoglyphs,
  digits, URLs, unbalanced quotes, repeated punctuation), require 5–30 words
  and final `.` `?` or `!`, deduplicate by a case/ё/punctuation-insensitive
  key, then draw 3000 with a seeded RNG from the sorted pool.
- Known quirks kept in the pool on purpose: occasional 19th-century quotes and
  dialogue fragments are part of the file's natural style; all were checked
  against the filter above.

## Wikisource — public domain

Fetched through the MediaWiki API (`action=parse&oldid=…`), one request per page
with a 2 s delay. Wikimedia rate-limited aggressive clients during this build;
if a cache file is missing, the build may need a pause between fetches.

All authors died before 1917, so the literary text is public domain in Russia
and the US. The Wikisource site layer (annotations, footnotes, prefaces) is
CC BY-SA 4.0 and is **not** used: the builder only takes `<p>` prose from the
pinned revision, drops wiki licence/footer boilerplate, and records author and
work for attribution as scholarly practice.

| work | author | died | written | revid | sentences | cached API response SHA-256 |
|---|---|---:|---:|---:|---:|---|
| Ася | И. С. Тургенев | 1883 | 1858 | 5759963 | 236 | `3e1303cd1ee6f5d649297053cf70175cf816903b404a5914b41fc15141942984` |
| Белые ночи | Ф. М. Достоевский | 1881 | 1848 | 5733096 | 217 | `cb10844bad5a635aa79ffe224978a4680aefa2a34f6b96f1ed8eec87e0ca9ca1` |
| Пиковая дама | А. С. Пушкин | 1837 | 1834 | 5644345 | 140 | `0e972157e8362a2680b9ba34dc743550678cb2e4b610f1a624ade908d951dc3c` |
| Муму | И. С. Тургенев | 1883 | 1854 | 5619020 | 109 | `ff2038b469238360b2ac8952aab0fbf434a56cf3fc3fda598c5d4d7755e9a67c` |
| Ионыч | А. П. Чехов | 1904 | 1898 | 5321162 | 80 | `8c788a0e6fb62068e3cce0ed7c17c1c213c477c2163d574ddca887993e4023d0` |
| После бала | Л. Н. Толстой | 1910 | 1911 | 5638188 | 61 | `be2c3af162b6517ed2ac7f529cc64e7fb44aa6abb23f09e6767113ac24c32f17` |
| Дама с собачкой | А. П. Чехов | 1904 | 1899 | 5736015 | 57 | `f4e74befd252d908057b963e8fe06b1adc470cb7e475ca2b7ea3fd9666d12113` |

Row URL: `https://ru.wikisource.org/w/index.php?oldid=<revid>` and
`source_id` is `wikisource:<page title>@<revid>`.

Sentence extraction: split each paragraph on `.!?` before a capital letter,
quote or dash; strip leading dashes/quotes; reject fragments shorter than five
words, dialogue tails (`— сказал он.`), wiki boilerplate, digits/Latin, and
sentences outside 5–30 words.

## Project-own sentences and templates

- `scripts/text/own/batch1.txt` — 600 conversational sentences written for
  this project (by the F2 agent). No source text was copied; topics and wording
  are original. Deduplicated against Common Voice and Wikisource.
- `scripts/text/own/batch2.txt` — 900 sentences written for this project with
  numbers, dates, times, money, names and questions; numbers are already
  written as words.
- Templates in `scripts/text/build_ru_scripts.py` (`gen_numbers`, `gen_dates`,
  `gen_times`, `gen_money`, `gen_names`, `gen_questions`, `gen_lists`,
  `gen_abbreviations`, 2600 selected) cover the phenomena Common Voice forbids:
  cardinal/ordinal numerals with noun agreement, genitive dates, clock times,
  roubles/kopecks, personal names in four cases, questions, lists and
  abbreviations (`т. е.`, `см.`, `ул.`, `д.`, `кв.`, `руб.`, `г.`, `им.`,
  `с.`, `стр.`).
- Both are released under the project's CC0-1.0 waiver. The templates and
  written sentences contain no third-party text, and no copyrighted passage was
  used as a prompt.

## Filtering and split policy

Every row (all sources) satisfies:

- 5–30 whitespace-separated words, ends with `.`, `?` or `!`;
- Cyrillic (including `ё`) and ordinary punctuation only — no digits, no Latin,
  no URLs, no `..`/`,,`/`??`/`!!`, balanced `«»`;
- unique under `lowercase → ё→е → strip punctuation → collapse spaces`;
- `id = sha256(normalised text)[:16]`, so the same sentence always gets the
  same id;
- no sentence occurs in two batches or two splits.

Batch 1 receives its split by shuffling the sorted id list with seed `20261120`
(315 val, 315 test). Batch 2 is train only. `SPLITS.sha256` freezes both file
hashes and the sorted `id split` manifest hash.

## Licence notes

- CV file: CC0-1.0, no attribution required.
- Wikisource prose: public domain; authors/works recorded above.
- Own sentences and templates: project-own, CC0-1.0 waiver.
- No downloaded data, audio, checkpoints or caches are committed; only the two
  JSONL files, this page and `SPLITS.sha256` live under `data/scripts-ru/`.
