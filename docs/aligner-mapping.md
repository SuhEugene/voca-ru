# eSpeak-ru → aligner label mapping (contract)

What F3/F7 may assume when they turn eSpeak `ru` output into durations. This
table is **for forced alignment only**: it says which label in the recogniser's
`vocab.json` represents a given eSpeak phone. The token inventory the model
trains on is F3's decision and does not have to equal this table — but every
training phone must have a row here, or its clips cannot be aligned.

Written by F6, 2026-11-19. Evidence and measurement numbers:
[`reports/F6-align.md`](../reports/F6-align.md). Implementation:
[`scripts/align_spike/mapping.py`](../scripts/align_spike/mapping.py).

## 1. Pinned versions

| Thing | Value |
|---|---|
| eSpeak NG | 1.52.0 (Homebrew, `--ipa --sep=_ -v ru`) |
| Aligner | `facebook/wav2vec2-lv-60-espeak-cv-ft`, commit `ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4` |
| Model file | `pytorch_model.bin`, 1,263,535,127 bytes, SHA-256 `3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e` |
| `vocab.json` | 392 labels |
| Alignment | CTC Viterbi, 20 ms emissions at 16 kHz, onset offset −19 ms, durations on the 24 kHz / hop-256 grid |

The model was fine-tuned in 2021 on that year's eSpeak label set. Every
mismatch below comes from eSpeak having changed its Russian labels since, or
from phones that only Russian needs.

## 2. Inventory

`espeak-ng -v ru -q --ipa --sep=_` over the 775 FLEURS `ru_ru` test transcripts
(93,780 phone tokens): **71 distinct phones**. The 48 most frequent cover 99.5 % of tokens; the rest
are allophones, and a tail of about a dozen foreign phones (`θ ð ɹ ɻ ɒ ɔ əʊ eɪ aɪ
ɑː iː ɜː ø` …) that appear only because FLEURS sentences contain Latin words.
F3 owns the policy for those; the table maps them so a clip cannot fail mid-way.

All 71 are labels of the recogniser, so nothing is unalignable. What varies is
how well the model hears them; the `prob` column below is the mean CTC
posterior over the phone's aligned frames on 131 real Russian clips.

## 3. Table

`kind` is the decision: `exact` (eSpeak phone is a model label), `canonical`
(phone is a label but another label scores better — use the target), `fallback`
(phone is not a label; nearest target). Counts are over 775 sentences; `n` and
`frames`/`prob` are the measured occurrences on the audio (mean frames of
10.67 ms).

| phone | count | label | kind | n | frames | prob |
|---|---:|---|---|---:|---:|---:|
| `ʌ` | 8681 | `ʌ` | exact | 1367 | 8.75 | 0.54 |
| `i` | 7949 | `i` | exact | 1306 | 7.77 | 0.65 |
| `n` | 4479 | `n` | exact | 746 | 6.74 | 0.82 |
| `t` | 4445 | `t` | exact | 709 | 6.04 | 0.81 |
| `s` | 3709 | `s` | exact | 602 | 10.67 | 0.82 |
| `ɪ` | 3661 | `i` | canonical | 582 | 14.67 | 0.46 |
| `ɑ` | 3621 | `ɑ` | exact | 612 | 11.02 | 0.72 |
| `o` | 3472 | `o` | exact | 584 | 10.62 | 0.64 |
| `r` | 3110 | `r` | exact | 481 | 5.77 | 0.90 |
| `y` | 2905 | `y` | exact | 538 | 10.96 | 0.37 |
| `v` | 2893 | `v` | exact | 455 | 6.8 | 0.85 |
| `j` | 2870 | `j` | exact | 469 | 9.95 | 0.19 |
| `k` | 2435 | `k` | exact | 432 | 8.49 | 0.89 |
| `a` | 2408 | `ʌ` | canonical | 393 | 15.51 | 0.55 |
| `e` | 2320 | `e` | exact | 374 | 10.8 | 0.55 |
| `p` | 2282 | `p` | exact | 381 | 7.61 | 0.92 |
| `ɭ` | 2157 | `ɭ` | exact | 339 | 6.84 | 0.59 |
| `u` | 1949 | `u` | exact | 303 | 13.17 | 0.64 |
| `m` | 1940 | `m` | exact | 312 | 12.69 | 0.85 |
| `ɭʲ` | 1816 | `ɭʲ` | exact | 319 | 4.29 | 0.80 |
| `tʲ` | 1767 | `tʲ` | exact | 291 | 8.47 | 0.68 |
| `nʲ` | 1693 | `nʲ` | exact | 244 | 4.93 | 0.91 |
| `d` | 1674 | `d` | exact | 276 | 5.17 | 0.75 |
| `rʲ` | 1351 | `rʲ` | exact | 229 | 3.37 | 0.88 |
| `b` | 1171 | `b` | exact | 180 | 4.98 | 0.79 |
| `sʲ` | 1157 | `sʲ` | exact | 186 | 12.71 | 0.71 |
| `z` | 1139 | `z` | exact | 201 | 7.33 | 0.83 |
| `x` | 1068 | `x` | exact | 197 | 14.82 | 0.74 |
| `mʲ` | 1050 | `mʲ` | exact | 183 | 5.1 | 0.79 |
| `ɡ` | 1020 | `ɡ` | exact | 174 | 4.98 | 0.74 |
| `tʃʲ` | 1012 | `tʃʲ` | exact | 165 | 8.79 | 0.65 |
| `ts` | 976 | `ts` | exact | 182 | 9.24 | 0.75 |
| `f` | 953 | `f` | exact | 166 | 19.52 | 0.67 |
| `ja` | 935 | `ja` | exact | 156 | 17.48 | 0.12 |
| `dʲ` | 926 | `dʲ` | exact | 130 | 4.64 | 0.68 |
| `vʲ` | 887 | `vʲ` | exact | 135 | 3.61 | 0.65 |
| `ɛ` | 865 | `e` | canonical | 114 | 17.12 | 0.16 |
| `ʃ` | 848 | `ʃ` | exact | 122 | 9.8 | 0.83 |
| `ʒ` | 799 | `ʒ` | exact | 152 | 6.46 | 0.91 |
| `kʲ` | 495 | `kʲ` | exact | 89 | 4.27 | 0.74 |
| `ju` | 463 | `ju` | exact | 74 | 9.66 | 0.07 |
| `ɕ` | 418 | `ɕ` | exact | 74 | 8.36 | 0.66 |
| `pʲ` | 403 | `pʲ` | exact | 75 | 9.6 | 0.63 |
| `ɡʲ` | 263 | `ɡʲ` | exact | 42 | 4.14 | 0.31 |
| `ə` | 255 | `ʌ` | canonical | 39 | 9.03 | 0.55 |
| `bʲ` | 252 | `bʲ` | exact | 39 | 3.51 | 0.58 |
| `ʑ` | 206 | `ʑ` | exact | 43 | 7.02 | 0.32 |
| `u"` | 174 | `u` | canonical | 25 | 10.24 | 0.26 |
| `ɵ` | 145 | `o` | canonical | 29 | 9.03 | 0.45 |
| `fʲ` | 102 | `fʲ` | exact | 13 | 35.54 | 0.22 |
| `ɔ` | 40 | `o` | canonical | 6 | 7.33 | 0.04 |
| `iː` | 33 | `i` | canonical | 5 | 20 | 0.77 |
| `ɪ^` | 28 | `i` | canonical | 1 | 2 | 0.18 |
| `eɪ` | 21 | `e` | canonical | 5 | 14.8 | 0.00 |
| `aɪ` | 16 | `i` | canonical | 1 | 6 | 0.79 |
| `əʊ` | 11 | `o` | canonical | 2 | 12.5 | 0.48 |
| `ɹ` | 9 | `r` | canonical | 1 | 4 | 0.99 |
| `tʃ` | 8 | `tʃʲ` | canonical | 3 | 24 | 0.00 |
| `dʒ` | 7 | `dʒ` | exact | – | – | – |
| `ɑː` | 7 | `ɑ` | canonical | 1 | 18 | 0.81 |
| `θ` | 5 | `t` | canonical | – | – | – |
| `l` | 5 | `l` | exact | 1 | 7 | 0.10 |
| `eə` | 3 | `eə` | exact | – | – | – |
| `ɒ` | 3 | `ɒ` | exact | – | – | – |
| `h` | 3 | `h` | exact | – | – | – |
| `əl` | 3 | `l` | canonical | 1 | 22 | 0.02 |
| `ɜː` | 2 | `ɜ` | canonical | – | – | – |
| `dʒʲ` | 2 | `dʒ` | canonical | 1 | 11 | 0.00 |
| `uː` | 2 | `uː` | exact | – | – | – |
| `w` | 2 | `v` | canonical | 1 | 2 | 0.03 |
| `ø` | 1 | `o` | canonical | – | – | – |

Rows with `kind = canonical` or `fallback` are the whole contract. The exact
rows need no decision; they are listed so the inventory is complete and a
missing phone is obvious.

## 4. Decisions and why

Every number below is mean log-probability of that label over the phone's
aligned span (higher is better), measured on FLEURS.

| eSpeak-ru phone | use label | why |
|---|---|---|
| `zʲ` | `ʑ` | eSpeak's soft /z/; the 2021 label set has no `zʲ`, only `ʑ`. **Not observed** in 775 sentences (rare), so unverified on audio — same substitution Polish uses. |
| `ɕː` | `ɕ` | long /ɕ/ (щ); no length variant in the label set. **Not observed** either. |
| `u"` | `u` | eSpeak's /u/ after a soft consonant (ю). Model has `u"` but `u` scores −2.01 vs −4.12. |
| `ɪ^` | `i` | allophone of /и/ after a hard consonant; 28 occurrences in 775 sentences, 1 on the audio, so the choice follows `ɪ→i` and is unverified. |
| `ɵ` | `o` | ё. `o` −1.30 vs `ɵ` −6.38; `o` was the top label on 21 of 29 spans. |
| `ə` | `ʌ` | reduced vowel. `ʌ` −0.86 vs `ə` −5.23; top label on 31 of 39 spans. |
| `ɪ` | `i` | reduced /и/,/е/. `i` −1.50 vs `ɪ` −2.42; top label on 346 of 582 spans. |
| `a` | `ʌ` | /a/ after a hard consonant. `ʌ` −0.92 vs `a` −2.08; top label on 322 of 393 spans. |
| `ɛ` | `e` | э. `e` −3.26 vs `ɛ` −3.46 — a tie; top label is `ʌ` 23, `e` 21, `ɛ` 17. Low confidence, kept only because a label must be chosen. |
| `tʃ` | `tʃʲ` | foreign /tʃ/ (Latin words). Russian ч is always `tʃʲ`. |
| `dʒʲ` | `dʒ` | foreign; no soft `dʒ` in the label set. |
| `w` | `v` | foreign; Russian has no /w/. |
| `ɹ` `ɻ` | `r` | foreign retroflex approximants. |
| `θ` `ð` | `t` `d` | foreign dental fricatives. |
| `eɪ` `aɪ` `ɔ` `əʊ` `iː` `ɑː` `ɜː` `ø` `øː` `œ` `uː` `ɒ` `eə` | nearest Russian vowel (`e i o o i ɑ ɜ o o o u ɒ eə`) | Latin words only; F3 decides whether to transliterate, drop or keep them. |
| `əl` `n̩` `l̩` `m̩` `ɾ` | `l` `n` `l` `m` `r` | syllabic/foreign allophones. |

### Stress marks

`ˈ` and `ˌ` are **not** tokens. They are parsed into the parallel 0/1/2 feature,
exactly as upstream `tiny_voca.text.parse_espeak` does. eSpeak writes the mark
before the stressed vowel's unit (`p_ʌ_mʲ_ˈi_m_ʌ` → stress 1 on `i`).

### eSpeak artefacts that never reach the aligner

These are stripped or remapped by rule; if one survives, the clip is rejected
rather than silently aligned:

| artefact | where | action |
|---|---|---|
| `(en)`, `(ru)` | language switch inside Russian text | deleted before splitting (upstream regex) |
| empty unit from `s__rʲ` | between phones | dropped (empty string) |
| `a#`, `u#`, `I#`, `a#ː`, `@-` | word-final allophones in some eSpeak builds/voices | map to `a`, `u`, `ɪ`, `aː`, `ə` |
| `8` | eSpeak's internal name for ё's vowel in some builds | map to `o` |

### What does **not** work: splitting `ja` / `ju`

eSpeak emits `ja` and `ju` as single units. The model hears them poorly as
single labels (0.12 and 0.07 — blank is the top label on 50 of 135 and 33 of 62
spans). Splitting them into `j`+`a` and `j`+`u` for alignment was measured and
made it worse (0.036 / 0.038), because the model emits blank across the glide
either way. **Decision: keep the single eSpeak label.** Consequence for F3/F7:
the /j/ onset boundary inside я/е/ю/ё is not observable; the token's total
duration is bounded by its neighbours and is usable, but do not expect the
aligner to separate the glide from the vowel. Same for `j` as a standalone
phone (0.19, blank on 330 of 398 spans).

### Phones that align weakly and need QC tolerance

`ja` 0.12, `ju` 0.07, `j` 0.19, `ɛ` 0.16, `fʲ` 0.22, `ɡʲ` 0.31, `ʑ` 0.32,
`y` (ы) 0.37, `ɭ` 0.59. Everything else is ≥ 0.54 and most consonants ≥ 0.8.
`fʲ` has only 13 occurrences and a 35.5-frame mean (379 ms), so its low score is
probably a few mis-matched spans, not a systematic problem.

## 5. How F3/F7 should use this

1. Emit eSpeak phones exactly as `scripts/align_spike/espeak_ru.py` parses them
   (strip stress into 0/1/2, drop empty units, drop language tags).
2. For every phone, look up the label with this precedence: `CANONICAL`, then
   `vocab.json` membership, then `FALLBACK`. A phone with no label is a data
   bug: fail the clip, do not drop the phone.
3. Pin eSpeak to 1.52.0 and record the version in the frontend fingerprint, as
   the other ports do — 1.51 emits different Russian labels.
4. Keep the alignment label separate from the token symbol. A training inventory
   that keeps `ə` distinct from `ʌ` is fine; both align to `ʌ`.
