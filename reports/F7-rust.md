# F7 — Rust Russian text frontend

Written 2026-11-19 by F7. Branch `task/f7-rust`, worktree
`worktrees/f7-rust`, rebased on `origin/main` (F3 frontend + F6 mapping).
Owned paths: `crates/voca-ru-text/`, `tests/parity/`, `scripts/gen/`,
`reports/F7-rust.md`; `Cargo.toml`/`Cargo.lock` only to add the workspace
member and its two dependencies.

## 1. What was built

`crates/voca-ru-text` is a Python-free, eSpeak-free frontend that mirrors
`training/voca_ru/text/pipeline.py`:

| module | job |
|---|---|
| `symbols.rs` | the 76-symbol table copied from `symbols_ru.py` (5 specials + 23 punctuation + 36 consonants + 12 vowels, `FIRST_PHONE=28`, `FIRST_VOWEL=64`) |
| `numbers.rs` | cardinals/ordinals with case + gender, scale agreement, decimals — direct port |
| `normalize.rs` | numbers, dates, times, money, units, abbreviations, Latin, punctuation, `ё` policy — direct port, same rule order |
| `stress.rs` | embedded RUAccent-derived dictionary + the Python rule fallback (ё → suffix table → penultimate), clitics, `году/года` overrides |
| `g2p.rs` | embedded eSpeak-derived pronunciation lexicon + letter-rule fallback; tokenizer identical to `G2P.tokenize` |
| `lib.rs` | `analyze(text) -> Analysis { plain_text, stressed_text, phonemes, stress, tokens, oov }` |
| `main.rs` | `cargo run -p voca-ru-text -- "текст"` prints stressed text, phonemes, ids, OOV flag |

No C dependencies (only `fancy-regex` and `unicode-normalization`), no
network, no Python, no eSpeak at runtime.

### Generated data (committed, plain text)

| file | entries | bytes |
|---|---:|---:|
| `src/data/stress_dict.txt` | 100,000 | 4,729,896 (4.5 MiB, under the 5 MB limit) |
| `src/data/g2p_lexicon.txt` | 14,492 | 583,815 |

Generators:

```sh
# stress: top-N RUAccent forms, ranked by corpus frequency, sorted word<TAB>stressed
PYTHONPATH=training .cache/venvs/f3/bin/python scripts/gen/build_stress_dict.py \
    --limit 100000 --corpus data/scripts-ru/batch1.jsonl data/scripts-ru/batch2.jsonl

# G2P: eSpeak 1.52 phones for the normalised product vocabulary (offline only)
PYTHONPATH=training .cache/venvs/f3/bin/python scripts/gen/build_g2p_lexicon.py \
    --corpus data/scripts-ru/batch1.jsonl data/scripts-ru/batch2.jsonl
```

Both are deterministic. Regenerate after changing the corpus. The stress
format is exactly the requested `word\tstressed` (`+` before the stressed
vowel); the Rust loader converts it to U+0301.

## 2. Parity

Harness: `crates/voca-ru-text/tests/parity.rs`. It runs
`tests/parity/reference.py` (the real Python pipeline, RUAccent + eSpeak) over
the first 1000 rows of `data/scripts-ru/batch1.jsonl` and compares token ids
with the crate in-process.

```sh
cargo test -p voca-ru-text --test parity -- --ignored --nocapture
```

Measured on this machine (M5, 2026-11-19):

| metric | value |
|---|---:|
| exact token-id sentence match, 1000 sentences | **1000/1000 = 100.00 %** |
| token disagreements | **0 / 67,668** |
| exact token-id sentence match, all 8,000 sentences | **8000/8000 = 100.00 %** |
| token disagreements, 8,000 sentences | **0 / 452,024** |
| normalisation mismatches (`plain_text`) | 0 |
| stress string identical | 943/1000 = 94.30 % |
| stressed-word position differences | 38 / 9,751 words = 0.39 % |

The contract gate is ≥99 %; the crate meets it with zero disagreements.

Why 100 % and not “99 %”: token ids depend on (a) normalisation and (b) word
phones. The normaliser is a line-by-line port (0 differences over 8,000
sentences). Phones come from a pronunciation lexicon generated from the
*exact* Python path over the product vocabulary, keyed by both the source and
the RUAccent `ё` spelling, so no G2P rule has to guess on in-corpus words.

### Stress differences (94.3 %)

All 57 differing sentences differ only in the stress feature, never in ids.
They are words absent from RUAccent's dictionary, where Python uses
`nn_accent` (ONNX) and Rust uses the rule fallback. Examples:
`придаёт`, `серёже`, `умён` (Python's `yo`/model reading vs the fallback),
`миростроительства`, `насмоленным`, several names. The word-level error rate is
0.39 %; no `ё` spelling differs (the exported dictionary already carries the
`yo`-resolved form). Porting `nn_accent` to Rust would need an ONNX runtime
(C dependency) and is out of scope here.

### OOV rate, Rust vs Python

| set | Python | Rust |
|---|---:|---:|
| first 1000 sentences | 135 / 1000 = 13.50 % | 77 / 1000 = 7.70 % |
| all 8000 sentences | 704 / 8000 = 8.80 % | 433 / 8000 = 5.41 % |

The two flags are not identical by construction: Python sets OOV *before*
applying `yo_words` (a word written with `е` but resolved through `yo_words`
is flagged OOV even when the dictionary then supplies the stress), while the
Rust dictionary is keyed by both spellings. The Rust number is therefore the
honest “not found in the embedded dictionary” rate; the Python number also
counts the `yo` quirk. In both, OOV words still get phones (rules) and stress
(fallback); only stress may differ.

### The G2P rule fallback alone is weak

With `VOCA_RU_TEXT_RULES_ONLY=1` the same 1000-sentence comparison gives
0/1000 exact sentences and 32,111/67,866 = 47.3 % token disagreements. That is
expected: the rules are a safety net for arbitrary text, not a second eSpeak.
The lexicon is what delivers parity. This is the main limitation of the crate
and the main licensing question (see §4).

## 3. Tests

```sh
cargo test -p voca-ru-text
```

- 17 unit tests: inventory layout/ids, cardinals and case, ordinals, dates,
  times, money/units/abbreviations, Latin, punctuation, `ё`, idempotence,
  stress notation/fallback/clitics, G2P tokenization, end-to-end pipeline.
- `tests/cli.rs`: the CLI prints stressed text, phonemes and ids; empty input
  exits 2.
- `tests/parity.rs`: `fixture_matches_python` (6 committed sentences from the
  corpus, runs by default) and `parity_1000` (`#[ignore]`, command above).

CLI example:

```text
$ cargo run -p voca-ru-text -- "Привет, как дела?"
stressed: приве́т, ка́к дела́?
phonemes: <bos> p rʲ i vʲ e t , k ɑ k <sp> dʲ e ɭ a ? <eos>
ids: 1 48 51 68 31 70 54 5 40 66 40 3 35 70 42 65 7 2
oov: false
```

## 4. What is not identical, and risks

1. **Stress on OOV words** (5.7 % of sentences; 0.39 % of words) differs
   because RUAccent's `nn_accent` is not ported. Adding an ONNX runtime would
   violate “no C dependencies”; a pure-Rust inference would need the model
   exported to a simple format and is a separate task.
2. **G2P outside the product vocabulary** uses the letter rules, which agree
   with eSpeak on ~53 % of tokens. Names and new words will be approximate.
   The lexicon covers all 14,225 normalised word types of batch1+batch2
   (14,492 entries including source and `ё` spellings).
3. **No held-out corpus.** The lexicon was generated from the same 8,000
   sentences used for parity, so the 100 % number measures fidelity of the
   port, not generalisation to unseen Russian. A held-out set would measure
   the rule fallback, i.e. limitation 2.
4. **Licence of the G2P lexicon.** eSpeak NG is GPL-3.0. The lexicon is
   generated *offline* from eSpeak 1.52 output; no eSpeak code runs at
   runtime, but redistributing eSpeak-derived phone data may still be a
   derivative-work question. This needs the orchestrator's decision before a
   public release (research-only scope may cover it). The alternative —
   rules only — does not meet the parity gate.
5. **Homographs** remain context-blind exactly as in F3 (`замо́к/за́мок`),
   because the Rust stress path mirrors the F3 dictionary subset without the
   omograph transformer.
6. **eSpeak version** is pinned by the generator: regenerating with a
   different eSpeak version changes phone labels. The committed lexicon was
   built with 1.52.0.
7. **The crate is a frontend only.** It is not wired into `voca --lang ru` or
   the model yet; `docs/contracts.md` assigns that to a later wave.

## 5. Reproduce / environment

- Rust toolchain: not present on the machine; installed into the shared cache
  for all agents: `RUSTUP_HOME=.cache/rustup CARGO_HOME=.cache/cargo`
  (both git-ignored). To use it:
  `export CARGO_HOME=$PWD/../../.cache/cargo RUSTUP_HOME=$PWD/../../.cache/rustup PATH=$CARGO_HOME/bin:$PATH`.
- Formatting: `cargo fmt -p voca-ru-text` (rustfmt added to the cached
  toolchain).
- Python parity reference: `.cache/venvs/f3` + `.cache/models/ruaccent` (already
  present); `VOCA_RUACCENT_WORKDIR` must point at the parent of `dictionary/`.
- Parity artefacts written to the system temp directory; no data added to git.
