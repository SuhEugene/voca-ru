# F8 — batch 3 and sentence retirement

Branch `task/f8-batch3`, worktree `worktrees/f8-batch3`, based on `origin/main` `531033c`.

## What was built

### A. `scripts/text/build_ru_scripts.py --extra [N]`

`--extra [N]` (default **2500** when the flag has no value) takes N sentences from
the **unused** Common Voice ru sentence-collector pool (same pinned file and
SHA-256, `cc0-1.0`) and writes `data/scripts-ru/batch3.jsonl`:

- same schema as batch1/batch2 (`FIELDS` incl. `batch`), `batch: 3`,
- `split: "train"` for every row (no val/test: the frozen splits stay as they are),
- same `text_id` scheme: `sha256(norm_key(text))[:16]`,
- same `clean_text` / `text_problem` filters (`text_problem` is applied inside
  `common_voice_pool`, which is reused unchanged),
- `category: "conversation"`, `origin: common_voice_sentence_collector`,
  `source_id: cv-ru-sentence-collector`, `source_license: cc0-1.0`, `source_url` = CV URL,
- selection is `sorted(unused)` → `random.Random("20261119:cv-extra").shuffle` → prefix,
  so the pick is a fixed permutation and a larger `--extra` only appends.

Wikisource is not used for batch 3: 34 427 unused CV sentences remain, so the
top-up path was never needed (the builder exits with a clear message if the pool
is ever smaller than N).

`write_splits()` now hashes every batch file that exists (batch1, batch2 and
batch3 if present) and appends `batch3.jsonl sha256:...` plus
`batch3-train=2500 batch3-val=0 batch3-test=0` to `SPLITS.sha256`. Running the
builder **without** `--extra` keeps an existing `batch3.jsonl` and its entry
(verified: the plain run rewrites an identical `SPLITS.sha256`).

`batch1.jsonl` and `batch2.jsonl` are byte-identical to the committed files
(sha256 `56fd25c3…`, `72d82135…`, checked after every run).

### B. `scripts/gen/retire.py`

```
python3 scripts/gen/retire.py --corpus data/corpus-ru --out data/scripts-ru/retired.jsonl
```

Reads `<corpus>/manifest.jsonl` (accepted) and `<corpus>/review.jsonl` (dropped),
groups by `text_id`, and writes one row per sentence with **zero accepted clips
in any voice**:

```json
{"id": "<text_id>", "text": "...", "voices_rejected": ["f_neutral", "m_neutral"], "reasons": ["..."]}
```

`voices_rejected` and `reasons` are sorted; output rows are sorted by `id`, so the
file is reproducible. Prints:

```
sentences retired: N
sentences with one of 2 voices: N     (accepted in some but not all voices)
sentences fully healthy: N            (accepted in every voice)
```

`scripts/gen/build_corpus.py::load_rows` now seeds its `seen` set with
`retired_ids()`: when `data/scripts-ru/retired.jsonl` exists those ids are never
returned, so they are never generated again (the filter is silent and is a no-op
when the file is absent). `scripts/gen/split_corpus.py` calls the same
`load_rows`, so it inherits the skip. `build_corpus.py` also exports `RETIRED`,
so a different path is possible from one place.

## Reproduce

```sh
cd worktrees/f8-batch3
. "$(git rev-parse --show-toplevel)/scripts/cache-env.sh"
PY=/Users/suheugene/projects/22-tiniest-voca-ru/.cache/venvs/r4/bin/python

$PY scripts/text/build_ru_scripts.py --extra   # writes batch3.jsonl + SPLITS.sha256
$PY scripts/text/build_ru_scripts.py --extra   # second run: byte-identical
git status --porcelain                          # clean after the two runs
sha256sum data/scripts-ru/*.jsonl

$PY tests/text/test_scripts_ru.py
$PY tests/text/test_retire.py
```

## Numbers

| Thing | Value |
|---|---:|
| batch1 / batch2 | 4 500 / 3 500 (unchanged, byte-identical) |
| batch3 written | **2 500** (all `split=train`) |
| corpus after batch 3 | 10 500 sentences → up to 21 000 clips |
| Common Voice pool (usable) | 37 427 |
| CV sentences used by batch1 | 3 000 |
| CV sentences used by batch3 | 2 500 |
| **remaining unused CV pool** | **31 927** |
| duplicate ratio | 0 ids / 0 norm-text vs batch1+batch2; 0 duplicates inside batch3 |
| batch3 word counts | 5–16 words |
| batch3 sha256 | `23d4b08e0380c74f41d85eca2dbb13b7f20993756cad70c8573ddefb40a8aff9` |

Retirement: the last reported screening was 1 433 rejected clips (8.9 %). The
exact number of *sentences* with zero accepted clips can only be computed from
the real `manifest.jsonl` / `review.jsonl`; `retire.py` was exercised on a
two-sentence fixture, not on `data/corpus-ru` (that directory does not exist in
this worktree — the corpus lives on the pod / in the main checkout).

## Wiring into the pod run

1. Add `batch3.jsonl` next to batch1/2 in the bundle and pass it to the builder:
   - `remote/pack.sh`: add `"$ROOT"/data/scripts-ru/batch3.jsonl` to the copy line;
   - `remote/lib/bundle_files.py`: add `"data/scripts-ru/batch3.jsonl"`;
   - `remote/generate.sh`: `--sources batch1.jsonl batch2.jsonl batch3.jsonl`.
   These files belong to other tasks, so they were **not** edited here.
2. Generate wave 2 (`remote/generate.sh --split train`), pull `manifest.jsonl`
   and `review.jsonl` back to the Mac.
3. Run `python3 scripts/gen/retire.py --corpus data/corpus-ru` once; commit
   `data/scripts-ru/retired.jsonl` (a list of sentence ids/texts, not audio).
4. Ship `retired.jsonl` in the same bundle (pack.sh + bundle_files.py). Every
   later `build_corpus.py` run then skips those sentences, including retries and
   `split_corpus.py`.
5. Re-run `retire.py` after every generation wave; it rewrites the file from the
   full manifest/review, so it is idempotent.

## Open issues / not verified

- `retire.py` was not run against the real 16 000-clip corpus (absent locally),
  so the actual retired-sentence count is unknown here.
- "Sentences with one of two voices" uses the voices observed in the two files;
  a sentence whose second voice was never attempted shows as *partial*, not
  retired, which is why `partial` is printed separately.
- The builder's CV-only policy means a future request larger than the 31 927
  remaining sentences fails loudly instead of falling back to Wikisource.
- `remote/*` wiring above is a recommendation; those files were left untouched.
- `git fetch origin` was a no-op (origin/main is still `531033c`), so the branch
  needs no rebase.
