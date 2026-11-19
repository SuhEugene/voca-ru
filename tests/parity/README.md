# Rust frontend parity (F7)

Token ids are the contract: the Rust crate must produce the same id sequence as
`training/voca_ru/text` on the product sentences. Stress is a parallel feature
and is reported separately.

## Command

```sh
# from the worktree root, with the F3 venv and RUAccent dictionary in .cache
cargo test -p voca-ru-text --test parity -- --ignored --nocapture
```

The test runs `tests/parity/reference.py` (Python reference frontend) over the
first 1000 rows of `data/scripts-ru/batch1.jsonl`, then compares them with the
Rust crate in-process. It fails below 99 % exact sentence match.

Requirements:

- Python interpreter: `$VOCA_RU_PYTHON`, else `.cache/venvs/f3/bin/python`
  (checked in this worktree and in the main checkout).
- RUAccent dictionary: `$VOCA_RUACCENT_WORKDIR` (parent of `dictionary/`), else
  `.cache/models/ruaccent` in the worktree or main checkout.
- Offline alternative: set `VOCA_PARITY_REFERENCE` to a TSV produced by
  `reference.py` and the test skips Python.

`fixture_matches_python` is not ignored: it checks a committed 6-sentence
fixture and runs on every `cargo test`.

## Output

The test prints exact-match %, per-token disagreements, stress agreement, OOV
sentence counts for both sides, and a cause breakdown (normalisation / G2P /
stress-only). See `reports/F7-rust.md` for the measured numbers.
