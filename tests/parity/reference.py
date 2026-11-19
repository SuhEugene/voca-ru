"""Python half of the F7 parity harness.

Runs the reference frontend (`training/voca_ru/text`) over the first N rows of
a batch file and writes TSV for `tests/parity.rs`:

    id <TAB> tokens <TAB> plain_text <TAB> stressed_text <TAB> phonemes <TAB> oov

Usage (the Rust test calls this):

    PYTHONPATH=training .cache/venvs/f3/bin/python tests/parity/reference.py \
        --batch data/scripts-ru/batch1.jsonl --limit 1000 --out /tmp/ref.tsv
"""

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))


def default_dictionary():
    for base in (ROOT, ROOT.parent.parent):
        candidate = base / ".cache/models/ruaccent/dictionary"
        if candidate.is_dir():
            return candidate
    return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    dictionary = default_dictionary()
    if dictionary and not os.environ.get("VOCA_RUACCENT_WORKDIR"):
        os.environ["VOCA_RUACCENT_WORKDIR"] = str(dictionary.parent)

    from voca_ru.text.pipeline import analyze, default_stressor

    stressor = default_stressor()
    written = 0
    with open(args.batch, encoding="utf-8") as source, \
            open(args.out, "w", encoding="utf-8") as target:
        for line in source:
            if written >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            result = analyze(row["text"], stressor=stressor)
            fields = [
                row["id"],
                " ".join(str(token) for token in result["tokens"]),
                result["plain_text"],
                result["stressed_text"],
                " ".join(result["phonemes"]),
                "1" if result["oov"] else "0",
            ]
            target.write("\t".join(fields) + "\n")
            written += 1
    print(f"wrote {written} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
