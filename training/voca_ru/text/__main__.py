"""CLI: python -m voca_ru.text --in file.jsonl --out file.jsonl

Each input row must have a ``text`` field; the output row keeps every input
field and adds normalisation, stress, phonemes, stress levels, token ids and
an ``oov`` flag. Run from the repo root with ``PYTHONPATH=training``.
"""

import argparse
import json
import sys

from .pipeline import analyze, default_stressor
from .stress import Stressor


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m voca_ru.text")
    parser.add_argument("--in", dest="input", required=True, metavar="FILE")
    parser.add_argument("--out", dest="output", required=True, metavar="FILE")
    parser.add_argument("--limit", type=int, default=0, help="first N rows")
    parser.add_argument("--no-ruaccent", action="store_true",
                        help="skip RUAccent; rule fallback only (tests)")
    args = parser.parse_args(argv)

    stressor = Stressor() if args.no_ruaccent else default_stressor()

    total = tokens = oov_rows = 0
    prefix = []
    with open(args.input, encoding="utf-8") as source, \
            open(args.output, "w", encoding="utf-8") as target:
        for line in source:
            line = line.strip()
            if not line:
                continue
            if args.limit and total >= args.limit:
                break
            row = json.loads(line)
            result = analyze(row["text"], stressor=stressor)
            row.update(result)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            tokens += len(result["tokens"])
            oov_rows += int(result["oov"])
            prefix.append(len(result["tokens"]))
    rate = (oov_rows / total) if total else 0.0
    print(f"sentences={total} tokens={tokens} oov_sentences={oov_rows} "
          f"oov_rate={rate:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
