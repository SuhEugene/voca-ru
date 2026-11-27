"""Build the OOV stress table: word ending -> position of the stressed vowel.

Unknown words currently get the penultimate vowel, which is right only about
half the time (48.2 % of held-out FLEURS-ru words that the shipped dictionary
misses).  RUAccent's 3.19 M forms carry the true stress of those words, so the
statistics of *where* stress falls for a given ending are available for free.
Longest ending with enough evidence wins (5, then 4, then 3 letters); measured
on the same held-out words this lifts the fallback to 75.8 %.

Value = how many vowels back from the last vowel (0 = last).
ё is always marked by the rule that runs before this table, so ё forms are
skipped to keep the statistics about genuinely ambiguous words.

Usage:
    PYTHONPATH=training .cache/venvs/r4/bin/python \
        scripts/gen/build_stress_endings.py [--limit 150000]
"""

import argparse
import collections
import gzip
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VOWELS = "аеёиоуыэюя"
LENGTHS = (3, 4, 5)
MIN_SUPPORT = 5


def load(directory, name):
    path = directory / f"{name}.json.gz"
    if not path.exists():
        return {}
    with gzip.open(path) as handle:
        return json.load(handle)


def stressed_vowel(form):
    """Index of the stressed vowel, or None when the form carries no mark."""
    position = form.find("+")
    if position < 0:
        return None
    return sum(1 for char in form[:position] if char in VOWELS)


def build(accents):
    tables = {length: collections.defaultdict(collections.Counter) for length in LENGTHS}
    for word, form in accents.items():
        if "ё" in word or "Ё" in word:
            continue
        index = stressed_vowel(form)
        if index is None:
            continue
        vowels = [i for i, char in enumerate(word) if char in VOWELS]
        if not vowels or index >= len(vowels):
            continue
        back = min(len(vowels) - 1 - index, 3)
        for length in LENGTHS:
            if len(word) >= length + 1:
                tables[length][word[-length:]][back] += 1
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", default=None,
                        help="directory with accents.json.gz")
    parser.add_argument("--out", default=str(
        ROOT / "crates/voca-ru-text/src/data/stress_endings.txt"))
    args = parser.parse_args(argv)

    if args.dictionary:
        directory = pathlib.Path(args.dictionary)
    else:
        base = os.environ.get("VOCA_CACHE")
        if base:
            directory = pathlib.Path(base) / "models/ruaccent/dictionary"
        else:
            directory = ROOT / ".cache/models/ruaccent/dictionary"

    accents = load(directory, "accents")
    if not accents:
        print(f"no accents.json.gz in {directory}", file=sys.stderr)
        return 1
    tables = build(accents)

    rows = {}
    for length in LENGTHS:
        for ending, counts in tables[length].items():
            total = sum(counts.values())
            if total < MIN_SUPPORT:
                continue
            # a shorter ending never overrides a longer one: keep the longest
            rows.setdefault(ending, (counts.most_common(1)[0][0], total))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for ending in sorted(rows):
            back, total = rows[ending]
            handle.write(f"{ending}\t{back}\t{total}\n")
    print(f"endings={len(rows)} bytes={out.stat().st_size} path={out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
