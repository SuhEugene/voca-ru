"""Build the compact stress dictionary embedded by `crates/voca-ru-text`.

RUAccent's `accents.json.gz` is 3.19 M forms; the runtime crate needs a small
sorted `word<TAB>stressed` text. Stress is written RUAccent-style with `+`
before the stressed vowel, e.g. ``год+у``; the Rust loader turns it into the
combining acute ``году́``.

Selection: forms are ranked by frequency in the optional corpus (the product's
sentence list), then alphabetically, and the first ``--limit`` are kept. Without
a corpus the alphabetical order is used. ``ё`` spellings are resolved here the
same way `turn stress.py` resolves them (yo_words first), so the Rust side needs
no second dictionary.

Usage (from the worktree root):

    PYTHONPATH=training ../../.cache/venvs/f3/bin/python \
        scripts/gen/build_stress_dict.py --limit 100000 \
        --corpus data/scripts-ru/batch1.jsonl data/scripts-ru/batch2.jsonl
"""

import argparse
import collections
import gzip
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from voca_ru.text.normalize import normalize  # noqa: E402

WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)*")
CONTEXT_OVERRIDES = {"году", "года"}  # hardcoded in Rust stress.rs


def plus_to_accent(text):
    return re.sub(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])", lambda m: m.group(1) + "\u0301", text)


def load(directory, name):
    path = directory / f"{name}.json.gz"
    if not path.exists():
        return {}
    with gzip.open(path) as handle:
        return json.load(handle)


def corpus_frequencies(paths):
    counts = collections.Counter()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                for word in WORD_RE.findall(normalize(json.loads(line)["text"])):
                    counts[word] += 1
    return counts


def context_choices(paths):
    """word -> reading the omograph model picked most often in the corpus.

    RUAccent stores one reading per form in ``accents.json``; for a homograph the
    second reading only exists in ``omographs.json``.  A flat runtime dictionary
    cannot look at the sentence, so it takes the reading the corpus supports most
    often instead of silently keeping variants[0].
    """
    if not paths:
        return {}
    from voca_ru.text.stress import Stressor
    from voca_ru.text.normalize import normalize

    stressor = Stressor().load()
    tally = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                text = normalize(json.loads(line)["text"])
                words = text.split()
                for index, variant in stressor._resolve_omographs(text).items():
                    word = re.sub(r"[^а-яёa-z-]", "", words[index].lower())
                    bucket = tally.setdefault(word, collections.Counter())
                    bucket[variant] += 1
    return {word: counts.most_common(1)[0][0] for word, counts in tally.items()}


def build(accents, omographs, yo_words, freq, chosen=None):
    """source spelling -> RUAccent-style stressed string."""
    chosen = chosen or {}
    entries = {}
    for word, marked in accents.items():
        entries[word] = plus_to_accent(marked)
    for word, variants in omographs.items():
        if word in yo_words:
            continue  # Python looks up the yo spelling instead
        # a homograph decided by context wins over the single stored reading
        entries[word] = plus_to_accent(chosen.get(word, variants[0]))
    # Source spellings written with е but meant with ё.
    for source, yo in yo_words.items():
        if source in CONTEXT_OVERRIDES:
            continue
        marked = accents.get(yo)
        if marked is None:
            variants = omographs.get(yo)
            marked = variants[0] if variants else None
        if marked is not None:
            entries[source] = plus_to_accent(marked)
    return {w: v for w, v in entries.items() if w not in CONTEXT_OVERRIDES}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", default=None,
                        help="directory with accents.json.gz")
    parser.add_argument("--corpus", nargs="*", default=[])
    parser.add_argument("--context-corpus", nargs="*", default=[],
                        help="sentences that decide homograph readings")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--out", default=str(
        ROOT / "crates/voca-ru-text/src/data/stress_dict.txt"))
    args = parser.parse_args(argv)

    if args.dictionary:
        directory = pathlib.Path(args.dictionary)
    else:
        base = os.environ.get("VOCA_RUACCENT_WORKDIR")
        if base:
            directory = pathlib.Path(base)
        else:
            cache = os.environ.get("VOCA_CACHE")
            if cache:
                directory = pathlib.Path(cache) / "models/ruaccent/dictionary"
            else:
                directory = ROOT.parents[1] / ".cache/models/ruaccent/dictionary"

    accents = load(directory, "accents")
    omographs = load(directory, "omographs")
    yo_words = load(directory, "yo_words")
    freq = corpus_frequencies(args.corpus) if args.corpus else collections.Counter()
    chosen = context_choices(args.context_corpus)
    entries = build(accents, omographs, yo_words, freq, chosen)
    print(f"homographs decided by context: {len(chosen)}", file=sys.stderr)

    # Rank source spellings by corpus frequency, aggregating forms that map to
    # the same yo spelling.
    by_key = collections.Counter()
    for word, count in freq.items():
        by_key[word] += count
        yo = yo_words.get(word)
        if yo:
            by_key[yo] += count
    ranked = sorted(entries, key=lambda w: (-by_key.get(w, 0), w))
    selected = ranked[:args.limit]
    selected.sort()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for word in selected:
            handle.write(f"{word}\t{entries[word]}\n")
    size = out.stat().st_size
    print(f"entries={len(selected)} of {len(entries)} bytes={size} "
          f"path={out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
