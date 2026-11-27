"""Build the eSpeak-derived pronunciation lexicon embedded by the Rust crate.

The Python frontend gets Russian phones from eSpeak NG 1.52 (GPL, offline only).
Shipping eSpeak is not allowed, so this script runs it once over the product's
vocabulary and writes a plain `word<TAB>phones` text. The Rust G2P looks words
up here and falls back to letter rules for anything else.

Keys are the source spellings *and* RUAccent's `ё` spellings (the Python path
phonemises the yo-resolved word), so the Rust side needs no `yo_words` table to
produce the same phones.

Usage (from the worktree root):

    PYTHONPATH=training ../../.cache/venvs/f3/bin/python \
        scripts/gen/build_g2p_lexicon.py \
        --corpus data/scripts-ru/batch1.jsonl data/scripts-ru/batch2.jsonl
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from voca_ru.text.g2p import G2P  # noqa: E402
from voca_ru.text.normalize import normalize  # noqa: E402
from voca_ru.text.stress import Stressor  # noqa: E402

WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)*")


def collect(paths):
    words = set()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                words.update(WORD_RE.findall(normalize(json.loads(line)["text"])))
    return words


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--out", default=str(
        ROOT / "crates/voca-ru-text/src/data/g2p_lexicon.txt"))
    args = parser.parse_args(argv)

    words = collect(args.corpus)
    # The same yo mapping the Python stress stage applies before phonemisation.
    try:
        stressor = Stressor().load()
        yo_words = stressor.yo_words
    except Exception as error:  # dictionary not available: е spellings only
        print(f"warning: no RUAccent yo_words ({error})", file=sys.stderr)
        yo_words = {}

    g2p = G2P()
    stressor = None
    try:
        stressor = Stressor().load()
    except Exception as error:  # no dictionary: bare spellings only
        print(f"warning: no stressor for the lexicon ({error})", file=sys.stderr)
    entries = {}
    for source in sorted(words):
        # Python resolves yo per hyphen part (stress.py splits on "-").
        yo = "-".join(yo_words.get(part, part) for part in source.split("-"))
        # Phonemise the acute-marked form, exactly like the corpus pipeline
        # (pipeline.analyze: correct stress -> phonemize).  Without the marks
        # eSpeak falls back to the stress in its own dictionary, which is wrong
        # for words like "берег" (it reads the verb "берёг").
        marked = yo
        if stressor is not None:
            marked = "-".join(
                stressor.stressed(part)[0] for part in yo.split("-"))
        phones = g2p.word(marked)
        rendered = " ".join(
            f"{symbol}:{level}" if level else symbol for symbol, level in phones)
        entries[source] = rendered
        if yo != source:
            entries.setdefault(yo, rendered)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for word in sorted(entries):
            handle.write(f"{word}\t{entries[word]}\n")
    size = out.stat().st_size
    print(f"entries={len(entries)} words={len(words)} bytes={size} "
          f"path={out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
