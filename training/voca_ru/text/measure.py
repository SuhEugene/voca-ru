"""Corpus metrics for F3: token distribution, OOV rate, stress examples.

Usage (repo root, venv with requirements.txt installed):

    PYTHONPATH=training python -m voca_ru.text.measure \
        data/scripts-ru/batch1.jsonl data/scripts-ru/batch2.jsonl
"""

import argparse
import json
import sys

from .g2p import espeak
from .pipeline import analyze
from .stress import Stressor, WORD_RE


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args(argv)

    stressor = Stressor().load()
    counts, oov_sentences, words, oov_words = [], 0, 0, 0
    for path in args.files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                result = analyze(row["text"], stressor=stressor)
                counts.append(len(result["tokens"]))
                oov_sentences += int(result["oov"])
                words += len(WORD_RE.findall(result["plain_text"]))
                oov_words += stressor.stats["model"] + stressor.stats["fallback"]
    counts.sort()

    def quantile(p):
        return counts[int(p * (len(counts) - 1))]

    print(f"sentences {len(counts)}")
    print(f"tokens min {counts[0]} p50 {quantile(.5)} p90 {quantile(.9)} "
          f"p95 {quantile(.95)} p99 {quantile(.99)} max {counts[-1]} "
          f"mean {sum(counts) / len(counts):.2f}")
    print(f"sentences over 512 tokens {sum(1 for c in counts if c > 512)}")
    print(f"oov sentences {oov_sentences} ({oov_sentences / len(counts):.4f})")
    print(f"oov words {oov_words}/{words} ({oov_words / words:.4f})")
    from . import g2p

    dropped = g2p._G2P.dropped if g2p._G2P else set()
    print(f"phones {espeak().version}; dropped units {sorted(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
