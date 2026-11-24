#!/usr/bin/env python3
"""Retire sentences that produced no usable audio.

A sentence with zero accepted clips in any voice can never be trained on, so it
must not be generated again.  Reads a corpus directory (``manifest.jsonl`` =
accepted clips, ``review.jsonl`` = dropped clips; both rows carry ``text_id``
and ``voice``) and writes one row per retired sentence:

    data/scripts-ru/retired.jsonl   {id, text, voices_rejected, reasons}

``scripts/gen/build_corpus.py::load_rows`` skips these ids silently.

    . scripts/cache-env.sh
    python3 scripts/gen/retire.py --corpus data/corpus-ru
"""

from __future__ import annotations

import argparse
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUT = os.path.join(REPO, "data/scripts-ru/retired.jsonl")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retire(corpus, out=OUT):
    """Write the retired list; return {retired, partial, healthy, voices}."""
    manifest = read_jsonl(os.path.join(corpus, "manifest.jsonl"))
    review = read_jsonl(os.path.join(corpus, "review.jsonl"))
    if not manifest and not review:
        raise SystemExit(f"{corpus}: neither manifest.jsonl nor review.jsonl")
    voices = {row["voice"] for row in manifest + review}

    accepted = {}
    for row in manifest:
        accepted.setdefault(row["text_id"], set()).add(row["voice"])

    rejected = {}
    for row in review:
        entry = rejected.setdefault(row["text_id"],
                                    {"text": row.get("text", ""), "voices": set(),
                                     "reasons": set()})
        entry["voices"].add(row["voice"])
        entry["reasons"].update(row.get("reject_reasons") or [])

    retired = [{"id": text_id,
                "text": entry["text"],
                "voices_rejected": sorted(entry["voices"]),
                "reasons": sorted(entry["reasons"])}
               for text_id, entry in sorted(rejected.items())
               if text_id not in accepted]

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in retired:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    partial = sum(1 for seen in accepted.values() if seen != voices)
    summary = {"retired": len(retired),
               "partial": partial,
               "healthy": sum(1 for seen in accepted.values() if seen == voices),
               "voices": len(voices)}
    print(f"sentences retired: {summary['retired']}")
    print(f"sentences with one of {summary['voices']} voices: {summary['partial']}")
    print(f"sentences fully healthy: {summary['healthy']}")
    print(f"wrote {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=os.path.join(REPO, "data/corpus-ru"),
                    help="directory holding manifest.jsonl and review.jsonl")
    ap.add_argument("--out", default=OUT, help="retired list (data/scripts-ru/retired.jsonl)")
    args = ap.parse_args()
    retire(args.corpus, args.out)


if __name__ == "__main__":
    main()
