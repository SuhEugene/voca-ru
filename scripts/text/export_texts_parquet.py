#!/usr/bin/env python3
"""Export the source sentences of the corpus to Parquet for HF.

Every sentence the teacher was ever given, with the split it is frozen into,
the licence of its origin, and how many accepted clips it ended up with (0 or 1
per voice). Rows with 0 clips are the retired ones: their audio never passed
screening, so they are excluded from training but kept here for provenance.

    python scripts/text/export_texts_parquet.py --scripts data/scripts-ru \
        --manifest .cache/models/corpus-ru/corpus-final/manifest.jsonl \
        --out .cache/models/corpus-ru/parquet-texts
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("id", pa.string()), ("text", pa.string()), ("split", pa.string()),
    ("category", pa.string()), ("origin", pa.string()),
    ("source_license", pa.string()), ("source_url", pa.string()),
    ("word_count", pa.int32()), ("char_count", pa.int32()),
    ("batch", pa.int32()), ("clips", pa.int32()),
    ("voices", pa.string()), ("retired", pa.bool_()),
])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    voices = {}
    with open(args.manifest, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            voices.setdefault(row["text_id"], Counter())[row["voice"]] += 1

    rows = []
    for name in ("batch1", "batch2", "batch3"):
        with open(os.path.join(args.scripts, f"{name}.jsonl"), encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                got = voices.get(row["id"], Counter())
                rows.append({
                    "id": row["id"], "text": row["text"], "split": row["split"],
                    "category": row["category"], "origin": row["origin"],
                    "source_license": row["source_license"], "source_url": row["source_url"],
                    "word_count": row.get("word_count"), "char_count": row.get("char_count"),
                    "batch": row.get("batch"), "clips": sum(got.values()),
                    "voices": ",".join(sorted(got)) or None,
                    "retired": not got,
                })

    os.makedirs(args.out, exist_ok=True)
    column = {name: [row[name] for row in rows] for name in SCHEMA.names}
    path = os.path.join(args.out, "texts-0000.parquet")
    pq.write_table(pa.table(column, schema=SCHEMA), path, compression="zstd")
    retired = sum(1 for row in rows if row["retired"])
    print(f"texts-0000.parquet: {len(rows)} rows, {retired} retired, "
          f"{len(rows) - retired} with audio")


if __name__ == "__main__":
    main()
