#!/usr/bin/env python3
"""Export the forced-alignment layer to Parquet shards for the HF dataset.

One row per accepted clip: the phoneme sequence the aligner was given, the
frame durations it returned, and the QC numbers that decided acceptance. The
acoustic tensors are not part of it -- `prepare.py` recomputes those from the
FLAC.

    python scripts/gen/export_annotations_parquet.py \
        --annotations data/ru-corpus/annotations --corpus .cache/models/corpus-ru/corpus-final \
        --out .cache/models/corpus-ru/parquet-annotations
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("id", pa.string()), ("split", pa.string()), ("speaker", pa.string()),
    ("text", pa.string()), ("frames", pa.int32()), ("symbols", pa.string()),
    ("tokens", pa.list_(pa.int32())), ("stress", pa.list_(pa.int8())),
    ("durations", pa.list_(pa.int32())), ("seconds", pa.float32()),
    ("token_probability_mean", pa.float32()), ("weak_phone_fraction", pa.float32()),
    ("phones_per_second", pa.float32()), ("vowel_voiced", pa.float32()),
    ("origin", pa.string()), ("source_license", pa.string()),
    ("source_url", pa.string()), ("teacher_model", pa.string()),
])
SPLITS = {"train": "train", "validation": "validation", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-rows", type=int, default=2000)
    args = parser.parse_args()

    seconds = {}
    with open(os.path.join(args.corpus, "manifest.jsonl"), encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            seconds[row["id"]] = float(row["seconds"])

    rows = defaultdict(list)
    for name in SPLITS:
        path = os.path.join(args.annotations, f"{name}.jsonl")
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                rows[name].append(json.loads(line))

    os.makedirs(args.out, exist_ok=True)
    for split, records in rows.items():
        for start in range(0, len(records), args.shard_rows):
            chunk = records[start:start + args.shard_rows]
            column = {name: [] for name in SCHEMA.names}
            for row in chunk:
                qc, prov = row.get("qc", {}), row.get("provenance", {})
                column["id"].append(row["id"])
                column["split"].append(split)
                column["speaker"].append(row.get("speaker"))
                column["text"].append(row.get("text"))
                column["frames"].append(row.get("frames"))
                symbols = row["symbols"]
                # the journal stores symbols as one space-joined string
                column["symbols"].append(symbols if isinstance(symbols, str) else " ".join(symbols))
                column["tokens"].append(row["tokens"])
                column["stress"].append(row["stress"])
                column["durations"].append(row["durations"])
                column["seconds"].append(seconds.get(row["id"]))
                column["token_probability_mean"].append(qc.get("token_probability_mean"))
                column["weak_phone_fraction"].append(qc.get("weak_phone_fraction"))
                column["phones_per_second"].append(qc.get("phones_per_second"))
                column["vowel_voiced"].append(qc.get("vowel_voiced"))
                column["origin"].append(prov.get("origin"))
                column["source_license"].append(prov.get("source_license"))
                column["source_url"].append(prov.get("source_url"))
                column["teacher_model"].append(prov.get("teacher_model"))
            name = f"{split}-{start // args.shard_rows:04d}.parquet"
            pq.write_table(pa.table(column, schema=SCHEMA), os.path.join(args.out, name),
                           compression="zstd")
            print(f"{name}: {len(chunk)} rows")


if __name__ == "__main__":
    main()
