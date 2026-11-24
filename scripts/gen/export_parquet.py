#!/usr/bin/env python3
"""Export the voca-ru corpus to Parquet shards for the HF dataset viewer.

The FLAC tree plus manifest.jsonl stays the source of truth. Each shard
embeds the waveform as the ``{bytes, path}`` struct that Hugging Face treats
as an Audio column, so the viewer shows a player next to the text and
``load_dataset`` works without a loading script.

datasets>=4 wants torchcodec just to encode audio, which we do not need: the
files are already FLAC, so pyarrow writes them byte for byte.

    python scripts/gen/export_parquet.py --corpus .cache/models/corpus-ru/corpus-final \
        --out .cache/models/corpus-ru/parquet
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

STRINGS = ("id", "voice", "text", "stressed_text", "category", "split",
           "source_license", "source_url")
AUDIO = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
SCHEMA = pa.schema([(name, pa.string()) for name in STRINGS] +
                   [("seconds", pa.float32()), ("wer", pa.float32()),
                    ("seed", pa.int64()), ("audio", AUDIO)])
SPLITS = {"train": "train", "val": "validation", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-rows", type=int, default=2000)
    args = parser.parse_args()

    rows = defaultdict(list)
    with open(os.path.join(args.corpus, "manifest.jsonl"), encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[SPLITS[row["split"]]].append(row)

    os.makedirs(args.out, exist_ok=True)
    for split, records in rows.items():
        for start in range(0, len(records), args.shard_rows):
            chunk = records[start:start + args.shard_rows]
            column = {name: [] for name in SCHEMA.names}
            for row in chunk:
                full = os.path.join(args.corpus, row["audio"])
                for name in STRINGS:
                    column[name].append(row.get(name))
                column["seconds"].append(float(row["seconds"]))
                column["wer"].append(float(row["wer"]) if row.get("wer") is not None else None)
                column["seed"].append(int(row["seed"]) if row.get("seed") is not None else None)
                with open(full, "rb") as audio:
                    column["audio"].append({"bytes": audio.read(), "path": row["audio"]})
            path = os.path.join(args.out, f"{split}-{start // args.shard_rows:04d}.parquet")
            pq.write_table(pa.Table.from_pydict(column, schema=SCHEMA), path)
            print(json.dumps({"split": split, "rows": len(chunk), "path": path}), flush=True)


if __name__ == "__main__":
    main()
