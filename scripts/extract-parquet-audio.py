#!/usr/bin/env python3
"""Rebuild the FLAC tree from the parquet shards.

Downloading 19 217 loose files tops out at ~6 files/s on this host (HTTP
round-trip bound, 550 KB/s); the same audio inside 12 parquet shards comes down
at 56 MB/s. This unpacks the bytes back to audio/<voice>/<id>.flac so the
existing prepare pipeline can read them by path.

    python scripts/extract-parquet-audio.py --parquet /workspace/parquet --out /workspace/corpus
"""

from __future__ import annotations

import argparse
import glob
import os

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    written = 0
    for path in sorted(glob.glob(os.path.join(args.parquet, "data", "*.parquet"))):
        table = pq.read_table(path, columns=["audio"])
        for row in table.to_pylist():
            audio = row["audio"]
            target = os.path.join(args.out, audio["path"])
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target) and os.path.getsize(target) == len(audio["bytes"]):
                continue
            with open(target, "wb") as stream:
                stream.write(audio["bytes"])
            written += 1
        print(f"{os.path.basename(path)}: {table.num_rows} rows, total {written}", flush=True)
    print(f"wrote {written} files")


if __name__ == "__main__":
    main()
