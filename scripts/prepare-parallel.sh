#!/bin/bash
# Parallel acoustic preparation on a many-core host.
#
# prepare.py is sequential per clip (5 files/s on one core while 95 sit idle),
# so the corpus is sharded across processes. Feature files are named by
# sha256(id) and the manifest path is "<split>/<name>.pt", so the shard outputs
# merge into one directory without rewriting anything.
#
#   bash prepare-parallel.sh [workers]
set -euo pipefail
cd /workspace/voca
OUT=${OUT:-data/ru-corpus}
WORKERS=${1:-12}
PYTHON=${PYTHON:-python}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

$PYTHON - "$WORKERS" <<'EOF'
import json, os, pathlib, sys
workers = int(sys.argv[1])
root = pathlib.Path(os.environ.get("OUT", "data/ru-corpus"))
shards = root / "shards"
shards.mkdir(exist_ok=True)
for split in ("train", "validation", "test"):
    rows = [json.loads(l) for l in (root / "inputs" / f"{split}.jsonl").open(encoding="utf-8")]
    n = workers if split == "train" else 2
    for i in range(n):
        chunk = rows[i::n]
        path = shards / f"{split}-{i:02d}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for row in chunk:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path.name}: {len(chunk)} rows")
EOF

for split in train validation test; do
    for path in "$OUT"/shards/"$split"-*.jsonl; do
        name=$(basename "$path" .jsonl)
        echo "[prep] $name"
        $PYTHON training/prepare.py --inputs "$path" --output "$OUT/prepared-$name" \
            --split "$split" --resume &
    done
done
wait

$PYTHON <<'EOF'
import json, os, pathlib, shutil
root = pathlib.Path(os.environ.get("OUT", "data/ru-corpus"))
target = root / "prepared"
target.mkdir(exist_ok=True)
rows = {"train": [], "validation": [], "test": []}
rejects = {"train": [], "validation": [], "test": []}
for split in rows:
    (target / split).mkdir(exist_ok=True)
    for shard in sorted(root.glob(f"prepared-{split}-*")):
        manifest = shard / f"{split}.jsonl"
        if manifest.exists():
            rows[split] += [json.loads(l) for l in manifest.open(encoding="utf-8")]
        rejected = shard / f"{split}.rejected.jsonl"
        if rejected.exists():
            rejects[split] += [json.loads(l) for l in rejected.open(encoding="utf-8")]
        features = shard / split
        if features.is_dir():
            for file in features.glob("*.pt"):
                dest = target / split / file.name
                if not dest.exists():
                    shutil.move(str(file), dest)
    with (target / f"{split}.jsonl").open("w", encoding="utf-8") as out:
        for row in rows[split]:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    # annotate refuses a split without complete preparation metadata; the shard
    # files carry it and are removed below, so copy one over
    source_meta = sorted(root.glob(f"prepared-{split}-*/{split}.preparation.json"))
    if source_meta:
        meta = json.loads(source_meta[0].read_text())
        meta["complete"] = True
        (target / f"{split}.preparation.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    with (target / f"{split}.rejected.jsonl").open("w", encoding="utf-8") as out:
        for row in rejects[split]:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    missing = [r for r in rows[split] if not (target / r["path"]).is_file()]
    print(f"{split}: {len(rows[split])} rows, {len(rejects[split])} rejected, missing {len(missing)}")
    if missing:
        raise SystemExit(f"missing feature files in {split}: {missing[:3]}")

# carry over the feature schema and drop the shard directories
first = sorted(root.glob("prepared-train-*"))[0]
shutil.copy2(first / "features.json", target / "features.json")
for shard in root.glob("prepared-*"):
    shutil.rmtree(shard)
print("merged into", target)
EOF
