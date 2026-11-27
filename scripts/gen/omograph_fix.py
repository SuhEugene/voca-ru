"""Find the clips whose stress marks change once homographs are resolved by context,
drop them, and hand their text ids to the next generation wave.

Run on the host, after generation and before screening. Female rows come from the
run's own raw.jsonl; male rows come from the released corpus, which was made with
the old single-reading dictionary.

    python scripts/gen/omograph_fix.py --raw data/corpus-v2 --male /workspace/v1/corpus
"""

import argparse
import json
import os
from pathlib import Path

from voca_ru.text.normalize import normalize
from voca_ru.text.stress import Stressor


def reread(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/corpus-v2"))
    parser.add_argument("--male", type=Path, help="released corpus with m_neutral clips")
    parser.add_argument("--male-ids", type=Path, default=Path("/workspace/v2-male-ids.txt"))
    args = parser.parse_args()

    stressor = Stressor().load()
    raw_path = args.raw / "raw.jsonl"
    rows = reread(raw_path)

    keep, drop = [], []
    for row in rows:
        fresh, _ = stressor.stressed(normalize(row["text"]))
        if fresh == row["stressed_text"]:
            keep.append(row)
        else:
            drop.append(row)

    for row in drop:
        audio = Path(row["audio"])
        if audio.exists():
            audio.unlink()
    raw_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keep),
                        encoding="utf-8")
    text_ids = sorted({row["text_id"] for row in drop})
    (args.raw / "omograph-fix-ids.txt").write_text("\n".join(text_ids) + "\n", encoding="utf-8")
    print(f"raw rows kept {len(keep)}, dropped {len(drop)}, distinct sentences {len(text_ids)}")

    # Source files for the next generation runs: the changed sentences for both
    # voices, and the validation/test sentences the training split never covered.
    scripts = Path("data/scripts-ru")
    wanted = set(text_ids)
    retired = {json.loads(line)["id"] for line in
               (scripts / "retired.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    changed_rows, held_out = [], []
    for name in ("batch1.jsonl", "batch2.jsonl", "batch3.jsonl"):
        for line in (scripts / name).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in retired:
                continue
            if row["id"] in wanted:
                changed_rows.append(row)
            if row.get("split") in ("val", "test"):
                held_out.append(row)
    for name, rows in (("batch-omograph.jsonl", changed_rows), ("batch-valtest.jsonl", held_out)):
        (scripts / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                    encoding="utf-8")
        print(f"{name}: {len(rows)} sentences")

    if args.male:
        manifest = reread(args.male / "manifest.jsonl")
        changed = []
        seen = set()
        for row in manifest:
            if row["voice"] != "m_neutral":
                continue
            fresh, _ = stressor.stressed(normalize(row["text"]))
            if fresh != row["stressed_text"]:
                if row["text_id"] not in seen:
                    seen.add(row["text_id"])
                    changed.append(row["text_id"])
        args.male_ids.write_text("\n".join(changed) + "\n", encoding="utf-8")
        print(f"male clips with changed stress: {len(changed)} sentences -> {args.male_ids}")


if __name__ == "__main__":
    main()
