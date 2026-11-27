"""Build training inputs for the second corpus: a new female voice plus the male
clips we already have.

The new female clips come from a fresh generation run (`split_corpus`), the male
ones from the released corpus. Both are renamed to the model's speaker names so
the checkpoint, the Rust crate and the browser page keep the same `F2`/`M2`
interface.

    python scripts/gen/v2_inputs.py --female data/corpus-v2 --male <corpus-final> \
        --out data/ru-corpus-v2/inputs --female-voice f_storyteller
"""

import argparse
import json
import os
from pathlib import Path

SPEAKERS = {"female": "F2", "male": "M2"}
SPLITS = ("train", "validation", "test")
# scripts carry "val", the released corpus carries "validation"
SPLIT_ALIAS = {"val": "validation", "validation": "validation", "test": "test", "train": "train"}


def rows_from(manifest: Path, voice: str, speaker: str, audio_root: Path, prefix: str) -> list[dict]:
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("voice") != voice:
            continue
        row = {**row, "speaker": speaker, "split": SPLIT_ALIAS[row["split"]]}
        # inputs carry an absolute audio path; rewrite it for the target host
        name = Path(row["audio"]).name
        row["audio"] = str(Path(audio_root).resolve() / "audio" / voice / name)
        rows.append(row)
    print(f"{prefix}: {len(rows)} rows for {voice} -> {speaker}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--female", type=Path, required=True, help="new generation run directory")
    parser.add_argument("--male", type=Path, required=True, help="released corpus with the male clips")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--female-voice", default="f_storyteller")
    parser.add_argument("--male-voice", default="m_neutral")
    parser.add_argument("--male-extra", type=Path,
                        help="regenerated male clips that replace the released ones")
    parser.add_argument("--female-audio", type=Path, help="audio root on the target host")
    parser.add_argument("--male-audio", type=Path)
    args = parser.parse_args()

    # absolute paths: prepare resolves relative audio against its shard directory
    female_audio = Path(args.female_audio or (args.female / "audio").parent).resolve()
    male_audio = Path(args.male_audio or (args.male / "audio").parent).resolve()

    female = rows_from(args.female / "manifest.jsonl", args.female_voice, SPEAKERS["female"],
                       female_audio, "female")
    male = rows_from(args.male / "manifest.jsonl", args.male_voice, SPEAKERS["male"],
                     male_audio, "male")
    if args.male_extra:
        # clips regenerated with context-resolved stress outrank the released ones
        extra = rows_from(args.male_extra / "manifest.jsonl", args.male_voice, SPEAKERS["male"],
                          args.male_extra.resolve(), "male-new")
        replaced = {row["text_id"] for row in extra}
        male = [row for row in male if row["text_id"] not in replaced] + extra
        print(f"male: {len(extra)} regenerated clips replace released ones")

    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = [row for row in female + male if row.get("split") == split]
        if not rows:
            raise SystemExit(f"no rows for split {split!r}")
        text_ids = {row["text_id"] for row in rows}
        female_ids = {row["text_id"] for row in rows if row["speaker"] == SPEAKERS["female"]}
        both = len(female_ids & {row["text_id"] for row in rows if row["speaker"] == SPEAKERS["male"]})
        path = args.out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path.name}: {len(rows)} clips, {len(text_ids)} sentences, {both} with both voices")


if __name__ == "__main__":
    main()
