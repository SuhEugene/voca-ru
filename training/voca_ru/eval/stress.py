"""Stress audit over the 200-item hand-built set in ``data/stress_ru.jsonl``.

What ASR text can and cannot say: Whisper/GigaAM never emit U+0301, so a
transcript cannot reveal where the speaker put the stress.  Two signals are
reported instead, and the report states which one produced a number:

* ``word_match`` — the target word survives transcription at all (a gross
  intelligibility proxy; it catches stress errors only when they destroy the
  word, e.g. stress on the wrong syllable of a rare word).
* ``detector`` — an audio stress detector (``meter2``, Apache-2.0, from the
  Qwen stress-adapter repo) when configured.  Not installed in F5; requesting
  it without the package fails loudly.

The item list (word, correct stress, common wrong stress) is complete and is
the input for the detector once it lands.
"""

from __future__ import annotations

import json
from pathlib import Path

from .scoring import score_text

DEFAULT_SET = Path(__file__).with_name("data") / "stress_ru.jsonl"


def load_stress_set(path: str | Path = DEFAULT_SET) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines() if line.strip()]
    for row in rows:
        missing = {"id", "word", "correct", "wrong"} - row.keys()
        if missing:
            raise ValueError(f"stress item {row.get('id')} misses {sorted(missing)}")
        for key in ("correct", "wrong"):
            if "\u0301" not in row[key] and "ё" not in row[key]:
                raise ValueError(f"stress item {row['id']}: {key} has no stress mark")
    return rows


def evaluate_stress(
    rows: list[dict],
    transcripts: dict[str, str],
    detector=None,
) -> dict:
    """Compare each item's transcript with the expected word.

    ``transcripts`` maps item id -> ASR text.  ``detector(item, audio)`` is
    optional and must return the stress mark position it heard.
    """
    items = []
    word_misses = []
    for row in rows:
        text = transcripts.get(row["id"])
        if text is None:
            continue
        hyp = score_text(text)
        target = score_text(row["word"])
        # Strip the marks from correct/wrong: a transcript only tells spelling.
        matches = target in hyp.split()
        item = {
            "id": row["id"],
            "word": row["word"],
            "transcript": text,
            "word_match": matches,
            "stress_verdict": "untested (transcript has no stress)",
        }
        if detector is not None:
            heard = detector(row)
            item["stress_verdict"] = "ok" if heard == row["correct"] else "error"
            item["heard"] = heard
            if heard != row["correct"]:
                item["expected"] = row["correct"]
                item["common_wrong"] = row["wrong"]
        if not matches:
            word_misses.append({"id": row["id"], "word": row["word"], "asr": text})
        items.append(item)
    tested = [i for i in items if i["stress_verdict"] in ("ok", "error")]
    errors = [i for i in tested if i["stress_verdict"] == "error"]
    return {
        "items": len(rows),
        "with_audio": len(items),
        "word_miss_rate": len(word_misses) / len(items) if items else None,
        "word_misses": word_misses[:50],
        "stress_errors": errors,
        "stress_error_rate": len(errors) / len(tested) if tested else None,
        "detector": getattr(detector, "name", None) if detector is not None else None,
        "note": (
            "ASR transcripts carry no stress marks; stress_error_rate needs an "
            "audio stress detector (meter2)."
        ),
    }
