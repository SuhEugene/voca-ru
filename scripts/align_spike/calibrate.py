"""F6 spike: measure CTC alignment of eSpeak-ru phones on real Russian audio.

    . scripts/cache-env.sh
    .cache/venvs/r4/bin/python scripts/align_spike/calibrate.py --clips 150

Reads FLEURS ru test clips already extracted into
`.cache/models/fleurs-ru/audio/test`, phonemizes the reference transcript with
eSpeak 1.52 `ru`, aligns, and writes a JSON summary plus per-phone calibration
to `.cache/align-spike/`. Prints the numbers that go into reports/F6-align.md.
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import align
import espeak_ru
import mapping

REPO = Path(__file__).resolve().parents[2]
# The shared cache lives next to HF_HOME, which scripts/cache-env.sh sets.
CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")).parent
DEFAULT_MODEL = CACHE / "models/wav2vec2-lv-60-espeak-cv-ft"
DEFAULT_FLEURS = CACHE / "models/fleurs-ru"
OUT_DIR = CACHE / "align-spike"


def load_clips(fleurs_dir, limit):
    rows = list(csv.reader(open(fleurs_dir / "test.tsv", encoding="utf-8"), delimiter="\t"))
    clips, seen = [], set()
    for row in rows:
        name, text = row[1], row[2]
        path = fleurs_dir / "audio/test" / name
        if path.exists() and name not in seen:
            seen.add(name)
            clips.append({"id": name[:-4], "text": text, "path": path})
        if len(clips) >= limit:
            break
    return clips


def load_silero(limit):
    """Secondary set: the Russian TTS clips already in out/silero (test only).

    Synthetic audio is the training domain, so it checks the aligner where it
    will actually run; FLEURS checks it on human speech. Digits in these
    transcripts were read differently by the TTS, so they are expected to fail.
    """
    directory = CACHE.parent / "out/silero"
    transcripts = json.loads((directory / "transcripts-whisper-small.json").read_text(encoding="utf-8"))
    clips = [{"id": name[:-4], "text": text, "path": directory / name}
             for name, text in transcripts.items() if (directory / name).exists()]
    return clips[:limit]


def percentile(values, q):
    return round(float(np.percentile(values, q)), 4) if values else None


# Polish port's calibrated thresholds (docs/ports.md §3); used here only to
# report an accept rate, not as a decision.
GATE = {"token_probability_mean": 0.45, "weak_phone_fraction": 0.35, "vowel_voiced": 0.45}


def acceptance(results):
    accepted, rejected = 0, Counter()
    for result in results:
        reasons = []
        if result["qc"]["token_probability_mean"] < GATE["token_probability_mean"]:
            reasons.append("low_confidence")
        if result["qc"]["weak_phone_fraction"] > GATE["weak_phone_fraction"]:
            reasons.append("weak_alignment")
        if result["qc"]["vowel_voiced"] is not None and result["qc"]["vowel_voiced"] < GATE["vowel_voiced"]:
            reasons.append("vowels_not_voiced")
        if reasons:
            rejected.update(reasons)
        else:
            accepted += 1
    return {"thresholds": GATE, "accepted": accepted, "attempted": len(results),
            "rejected_reasons": dict(rejected)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=150)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fleurs", type=Path, default=DEFAULT_FLEURS)
    parser.add_argument("--tag", default="run", help="output file suffix")
    parser.add_argument("--silero", action="store_true", help="align out/silero clips instead of FLEURS")
    args = parser.parse_args()

    clips = load_silero(args.clips) if args.silero else load_clips(args.fleurs, args.clips)
    print(f"{len(clips)} clips with audio", flush=True)

    # 1. eSpeak-ru inventory over every transcript we have (200+ sentences).
    tsv_texts = [row[2] for row in csv.reader(open(args.fleurs / "test.tsv", encoding="utf-8"), delimiter="\t")][:200]
    t0 = time.perf_counter()
    inventory_sentences = espeak_ru.phonemize_sentences(tsv_texts)
    inventory_seconds = time.perf_counter() - t0
    inventory = Counter(p for words in inventory_sentences for w in words for p, _ in w)
    print(f"inventory: {len(inventory)} types over {len(tsv_texts)} sentences "
          f"({inventory_seconds:.1f}s)", flush=True)

    info = align.load(args.model, args.device)
    unknown = sorted(p for p in inventory if p not in info.vocab and p not in mapping.FALLBACK and p not in mapping.CANONICAL)
    print("phones without a label:", unknown, flush=True)

    results, failures = [], []
    wall = time.perf_counter()
    for index, clip in enumerate(clips):
        try:
            wave, sr = __import__("soundfile").read(str(clip["path"]), dtype="float32")
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            words = espeak_ru.phonemize(clip["text"])
            result = align.align_phones(info, words, wave, sr)
        except Exception as error:  # noqa: BLE001 - a failed clip is data too
            failures.append({"id": clip["id"], "error": f"{type(error).__name__}: {error}"})
            continue
        result.update({"id": clip["id"], "text": clip["text"]})
        results.append(result)
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(clips)} clips, {time.perf_counter() - wall:.0f}s", flush=True)
    elapsed = time.perf_counter() - wall

    qc_keys = ["token_probability_mean", "weak_phone_fraction", "one_frame_phone_fraction",
               "max_phone_frames", "phones_per_second", "token_coverage", "vowel_voiced",
               "voiceless_voiced"]
    qc = {key: [r["qc"][key] for r in results if r["qc"][key] is not None] for key in qc_keys}
    summary = {
        "clips_attempted": len(clips),
        "clips_aligned": len(results),
        "clips_failed": len(failures),
        "failures": failures,
        "wall_seconds": round(elapsed, 1),
        "seconds_per_clip": round(elapsed / max(len(clips), 1), 2),
        "inference_seconds_per_clip": round(statistics.mean(r["inference_seconds"] for r in results), 3) if results else None,
        "audio_seconds": round(sum(r["seconds"] for r in results), 1),
        "real_time_factor": round(elapsed / max(sum(r["seconds"] for r in results), 1e-6), 3),
        "qc": {key: {"mean": round(statistics.mean(v), 4), "p10": percentile(v, 10),
                     "p50": percentile(v, 50), "p90": percentile(v, 90),
                     "min": round(min(v), 4), "max": round(max(v), 4)} for key, v in qc.items() if v},
        "stressed_vowels": sum(r["qc"]["stressed_vowels"] for r in results),
        "stressed_short": sum(r["qc"]["stressed_short"] for r in results),
        "stressed_unvoiced": sum(r["qc"]["stressed_unvoiced"] for r in results),
        "unknown_phones": unknown,
        "inventory": dict(inventory.most_common()),
        "inventory_seconds": round(inventory_seconds, 1),
    }

    # Per-phone duration and probability, plus the calibration evidence.
    per_phone = defaultdict(lambda: {"count": 0, "frames": [], "prob": [], "candidates": defaultdict(list),
                                     "top": Counter(), "scores": defaultdict(list), "kinds": Counter()})
    for result in results:
        for phone, frames, prob, item in zip(result["phones"], result["durations_frames"],
                                             result["probability"], result["calibration"]):
            entry = per_phone[phone]
            entry["count"] += 1
            entry["frames"].append(frames)
            entry["prob"].append(prob)
            entry["kinds"][item["kind"]] += 1
            entry["top"][item["top"][0][0]] += 1
            for name, value in item["top"]:
                entry["scores"][name].append(value)
            for name, value in item["candidates"].items():
                entry["candidates"][name].append(value)
    phone_table = {}
    for phone, entry in sorted(per_phone.items(), key=lambda kv: -kv[1]["count"]):
        phone_table[phone] = {
            "count": entry["count"],
            "frames_mean": round(statistics.mean(entry["frames"]), 2),
            "frames_p10": percentile(entry["frames"], 10),
            "frames_max": int(max(entry["frames"])),
            "prob_mean": round(statistics.mean(entry["prob"]), 4),
            "chosen": info.label(phone)[0] if phone in info.vocab or phone in mapping.FALLBACK or phone in mapping.CANONICAL else None,
            "kind": info.label(phone)[1] if phone in info.vocab or phone in mapping.FALLBACK or phone in mapping.CANONICAL else None,
            "candidates": {name: round(statistics.mean(v), 4) for name, v in entry["candidates"].items()},
            "top_label": entry["top"].most_common(3),
            "top_scores": {name: round(statistics.mean(v), 3)
                           for name, v in sorted(entry["scores"].items(), key=lambda kv: -statistics.mean(kv[1]))[:4]},
            "kinds": dict(entry["kinds"]),
        }
    summary["phones"] = phone_table
    summary["gate"] = acceptance(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"alignment-{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1))
    print(f"wrote {out}", flush=True)
    print(f"aligned {len(results)}/{len(clips)}  {summary['seconds_per_clip']}s/clip  "
          f"RTF {summary['real_time_factor']}", flush=True)
    for key, stats in summary["qc"].items():
        print(f"  {key:28} mean {stats['mean']:>8}  p10 {stats['p10']:>8}  p50 {stats['p50']:>8}  p90 {stats['p90']:>8}", flush=True)
    print(f"  stressed vowels {summary['stressed_vowels']}  short(<2 frames) {summary['stressed_short']}  "
          f"unvoiced {summary['stressed_unvoiced']}", flush=True)
    gate = summary["gate"]
    print(f"  gate (Polish thresholds): {gate['accepted']}/{gate['attempted']} accepted  "
          f"reasons {gate['rejected_reasons']}", flush=True)
    print("  worst phones by mean prob:", flush=True)
    for phone, stats in sorted(phone_table.items(), key=lambda kv: kv[1]["prob_mean"])[:12]:
        print(f"    {phone!r:10} n={stats['count']:4} frames={stats['frames_mean']:6} "
              f"prob={stats['prob_mean']:.3f} chosen={stats['chosen']!r} top={stats['top_label']}", flush=True)


if __name__ == "__main__":
    main()
