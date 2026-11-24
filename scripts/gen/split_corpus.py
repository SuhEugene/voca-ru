"""Two-phase corpus build: generate a wave, screen it later, repeat for failures.

Generation and screening used to share one loop, so the GPU idled for the whole
screening pass and the batch size was capped by memory shared with Whisper.
This module splits the same pieces into separate runs::

    # 1. generate as many clips as the GPU takes in one batch
    split_corpus.py --generate-only --batch-size 16 ...  -> raw.jsonl + audio/
    # 2. screen what was generated (no teacher loaded)
    split_corpus.py --screen-only ...                    -> manifest.jsonl / review.jsonl
    # 3. take the rejects and generate a second wave
    split_corpus.py --wave-report --ids-file data/wave2.ids -> histogram + id list
    split_corpus.py --generate-only --wave 2 --ids-file data/wave2.ids ...

Everything is the sequential path's own code: the same F3 frontend, teacher,
LoRA, ``make_batches``, ``seed_for``, ``prepare_audio``, ``accept``/``review_row``
and ``screen.screen_clip``.  Wave 1 therefore generates byte-identical FLACs to
the screened run for the same sentence set and ``--batch-size``.

Run it directly (it reuses ``build_corpus.py``'s helpers, so the sequential
script does not need new flags):

    PYTHONPATH=training .cache/venvs/r4/bin/python scripts/gen/split_corpus.py \
        --generate-only --batch-size 16 --out data/corpus-ru

Output (``--out``, default ``data/corpus-ru``)::

    raw.jsonl               one row per generated clip and wave
    audio/<voice>/<id>.flac trimmed, peak-normalised FLAC
    manifest.jsonl          accepted clips (unchanged schema)
    review.jsonl            rejected clips and why (unchanged schema)
    rejected/<voice>/<id>.flac audio of a rejected clip (audit)
    screen-stats.json       last screening report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import soundfile as sf

import build_corpus as bc
import screen

SR = bc.SR
# Fields copied from the sentence row into raw.jsonl so screening can rebuild the
# exact manifest row later without re-running the text frontend.
PROVENANCE = ("category", "split", "origin", "source_license", "source_url")
# Stats remembered so screen_clip sees the same numbers the sequential path did.
STATS = ("seconds", "peak", "rms", "raw_peak", "raw_rms", "raw_clipped",
         "lead_trim_s", "tail_trim_s", "gain")


# ---------------------------------------------------------------- jsonl state

def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def dedupe_jsonl(path):
    """Last row per id wins; rewrite the file only if it had duplicates.

    A second wave screens an id that is already in review.jsonl, so a kill can
    leave two review rows for it.  Keeping the last one matches "latest wave is
    the verdict" and mirrors ``pipeline.repair_outputs``.
    """
    rows = read_jsonl(path)
    keep = {}
    for row in rows:
        if row.get("id") is not None:
            keep[row["id"]] = row
    if os.path.exists(path) and len(keep) != len(rows):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for row in keep.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    return keep


def latest_raw(path):
    """Latest (wave, attempt) row per clip id, in raw.jsonl order."""
    latest = {}
    for row in read_jsonl(path):
        cid = row.get("id")
        cur = latest.get(cid)
        key = (row.get("wave", 1), row.get("attempt", 1))
        if cur is None or key >= (cur.get("wave", 1), cur.get("attempt", 1)):
            latest[cid] = row
    return latest


def paths(out):
    return {
        "raw": os.path.join(out, "raw.jsonl"),
        "manifest": os.path.join(out, "manifest.jsonl"),
        "review": os.path.join(out, "review.jsonl"),
    }


# ---------------------------------------------------------------- generation

def raw_row(row, voice, audio_rel, seed, wave, stats, synth_seconds,
            teacher_adapter):
    out = {
        "id": bc.clip_id(row, voice),
        "text_id": row["id"],
        "voice": voice,
        "text": row["text"],
        "norm_text": row["norm_text"],
        "stressed_text": row["stressed_text"],
        "phonemes": row["phonemes"],
        "tokens": row.get("tokens"),
        "audio": audio_rel,
        "seconds": stats["seconds"],
        "seed": seed,
        "attempt": wave,
        "wave": wave,
        "synth_seconds": round(synth_seconds, 3),
        "teacher_adapter": teacher_adapter,
    }
    out.update({k: stats.get(k) for k in STATS if k != "seconds"})
    out.update({k: row.get(k) for k in PROVENANCE})
    return out


def _works(rows, voices, batch_size, wave, ids, done):
    """Generation units: whole length-bucketed batches in wave 1, single clips after.

    Wave 1 keeps ``make_batches`` and the whole-batch seed, so its audio matches
    the screened run.  A later wave regenerates exactly the ids it is given, one
    clip per teacher call with the sequential retry seed
    (``seed_for([text_id], voice, wave - 1, seed)``).
    """
    work = []
    if wave == 1:
        for voice in voices:
            for batch in bc.make_batches(rows, batch_size):
                pending = [r for r in batch if bc.clip_id(r, voice) not in done]
                if pending:
                    work.append({"voice": voice, "rows": pending,
                                 "ids": [r["id"] for r in batch]})
        return work

    want = set(ids or ())
    for row in rows:
        for voice in voices:
            cid = bc.clip_id(row, voice)
            if cid not in want:
                continue
            prev = done.get(cid)
            if prev is not None and prev.get("wave", 1) >= wave:
                continue
            work.append({"voice": voice, "rows": [row], "ids": [row["id"]]})
    missing = want - {bc.clip_id(i["rows"][0], i["voice"]) for i in work} - set(done)
    for cid in sorted(missing):
        print(f"[gen] id {cid} not in the selected sentences; skipped", flush=True)
    return work


def _reset_device_peak():
    """``device_gib`` reports a running maximum; restart it per batch."""
    if bc.TTS_DEVICE.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()


def run_generate(*, rows, voices, batch_size, base_seed, out, teacher,
                 wave=1, ids=None, adapter="unknown"):
    """Generate one wave. No Whisper, no meter2, no screening."""
    if wave > 1 and ids is None:
        raise ValueError("wave > 1 needs --ids-file")
    os.makedirs(out, exist_ok=True)
    p = paths(out)
    done = latest_raw(p["raw"])
    work = _works(rows, voices, batch_size, wave, ids, done)
    clips = 0
    wall_total = 0.0
    started = time.perf_counter()
    for bi, item in enumerate(work, 1):
        voice = item["voice"]
        seed = bc.seed_for(item["ids"], voice, wave - 1, base_seed)
        _reset_device_peak()
        wavs, sr, wall = teacher.generate(
            voice, [r["stressed_text"] for r in item["rows"]], seed)
        wall_total += wall
        per_clip = wall / max(len(wavs), 1)
        if len(wavs) != len(item["rows"]):
            # The teacher can return fewer waveforms than prompts (2026-11-24:
            # 643 clips lost on the longest male sentences). zip() would drop
            # them silently, so name every lost id for the wave-2 ids file.
            lost = [bc.clip_id(r, voice) for r in item["rows"][len(wavs):]]
            print(f"[gen] batch {bi}: {len(wavs)} wavs for {len(item['rows'])} rows; "
                  f"lost {len(lost)}: {' '.join(lost)}", flush=True)
            with open(p["raw"] + ".lost", "a") as fh:
                fh.write("".join(cid + "\n" for cid in lost))
        for row, wav in zip(item["rows"], wavs):
            cid = bc.clip_id(row, voice)
            if sr != SR:
                print(f"[gen] {cid} sample rate {sr} != {SR}; skipped", flush=True)
                continue
            shaped, stats = screen.prepare_audio(wav, sr)
            rel = os.path.join("audio", voice, cid + ".flac")
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sf.write(dst, shaped, SR, subtype="PCM_16")
            bc.append_jsonl(p["raw"], raw_row(row, voice, rel, seed, wave, stats,
                                              per_clip, adapter))
            clips += 1
        peak = bc.device_gib()
        print(f"[gen] wave {wave} batch {bi}/{len(work)} {len(item['rows'])} clips "
              f"{wall:.1f}s ({per_clip:.2f}s/clip) peak_device_gib {peak}", flush=True)

    elapsed = time.perf_counter() - started
    summary = {"wave": wave, "generated": clips, "batches": len(work),
               "wall_s": round(elapsed, 2),
               "clips_per_min": round(clips / elapsed * 60, 2) if elapsed else None,
               "generation_wall_s": round(wall_total, 2),
               "peak_device_gib": bc.device_gib(),
               "peak_rss_gib": bc.peak_rss_gib()}
    print("[generate]", json.dumps(summary), flush=True)
    return summary


# ---------------------------------------------------------------- screening

def reason_key(reason):
    """Bucket a reject reason: numbers replaced so the histogram groups them."""
    return re.sub(r"[-+]?\d+(?:\.\d+)?", "N", reason)


def _screen_stats(row):
    stats = {k: row.get(k) for k in STATS}
    stats["seconds"] = row["seconds"]
    stats["raw_clipped"] = bool(row.get("raw_clipped"))
    stats["rms"] = row.get("rms") or 0.0
    return stats


def _pending_rows(out, voices):
    p = paths(out)
    rows = read_jsonl(p["raw"])
    latest = {}
    for row in rows:
        latest[row.get("id")] = row  # append order: last row is the latest wave
    accepted = dedupe_jsonl(p["manifest"])
    reviewed = dedupe_jsonl(p["review"])
    pending = []
    for row in rows:
        cid = row.get("id")
        if latest.get(cid) is not row:
            continue
        if voices and row.get("voice") not in voices:
            continue
        if cid in accepted:
            continue
        prev = reviewed.get(cid)
        if prev is not None and prev.get("attempt", 1) >= row.get("wave", 1):
            continue
        pending.append(row)
    return pending, reviewed


def _drop_rows(path, ids):
    """Remove stale review rows for clips that are being screened again."""
    rows = [r for r in read_jsonl(path) if r.get("id") not in ids]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def sentence_row(raw):
    """raw.jsonl row -> the sentence-shaped row ``manifest_row`` expects."""
    row = {"id": raw["text_id"], "text": raw["text"],
           "norm_text": raw.get("norm_text"),
           "stressed_text": raw["stressed_text"],
           "phonemes": raw.get("phonemes")}
    row.update({k: raw.get(k) for k in PROVENANCE})
    return row


def run_screen(*, out, asr, stress, max_wer=screen.MAX_WER, voices=None,
               batch=32):
    """Screen every raw clip that has no verdict at its current wave."""
    p = paths(out)
    os.makedirs(out, exist_ok=True)
    pending, reviewed = _pending_rows(out, voices)
    stale = {row["id"] for row in pending} & set(reviewed)
    if stale:
        _drop_rows(p["review"], stale)
    print(f"[screen] {len(pending)} clips to screen", flush=True)
    started = time.perf_counter()
    screened = accepted = rejected = 0
    reasons = Counter()
    wers = []
    stress_checked = stress_failed = 0

    for start in range(0, len(pending), max(batch, 1)):
        chunk = pending[start:start + max(batch, 1)]
        for row in chunk:
            cid = row["id"]
            voice = row["voice"]
            path = os.path.join(out, row["audio"])
            try:
                result = screen.screen_clip(
                    path, text=row["text"], stressed_text=row["stressed_text"],
                    stats=_screen_stats(row), asr=asr, stress=stress,
                    max_wer=max_wer,
                )
            except Exception as exc:  # one bad clip must not kill the run
                result = {"ok": False, "wer": None, "stress_ok": None,
                          "stress_checked": 0, "stress_errors": 0,
                          "reject_reasons": [f"screening error: "
                                             f"{type(exc).__name__}: {exc}"]}
            result.update({"id": cid, "text_id": row.get("text_id"),
                           "voice": voice, "seed": row.get("seed"),
                           "attempt": row.get("wave", 1),
                           "synth_seconds": row.get("synth_seconds")})
            screened += 1
            if result.get("wer") is not None:
                wers.append(result["wer"])
            if result.get("stress_checked"):
                stress_checked += 1
                if result.get("stress_ok") is False:
                    stress_failed += 1
            if result["ok"]:
                bc.append_jsonl(p["manifest"], bc.manifest_row(
                    sentence_row(row), result, voice,
                    row.get("teacher_adapter"), row["audio"]))
                accepted += 1
            else:
                # A rejected clip leaves audio/ for rejected/, as handle_clip does.
                rejected_path = os.path.join(out, "rejected", voice, cid + ".flac")
                os.makedirs(os.path.dirname(rejected_path), exist_ok=True)
                if os.path.exists(path):
                    os.replace(path, rejected_path)
                bc.append_jsonl(p["review"],
                                bc.review_row(sentence_row(row), result, voice))
                rejected += 1
                if result["reject_reasons"]:
                    reasons[reason_key(result["reject_reasons"][0])] += 1
        print(f"[screen] {min(start + len(chunk), len(pending))}/{len(pending)}", flush=True)

    elapsed = time.perf_counter() - started
    summary = {
        "screened": screened, "accepted": accepted, "rejected": rejected,
        "reject_reasons": dict(reasons.most_common()),
        "wer_mean": round(sum(wers) / len(wers), 4) if wers else None,
        "wer_p95": round(sorted(wers)[min(len(wers) - 1, int(0.95 * len(wers)))], 4)
        if wers else None,
        "stress_fail_pct": round(100 * stress_failed / stress_checked, 2)
        if stress_checked else None,
        "clips_per_min": round(screened / elapsed * 60, 2) if elapsed else None,
        "wall_s": round(elapsed, 2),
    }
    with open(os.path.join(out, "screen-stats.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("[screen]", json.dumps(summary), flush=True)
    return summary


# ---------------------------------------------------------------- wave loop

def run_wave_report(out, ids_file=None):
    """Histogram of review.jsonl and the id list for the next generation wave."""
    p = paths(out)
    reviewed = dedupe_jsonl(p["review"])
    counts = Counter()
    for row in reviewed.values():
        reasons = row.get("reject_reasons") or []
        counts[reason_key(reasons[0]) if reasons else "unknown (no reason)"] += 1
    latest = latest_raw(p["raw"])
    next_wave = max([r.get("wave", 1) for r in latest.values()] or [1]) + 1
    ids_file = ids_file or os.path.join(out, f"wave{next_wave}.ids")
    os.makedirs(os.path.dirname(os.path.abspath(ids_file)), exist_ok=True)
    with open(ids_file, "w") as f:
        for cid in sorted(reviewed):
            f.write(cid + "\n")
    summary = {"rejected": len(reviewed), "next_wave": next_wave,
               "ids_file": ids_file, "reject_reasons": dict(counts.most_common())}
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
    if not reviewed:
        print("[wave] no rejects; the ids file is empty", flush=True)
    return summary


def read_ids_file(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------- cli

def _add_shared(ap):
    ap.add_argument("--out", default=os.path.join(bc.REPO, "data/corpus-ru"))
    ap.add_argument("--voices", default=",".join(bc.VOICES))


def _selected_rows(args, voices, ids=None):
    rows = bc.load_rows(args.sources)
    if args.split and args.split != "all":
        rows = [r for r in rows if r.get("split") == args.split]
    rows = bc.select_pilot(rows) if args.pilot else rows
    if ids:
        want = set(ids)
        rows = [r for r in rows
                if any(bc.clip_id(r, voice) in want for voice in voices)]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("no sentences selected")
    return bc.prepare_texts(rows)


def generate_main(args):
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    ids = read_ids_file(args.ids_file) if args.ids_file else None
    if args.wave > 1 and ids is None:
        sys.exit("--wave > 1 needs --ids-file")
    if ids == []:
        print(f"[gen] wave {args.wave}: the ids file is empty; nothing to generate",
              flush=True)
        return {"wave": args.wave, "generated": 0}
    rows = _selected_rows(args, voices, ids)
    print(f"[data] {len(rows)} sentences, {len(voices)} voices, wave {args.wave}",
          flush=True)
    model, adapter = bc.load_teacher(use_lora=not args.no_lora,
                                     require_lora=args.require_lora)
    prompts = bc.load_prompts(model, voices, args.ref_dir)
    return run_generate(rows=rows, voices=voices, batch_size=args.batch_size,
                        base_seed=args.seed, out=os.path.abspath(args.out),
                        teacher=_Teacher(model, prompts),
                        wave=args.wave, ids=ids, adapter=adapter)


class _Teacher:
    """The sequential path's generate call, one method."""

    def __init__(self, model, prompts):
        self.model = model
        self.prompts = prompts

    def generate(self, voice, texts, seed):
        return bc.generate(self.model, self.prompts[voice], texts, seed)


def screen_main(args):
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    out = os.path.abspath(args.out)
    asr = screen.AsrJudge(args.asr_device, args.asr_compute_type)
    stress = None if args.no_stress else screen.StressJudge(device=args.stress_device)
    return run_screen(out=out, asr=asr, stress=stress, max_wer=args.max_wer,
                      voices=voices or None, batch=args.screen_batch)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--screen-only", action="store_true")
    mode.add_argument("--wave-report", action="store_true")
    _add_shared(ap)
    ap.add_argument("--sources", nargs="+", default=["batch1.jsonl", "batch2.jsonl"])
    ap.add_argument("--ref-dir", default=os.path.join(bc.REPO, "assets/voices"))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--wave", type=int, default=1)
    ap.add_argument("--ids-file", default="")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--require-lora", action="store_true")
    ap.add_argument("--max-wer", type=float, default=screen.MAX_WER)
    ap.add_argument("--asr-device", default=os.environ.get("VOCA_ASR_DEVICE", "mps"))
    ap.add_argument("--asr-compute-type", default="float16")
    ap.add_argument("--stress-device", default="cpu")
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--screen-batch", type=int, default=32,
                    help="clips read/progress-reported per chunk (models run per clip)")
    args = ap.parse_args(argv)
    if args.generate_only:
        generate_main(args)
    elif args.screen_only:
        screen_main(args)
    else:
        run_wave_report(os.path.abspath(args.out), args.ids_file or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
