"""Build the Russian teacher corpus: text frontend -> Qwen3-TTS -> screening.

    . scripts/cache-env.sh
    PYTHONPATH=training .cache/venvs/r4/bin/python scripts/gen/build_corpus.py \
        --pilot --measure-batch 4,8

Text is normalised, stressed (RUAccent) and phonemised by ``training/voca_ru/text``
(F3) — nothing is reimplemented here.  The teacher is Qwen3-TTS Base plus the
Russian stress LoRA ``qwen3-tts-ru-stress/lora`` loaded with ``peft``.  Every
clip is screened by ``screen.py`` (Whisper WER, duration/rate, level, meter2
stress); one regeneration, then it is dropped with a reason in ``review.jsonl``.

Output (``--out``, default ``data/corpus-ru``; git-ignored)::

    manifest.jsonl          one accepted row per clip
    review.jsonl            dropped clips and why
    audio/<voice>/<id>.flac 24 kHz PCM16, peak 0.95
    rejected/<voice>/<id>.flac  audio kept for the dropped rows (audit)
    run-stats.json          throughput, RAM, batch-size measurement, mark check

Resume: accepted and dropped ids are read back from the two JSONL files and
skipped.  Seeds are derived from the batch's ids, voice and attempt, so a rerun
produces the same batch seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import sys
import time

import numpy as np
import soundfile as sf
import torch

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPTS, "../.."))
TRAINING = os.path.join(REPO, "training")
for path in (TRAINING, SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

from screen import AsrJudge, StressJudge, prepare_audio, screen_clip  # noqa: E402
from voca_ru.text import analyze  # noqa: E402

# The cache is shared by every worktree and lives at the main repo root.
_git = os.popen(f"git -C {REPO} rev-parse --path-format=absolute --git-common-dir").read().strip()
SHARED = os.path.dirname(os.path.abspath(_git)) if _git else REPO
MODELS = os.environ.get("VOCA_MODELS", os.path.join(SHARED, ".cache/models"))
BASE_DIR = os.path.join(MODELS, "qwen3-tts-1.7b")
ADAPTER_DIR = os.path.join(MODELS, "qwen3-tts-ru-stress/lora")
# mps (this laptop) or cuda:<n> (rented GPU); see remote/README.md
TTS_DEVICE = os.environ.get("VOCA_TTS_DEVICE", "mps")
LANG = "Russian"
SR = 24000
VOICES = ("f_neutral", "m_neutral")
PILOT_PLAN = {"number": 4, "date": 4, "money": 3, "name": 3, "question": 3, "prose": 3}


# ---------------------------------------------------------------- data loading

def load_rows(sources):
    rows = []
    seen = set()
    for source in sources:
        path = source if os.path.isabs(source) else os.path.join(REPO, "data/scripts-ru", source)
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                rows.append(row)
    return rows


def spread(items, k):
    """k items evenly across an id-sorted list, deterministic."""
    items = sorted(items, key=lambda r: r["id"])
    if k >= len(items):
        return items
    if k == 1:
        return [items[len(items) // 2]]
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def select_pilot(rows):
    """20 sentences across number/date/money/name/question/prose."""
    usable = [r for r in rows if 20 <= len(r["text"]) <= 140 and len(r["text"].split()) >= 3]
    by_cat = {}
    for row in usable:
        by_cat.setdefault(row.get("category"), []).append(row)
    chosen = []
    for cat, k in PILOT_PLAN.items():
        if cat == "prose":
            prose = by_cat.get("conversation", []) + by_cat.get("reading", [])
            picked = spread(prose, k)
        else:
            picked = spread(by_cat.get(cat, []), k)
        chosen.extend(picked)
    return chosen


def prepare_texts(rows):
    """F3 frontend once per sentence; both voices reuse the result."""
    for row in rows:
        result = analyze(row["text"])
        row["plain_text"] = result["plain_text"]
        row["stressed_text"] = result["stressed_text"]
        row["norm_text"] = result["norm_text"]
        row["phonemes"] = result["phonemes"]
        row["oov"] = result["oov"]
    return rows


def make_batches(rows, size):
    """Length-bucketed batches: the longest clip sets the batch's cost."""
    ordered = sorted(rows, key=lambda r: (len(r["plain_text"]), r["id"]))
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def marked_examples(rows, n=5):
    """Proof that U+0301 survives into the string handed to the teacher."""
    out = []
    for row in sorted(rows, key=lambda r: len(r["stressed_text"]))[:n]:
        out.append({
            "text": row["text"],
            "prompt_text": row["stressed_text"],
            "marks": row["stressed_text"].count("\u0301"),
        })
    return out


# ---------------------------------------------------------------- teacher

def load_teacher(use_lora=True, require_lora=False):
    from qwen_tts import Qwen3TTSModel

    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        BASE_DIR, device_map=TTS_DEVICE, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"[load] Base {time.perf_counter() - t0:.1f} s", flush=True)
    adapter = "none"
    if use_lora:
        try:
            from peft import LoraConfig, inject_adapter_in_model
            from safetensors.torch import load_file

            spec = json.load(open(os.path.join(ADAPTER_DIR, "adapter.json")))
            inject_adapter_in_model(
                LoraConfig(r=spec["r"], lora_alpha=spec["alpha"],
                           target_modules=spec["target_modules"],
                           lora_dropout=0.0, bias="none"),
                model.model.talker.model,
            )
            state = load_file(os.path.join(ADAPTER_DIR, "adapter.safetensors"))
            state = {k.replace(".lora_A.weight", ".lora_A.default.weight")
                      .replace(".lora_B.weight", ".lora_B.default.weight"): v
                     for k, v in state.items()}
            missing, unexpected = model.model.talker.model.load_state_dict(state, strict=False)
            missing = [k for k in missing if "lora" in k]
            if missing or unexpected:
                raise RuntimeError(f"adapter tensors did not match: "
                                   f"{len(missing)} missing, {len(unexpected)} unexpected")
            adapter = os.path.relpath(ADAPTER_DIR, SHARED)
            print(f"[load] RU stress LoRA active (r={spec['r']}, alpha={spec['alpha']}, "
                  f"{len(state)} tensors)", flush=True)
        except Exception as exc:  # loud, never silent
            print("!" * 78, flush=True)
            print(f"!! LoRA FAILED TO LOAD ({type(exc).__name__}: {exc})", flush=True)
            print("!! generating with the BASE model only — stress marks will not be followed",
                  flush=True)
            print("!" * 78, flush=True)
            if require_lora:
                raise SystemExit(
                    "LoRA is mandatory when --require-lora is set; refusing to generate "
                    "with the Base model alone"
                )
            adapter = "none (LoRA failed: %s: %s)" % (type(exc).__name__, exc)
    return model, adapter


def load_prompts(model, voices, ref_dir):
    prompts = {}
    for voice in voices:
        wav = os.path.join(ref_dir, voice + ".wav")
        txt = os.path.join(ref_dir, voice + ".txt")
        if not (os.path.isfile(wav) and os.path.isfile(txt)):
            raise FileNotFoundError(
                f"voice reference missing: {wav} / {txt}\n"
                f"copy out/g1/voices/<name>.{{wav,txt}} into {ref_dir}/"
            )
        t0 = time.perf_counter()
        prompts[voice] = model.create_voice_clone_prompt(
            ref_audio=wav, ref_text=open(txt).read().strip(), x_vector_only_mode=False,
        )
        print(f"[prompt] {voice} {time.perf_counter() - t0:.2f} s", flush=True)
    return prompts


def seed_for(ids, voice, attempt, base):
    payload = "|".join(ids) + f"|{voice}|{attempt}|{base}"
    return int.from_bytes(hashlib.sha1(payload.encode()).digest()[:4], "big") & 0x7FFFFFFF


def generate(model, prompt, texts, seed):
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    wavs, sr = model.generate_voice_clone(
        text=texts, language=[LANG] * len(texts), voice_clone_prompt=prompt,
    )
    return wavs, sr, time.perf_counter() - t0


# ---------------------------------------------------------------- output

def peak_rss_gib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2, 2)


def device_gib():
    """Peak device memory in GiB for the device the teacher runs on."""
    try:
        if TTS_DEVICE.startswith("cuda"):
            return round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        return round(torch.mps.driver_allocated_memory() / 1024**3, 2)
    except Exception:
        return None


def read_ids(path, key="id"):
    ids = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)[key])
                except Exception:
                    pass
    return ids


def clip_id(row, voice):
    return f"{row['id']}_{voice}"


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def handle_clip(row, wav, sr, voice, seed, attempt, out, asr, stress, factory, max_wer):
    """Shape, save, screen one clip; move it into audio/ only if accepted."""
    cid = clip_id(row, voice)
    if sr != SR:
        return {"ok": False, "reject_reasons": [f"sample rate {sr} != {SR}"], "wer": None}
    shaped, stats = prepare_audio(wav, sr)
    rejected = os.path.join(out, "rejected", voice, cid + ".flac")
    os.makedirs(os.path.dirname(rejected), exist_ok=True)
    sf.write(rejected, shaped, SR, subtype="PCM_16")
    try:
        result = screen_clip(
            rejected, text=row["text"], stressed_text=row["stressed_text"],
            stats=stats, asr=asr, stress=stress, max_wer=max_wer,
        )
    except Exception as exc:  # one bad clip must not kill the run
        result = {"ok": False, "reject_reasons": [f"screening error: {type(exc).__name__}: {exc}"],
                  "wer": None, "stress_ok": None, "stress_checked": 0, "stress_errors": 0}
    result.update({"id": cid, "text_id": row["id"], "voice": voice, "seed": seed,
                   "attempt": attempt, "synth_seconds": round(factory["seconds"], 3)})
    return result


def manifest_row(row, result, voice, adapter, audio_rel):
    """One accepted row; only the caller's audio path differs between modes."""
    return {
        "id": clip_id(row, voice),
        "text_id": row["id"],
        "text": row["text"],
        "norm_text": row["norm_text"],
        "stressed_text": row["stressed_text"],
        "phonemes": row["phonemes"],
        "voice": voice,
        "audio": audio_rel,
        "seconds": result.get("document_seconds"),
        "wer": result.get("wer"),
        "stress_ok": result.get("stress_ok"),
        "seed": result["seed"],
        "attempt": result["attempt"],
        "category": row.get("category"),
        "split": row.get("split"),
        "origin": row.get("origin"),
        "source_license": row.get("source_license"),
        "source_url": row.get("source_url"),
        "duration_ratio": result.get("duration_ratio"),
        "word_rate": result.get("word_rate"),
        "peak": result.get("peak"),
        "rms": result.get("rms"),
        "lead_trim_s": result.get("lead_trim_s"),
        "tail_trim_s": result.get("tail_trim_s"),
        "stress_checked": result.get("stress_checked"),
        "stress_errors": result.get("stress_errors"),
        "stress_error_rate": result.get("stress_error_rate"),
        "asr_transcript": result.get("asr_transcript"),
        "synth_seconds": result.get("synth_seconds"),
        "teacher_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "teacher_adapter": adapter,
        "reference_audio": voice + ".wav",
    }


def accept(row, result, voice, out, adapter):
    cid = clip_id(row, voice)
    src = os.path.join(out, "rejected", voice, cid + ".flac")
    dst_rel = os.path.join("audio", voice, cid + ".flac")
    dst = os.path.join(out, dst_rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    return manifest_row(row, result, voice, adapter, dst_rel)


def review_row(row, result, voice):
    return {
        "id": clip_id(row, voice), "text_id": row["id"], "voice": voice,
        "text": row["text"], "stressed_text": row["stressed_text"],
        "reject_reasons": result.get("reject_reasons", []),
        "wer": result.get("wer"), "stress_ok": result.get("stress_ok"),
        "stress_checked": result.get("stress_checked"),
        "stress_errors": result.get("stress_errors"),
        "duration_ratio": result.get("duration_ratio"),
        "seconds": result.get("document_seconds"),
        "seed": result.get("seed"), "attempt": result.get("attempt"),
        "asr_transcript": result.get("asr_transcript"),
    }


# ---------------------------------------------------------------- measurement

def measure_batches(model, prompt, rows, sizes, base_seed):
    """Generate the same short texts at several batch sizes; RTF per size."""
    sample = sorted(rows, key=lambda r: len(r["plain_text"]))[:max(sizes)]
    measured = []
    for size in sizes:
        texts = [r["stressed_text"] for r in sample[:size]]
        seed = seed_for([r["id"] for r in sample[:size]], "measure", 0, base_seed)
        wavs, sr, wall = generate(model, prompt, texts, seed)
        audio = sum(len(w) for w in wavs) / sr
        measured.append({
            "batch": size, "clips": len(texts), "audio_s": round(audio, 2),
            "wall_s": round(wall, 2), "rtf": round(wall / audio, 3),
            "x_realtime": round(audio / wall, 2),
            "clips_per_min": round(len(texts) / wall * 60, 2),
            "peak_device_gib": device_gib(),
        })
        print("[measure]", json.dumps(measured[-1]), flush=True)
    return measured


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["batch1.jsonl", "batch2.jsonl"])
    ap.add_argument("--out", default=os.path.join(REPO, "data/corpus-ru"))
    ap.add_argument("--voices", default=",".join(VOICES))
    ap.add_argument("--ref-dir", default=os.path.join(REPO, "assets/voices"))
    ap.add_argument("--pilot", action="store_true", help="20 sentences across categories")
    ap.add_argument("--limit", type=int, default=0, help="cap sentences after selection")
    ap.add_argument("--split", default="train",
                    help="'train', 'val', 'test' or 'all' (corpus keeps every split)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--measure-batch", default="", help="e.g. 4,8 — measure then continue")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-wer", type=float, default=0.15,
                    help="screening WER ceiling (negative forces a drop, for testing)")
    ap.add_argument("--asr-device", default=os.environ.get("VOCA_ASR_DEVICE", "mps"))
    ap.add_argument("--asr-compute-type", default="float16")
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--require-lora", action="store_true",
                    help="exit non-zero if the stress adapter cannot be loaded")
    args = ap.parse_args()

    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    manifest = os.path.join(out, "manifest.jsonl")
    review = os.path.join(out, "review.jsonl")
    done = read_ids(manifest) | read_ids(review)

    rows = load_rows(args.sources)
    if args.split and args.split != "all":
        rows = [r for r in rows if r.get("split") == args.split]
    rows = select_pilot(rows) if args.pilot else rows
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("no sentences selected")

    print(f"[data] {len(rows)} sentences, {len(voices)} voices", flush=True)
    t_text = time.perf_counter()
    rows = prepare_texts(rows)
    text_seconds = time.perf_counter() - t_text
    examples = marked_examples(rows)
    assert all(e["marks"] > 0 for e in examples), "stress marks missing from prompt text"
    for e in examples:
        print("[mark]", e["marks"], repr(e["prompt_text"]), flush=True)

    batches = make_batches(rows, args.batch_size)
    print(f"[data] {len(batches)} length-bucketed batches of <= {args.batch_size}; "
          f"F3 frontend {text_seconds:.1f} s", flush=True)

    model, adapter = load_teacher(use_lora=not args.no_lora,
                                  require_lora=args.require_lora)
    prompts = load_prompts(model, voices, args.ref_dir)

    stats = {
        "device": TTS_DEVICE, "dtype": "bfloat16",
        "teacher_model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "teacher_adapter": adapter,
        "sentences": len(rows), "voices": voices,
        "batch_size": args.batch_size, "max_wer": args.max_wer, "pilot": bool(args.pilot),
        "split": args.split, "seed": args.seed,
        "text_seconds": round(text_seconds, 2),
        "mark_examples": examples,
        "measured_batches": [],
    }
    if args.measure_batch:
        sizes = [int(s) for s in args.measure_batch.split(",") if s.strip()]
        stats["measured_batches"] = measure_batches(
            model, prompts[voices[0]], rows, sizes, args.seed)

    asr = None if args.no_screen else AsrJudge(args.asr_device, args.asr_compute_type)
    stress = None if (args.no_screen or args.no_stress) else StressJudge(device="cpu")

    started = time.perf_counter()
    generation_wall = 0.0
    accepted = dropped = 0
    clips = 0
    try:
        for voice in voices:
            prompt = prompts[voice]
            for bi, batch in enumerate(batches):
                pending = [r for r in batch if clip_id(r, voice) not in done]
                if not pending:
                    continue
                ids = [r["id"] for r in batch]
                seed = seed_for(ids, voice, 0, args.seed)
                wavs, sr, wall = generate(model, prompt, [r["stressed_text"] for r in pending], seed)
                generation_wall += wall
                per_clip = wall / max(len(pending), 1)
                print(f"[gen] {voice} batch {bi + 1}/{len(batches)} "
                      f"{len(pending)} clips {wall:.1f}s ({per_clip:.2f}s/clip)", flush=True)

                failed = []
                for row, wav in zip(pending, wavs):
                    res = handle_clip(row, wav, sr, voice, seed, 1, out, asr, stress,
                                      {"seconds": per_clip}, args.max_wer)
                    clips += 1
                    if res["ok"]:
                        append_jsonl(manifest, accept(row, res, voice, out, adapter))
                        accepted += 1
                    else:
                        failed.append(row)
                if failed:
                    retry_ids = [r["id"] for r in failed]
                    retry_seed = seed_for(retry_ids, voice, 1, args.seed)
                    t0 = time.perf_counter()
                    rwavs, sr, rwall = generate(
                        model, prompt, [r["stressed_text"] for r in failed], retry_seed)
                    generation_wall += rwall
                    rpc = rwall / max(len(failed), 1)
                    print(f"[retry] {voice} {len(failed)} clips {rwall:.1f}s", flush=True)
                    for row, wav in zip(failed, rwavs):
                        res = handle_clip(row, wav, sr, voice, retry_seed, 2, out, asr, stress,
                                          {"seconds": rpc}, args.max_wer)
                        clips += 1
                        if res["ok"]:
                            append_jsonl(manifest, accept(row, res, voice, out, adapter))
                            accepted += 1
                        else:
                            append_jsonl(review, review_row(row, res, voice))
                            dropped += 1
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("[stop] interrupted; completed clips are in the manifest", flush=True)

    elapsed = time.perf_counter() - started
    audio_total = 0.0
    wer_values = []
    stress_fail = 0
    if os.path.exists(manifest):
        with open(manifest) as f:
            for line in f:
                row = json.loads(line)
                audio_total += row.get("seconds") or 0.0
                if row.get("wer") is not None:
                    wer_values.append(row["wer"])
                if row.get("stress_ok") is False:
                    stress_fail += 1
    stats.update({
        "clips_generated": clips,
        "clips_accepted": accepted,
        "clips_dropped": dropped,
        "clip_seconds": round(audio_total, 2),
        "generation_wall_s": round(generation_wall, 2),
        "wall_s": round(elapsed, 2),
        "clips_per_min": round(clips / elapsed * 60, 2) if elapsed else None,
        "rtf": round(generation_wall / audio_total, 3) if audio_total else None,
        "x_realtime": round(audio_total / generation_wall, 2) if generation_wall else None,
        "wer_mean": round(float(np.mean(wer_values)), 4) if wer_values else None,
        "wer_max": round(float(np.max(wer_values)), 4) if wer_values else None,
        "stress_failed_clips": stress_fail,
        "peak_rss_gib": peak_rss_gib(),
        "peak_device_gib": device_gib(),
    })
    if clips:
        with open(os.path.join(out, "run-stats.json"), "w") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
    else:
        print("[done] nothing new; run-stats.json left untouched", flush=True)
    print("[done]", json.dumps({k: v for k, v in stats.items()
                                if k not in ("mark_examples", "measured_batches")}),
          flush=True)


if __name__ == "__main__":
    main()
