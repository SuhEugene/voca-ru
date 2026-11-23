"""Health check for a running corpus generation.

Computes ``status.json`` from the corpus on disk and decides whether to alarm.
Pure logic (``build_status``, ``evaluate``) is importable without a GPU; the CLI
reads ``nvidia-smi`` and the filesystem and writes the status/log files.

Alarms (the task contract):
  * no new clip for 20 min
  * clips/min below 50 % of the first-10-minute average
  * stress failures above 2 % of checked words
  * WER mean above 5 %
  * disk free below 10 GB
  * GPU temperature above 85 C

The corpus itself is only ever opened read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NO_CLIP_SECONDS = 20 * 60
RATE_DROP = 0.5
STRESS_FAIL_PCT = 2.0
WER_MEAN = 0.05
DISK_FREE_GB = 10.0
GPU_TEMP_C = 85.0
BASELINE_WINDOW_S = 10 * 60


def _iter_jsonl(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def read_corpus(corpus_dir):
    """Return (accepted rows, rejected rows, newest audio mtime)."""
    rows = list(_iter_jsonl(os.path.join(corpus_dir, "manifest.jsonl")))
    rejected = list(_iter_jsonl(os.path.join(corpus_dir, "review.jsonl")))
    newest = None
    audio_dir = os.path.join(corpus_dir, "audio")
    for root, _dirs, files in os.walk(audio_dir):
        for name in files:
            if not name.endswith(".flac"):
                continue
            mtime = os.path.getmtime(os.path.join(root, name))
            newest = mtime if newest is None else max(newest, mtime)
    return rows, rejected, newest


def expected_clips(script_dir, voices):
    """Every row of every script file, read by every voice."""
    total = 0
    for name in sorted(os.listdir(script_dir)):
        if name.endswith(".jsonl"):
            with open(os.path.join(script_dir, name), encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
    return total * len(voices)


def gpu_stats():
    """(temp_c, vram_used_gb, vram_total_gb) or (None, None, None)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None, None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=temperature.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
        temp, used, total = (part.strip() for part in out.split(","))
        return int(float(temp)), round(float(used) / 1024, 2), round(float(total) / 1024, 2)
    except Exception:
        return None, None, None


def disk_free_gb(path):
    usage = shutil.disk_usage(path)
    return round(usage.free / 1024**3, 2)


def build_status(corpus_dir, expected, now, *, gpu=(None, None, None), free_gb=None,
                 history=None, run_started=None):
    """Status dict for one sampling instant.

    ``history`` is a list of {t, clips_done} samples; ``run_started`` is the
    epoch time the run began (defaults to the first history sample).
    """
    rows, rejected, newest_audio = read_corpus(corpus_dir)
    done = len(rows)
    wer_values = [float(r["wer"]) for r in rows if isinstance(r.get("wer"), (int, float))]
    stress_checked = sum(int(r.get("stress_checked") or 0) for r in rows)
    stress_errors = sum(int(r.get("stress_errors") or 0) for r in rows)

    history = list(history or [])
    if history:
        start = run_started or history[0]["t"]
        elapsed_min = max((now - start) / 60.0, 1e-9)
        clips_per_min = max((done - history[0]["clips_done"]) / elapsed_min, 0.0)
    else:
        start = run_started or now
        elapsed_min = 0.0
        clips_per_min = 0.0
    clips_per_min = round(clips_per_min, 2)

    remaining = max(expected - done, 0)
    eta_hours = round(remaining / clips_per_min / 60.0, 2) if clips_per_min > 0 else None

    temp, vram_used, _vram_total = gpu
    if free_gb is None:
        free_gb = disk_free_gb(corpus_dir)
    return {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
        "elapsed_min": round(elapsed_min, 1),
        "clips_done": done,
        "clips_expected": expected,
        "progress_pct": round(100.0 * done / expected, 2) if expected else None,
        "clips_per_min": clips_per_min,
        "eta_hours": eta_hours,
        "wer_mean": round(sum(wer_values) / len(wer_values), 4) if wer_values else None,
        "wer_p95": round(_percentile(wer_values, 0.95), 4) if wer_values else None,
        "dropped": len(rejected),
        "stress_fail_pct": round(100.0 * stress_errors / stress_checked, 3)
        if stress_checked
        else None,
        "gpu_temp_c": temp,
        "vram_used_gb": vram_used,
        "disk_free_gb": free_gb,
        "last_clip_age_s": round(now - newest_audio, 1) if newest_audio else None,
    }


def baseline_rate(history, run_started=None):
    """Average clips/min over the first 10 minutes of the run."""
    if len(history) < 2:
        return None
    start = run_started or history[0]["t"]
    window = [s for s in history if s["t"] <= start + BASELINE_WINDOW_S]
    if len(window) < 2:
        return None
    minutes = (window[-1]["t"] - window[0]["t"]) / 60.0
    if minutes <= 0:
        return None
    return (window[-1]["clips_done"] - window[0]["clips_done"]) / minutes


def evaluate(status, baseline, *, expected=None, no_clip_seconds=NO_CLIP_SECONDS):
    """List of alarm strings; empty means healthy. Pure."""
    alarms = []
    age = status.get("last_clip_age_s")
    expected = expected if expected is not None else status.get("clips_expected") or 0
    done = status.get("clips_done") or 0
    if age is not None and age > no_clip_seconds and done < expected:
        alarms.append(f"no new clip for {age / 60:.1f} min (limit 20)")
    rate = status.get("clips_per_min")
    if baseline and rate is not None and done < expected and rate < RATE_DROP * baseline:
        alarms.append(
            f"throughput {rate:.1f} clips/min is below 50 % of the "
            f"first-10-min baseline {baseline:.1f}"
        )
    stress = status.get("stress_fail_pct")
    if stress is not None and stress > STRESS_FAIL_PCT:
        alarms.append(f"stress failures {stress:.2f} % > {STRESS_FAIL_PCT} %")
    wer_mean = status.get("wer_mean")
    if wer_mean is not None and wer_mean > WER_MEAN:
        alarms.append(f"WER mean {100 * wer_mean:.2f} % > 5 %")
    free = status.get("disk_free_gb")
    if free is not None and free < DISK_FREE_GB:
        alarms.append(f"disk free {free:.1f} GB < {DISK_FREE_GB} GB")
    temp = status.get("gpu_temp_c")
    if temp is not None and temp > GPU_TEMP_C:
        alarms.append(f"GPU temperature {temp} C > {GPU_TEMP_C} C")
    return alarms


def _load_history(path):
    return list(_iter_jsonl(path))


def _append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def one_sample(corpus_dir, expected, history_path):
    now = time.time()
    history = _load_history(history_path)
    temp, used, _total = gpu_stats()
    status = build_status(corpus_dir, expected, now, gpu=(temp, used, None),
                          history=history)
    _append_jsonl(history_path, {"t": round(now, 1), "clips_done": status["clips_done"]})
    baseline = baseline_rate(history)
    alarms = evaluate(status, baseline)
    return status, baseline, alarms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--scripts-dir", default=None,
                        help="script corpus for the expected total (default: repo data/scripts-ru)")
    parser.add_argument("--expected", type=int, default=0)
    parser.add_argument("--voices", default="f_neutral,m_neutral")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--keep-going", action="store_true",
                        help="in loop mode, keep monitoring after an alarm")
    args = parser.parse_args(argv)

    corpus_dir = os.path.abspath(args.corpus_dir)
    os.makedirs(corpus_dir, exist_ok=True)
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    scripts_dir = args.scripts_dir or os.path.join(root, "data/scripts-ru")
    voices = [v for v in args.voices.split(",") if v.strip()]
    expected = args.expected
    if not expected:
        try:
            expected = expected_clips(scripts_dir, voices)
        except FileNotFoundError:
            expected = 0
    status_path = os.path.join(corpus_dir, "status.json")
    health_path = os.path.join(corpus_dir, "health.log")
    history_path = os.path.join(corpus_dir, "status-history.jsonl")

    def sample():
        status, baseline, alarms = one_sample(corpus_dir, expected, history_path)
        tmp = status_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(status, handle, ensure_ascii=False, indent=1)
        os.replace(tmp, status_path)
        line = (
            f"{status['updated_at']} clips={status['clips_done']}/{expected} "
            f"rate={status['clips_per_min']}/min eta={status['eta_hours']}h "
            f"wer={status['wer_mean']} dropped={status['dropped']} "
            f"stress={status['stress_fail_pct']}% temp={status['gpu_temp_c']}C "
            f"vram={status['vram_used_gb']}GB disk={status['disk_free_gb']}GB "
            f"age={status['last_clip_age_s']}s"
        )
        with open(health_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            for alarm in alarms:
                handle.write(f"ALARM: {alarm}\n")
        print(line, flush=True)
        for alarm in alarms:
            print(f"ALARM: {alarm}", file=sys.stderr, flush=True)
        return alarms

    if args.once:
        alarms = sample()
        return 2 if alarms else 0

    while True:
        alarms = sample()
        if alarms and not args.keep_going:
            print(f"WATCH: stopping on alarm; status kept in {status_path}",
                  file=sys.stderr, flush=True)
            return 2
        time.sleep(max(args.interval, 10))


if __name__ == "__main__":
    sys.exit(main())
