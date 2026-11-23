"""Staged corpus build: GPU generation -> screening workers -> single writer.

``build_corpus.py`` is one loop that generates a batch, screens every clip, then
generates the next batch; the GPU idles for the whole screening pass.  This
module wires the same pieces into three concurrent stages connected by bounded
queues, so generation never waits for Whisper/meter2 and the screeners never
wait for the teacher::

    sentences -> [A: teacher, 1 thread] --raw wav--> [B: screen, N threads]
                                                              |
                        retry queue (attempt 2) <-- fail, attempt 1
                                                              |
                                              [C: writer, 1 thread] -> manifest/review

Rules kept from the sequential path:

* the same text frontend, teacher, LoRA, seeds, screening thresholds and
  manifest/review schema (``build_corpus`` helpers are reused, not copied);
* exactly one retry, the retried clip is generated as a single-item batch with
  ``seed_for([text_id], voice, 1, seed)``, then it goes to ``review.jsonl``;
* resume is driven by ``manifest.jsonl`` + ``review.jsonl``; a torn last line
  from a kill is truncated and duplicate ids are collapsed before the run, so a
  crash can neither lose a completed row nor duplicate one;
* Stage C is the only writer, so manifest/review appends never interleave.

Run the real thing through ``build_corpus.py --pipeline``.  The fake benchmark
below proves the overlap without a model::

    python scripts/gen/pipeline.py --benchmark-fake
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import sys
import threading
import time

import numpy as np
import soundfile as sf

import build_corpus as bc

SR = bc.SR
SENTINEL = object()
GPU_IDLE_MAX = 0.25      # benchmark assertion: Stage A idle < 25 % of wall
SPEEDUP_MIN = 1.3        # benchmark assertion: pipelined clips/min vs sequential


# ------------------------------------------------------------------ resume

def repair_jsonl(path):
    """Read a JSONL file, dropping a torn final line left by a kill.

    Appends are line-at-a-time, so only the last line can be truncated.  It is
    removed from the file as well, otherwise the next append would be glued to
    the fragment and both rows would be lost.
    """
    if not os.path.exists(path):
        return []
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            f.seek(cut)
            f.truncate()
            data = data[:cut]
    rows = []
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def repair_outputs(manifest, review):
    """Done ids from both files, rewriting them if a line is torn or duplicated.

    A clip id in both files is a bug (one terminal row only); the manifest row
    wins because it is the one training reads.
    """
    done = set()
    for path in (manifest, review):
        rows = repair_jsonl(path)
        keep, seen = [], set()
        for row in rows:
            cid = row.get("id")
            if cid is None or cid in done or cid in seen:
                continue
            seen.add(cid)
            keep.append(row)
        if os.path.exists(path) and len(keep) != len(rows):
            tmp = path + ".repair"
            with open(tmp, "w") as f:
                for row in keep:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        done |= seen
    return done


# ------------------------------------------------------------------ backends

class Teacher:
    """Real teacher: Qwen3-TTS + LoRA prompt, same call as the sequential path."""

    def __init__(self, model, prompts):
        self.model = model
        self.prompts = prompts

    def generate(self, voice, texts, seed):
        return bc.generate(self.model, self.prompts[voice], texts, seed)


class Screener:
    """Real screening, unchanged: ``build_corpus.handle_clip`` -> ``screen.screen_clip``."""

    def __init__(self, out, asr, stress, max_wer, enabled=True):
        self.out = out
        self.asr = asr
        self.stress = stress
        self.max_wer = max_wer
        self.enabled = enabled

    def screen(self, row, wav, sr, voice, seed, attempt, synth_seconds):
        if self.enabled:
            return bc.handle_clip(row, wav, sr, voice, seed, attempt, self.out,
                                  self.asr, self.stress,
                                  {"seconds": synth_seconds}, self.max_wer)
        # --no-screen: shape and keep the audio, skip WER/stress entirely.
        cid = bc.clip_id(row, voice)
        shaped, stats = bc.prepare_audio(wav, sr)
        rejected = os.path.join(self.out, "rejected", voice, cid + ".flac")
        os.makedirs(os.path.dirname(rejected), exist_ok=True)
        sf.write(rejected, shaped, SR, subtype="PCM_16")
        return {"ok": True, "reject_reasons": [], "id": cid, "seed": seed,
                "attempt": attempt, "wer": None, "stress_ok": None,
                "document_seconds": stats["seconds"], "peak": stats["peak"],
                "rms": stats["rms"], "duration_ratio": None,
                "synth_seconds": round(synth_seconds, 3)}


# ------------------------------------------------------------------ stages

class _InFlight:
    """Clips emitted but not yet terminal (manifest or review row written).

    A retry re-emits the same clip without adding a second count, so the number
    is clips, not generation calls.
    """

    def __init__(self):
        self._n = 0
        self._cv = threading.Condition()
        self._last = time.monotonic()

    def add(self):
        with self._cv:
            self._n += 1
            self._last = time.monotonic()

    def done(self):
        with self._cv:
            self._n -= 1
            self._last = time.monotonic()
            if self._n <= 0:
                self._cv.notify_all()

    def count(self):
        with self._cv:
            return self._n

    def idle_for(self):
        with self._cv:
            return time.monotonic() - self._last


def run_pipelined(*, rows, voices, batch_size, base_seed, out, teacher, screener,
                  adapter, queue_size=32, screen_workers=2, done=None, observer=None):
    """Run the three stages until every clip has one terminal row.

    ``teacher.generate(voice, texts, seed) -> (wavs, sr, seconds)``
    ``screener.screen(row, wav, sr, voice, seed, attempt, synth) -> result``

    Returns the observer dict: counts, ``gpu_gen_s``, ``wall_s``,
    ``gpu_idle_pct``, ``max_jobs_q``, ``gen_order`` and any ``errors``.
    """
    obs = observer if observer is not None else {}
    if screen_workers < 1 or queue_size < 1:
        raise ValueError("screen_workers and queue_size must be >= 1")
    obs.update(clips=0, accepted=0, dropped=0, gen_order=[], max_jobs_q=0,
               gpu_gen_s=0.0, wall_s=0.0, gpu_idle_pct=0.0, errors=[])
    lock = threading.Lock()
    stop = threading.Event()
    inflight = _InFlight()

    manifest = os.path.join(out, "manifest.jsonl")
    review = os.path.join(out, "review.jsonl")
    done = repair_outputs(manifest, review) if done is None else set(done)
    work_dir = os.path.join(out, ".work")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    # Same batching as the sequential path, so an untouched batch keeps its seed.
    work = []
    for voice in voices:
        for batch in bc.make_batches(rows, batch_size):
            pending = [r for r in batch if bc.clip_id(r, voice) not in done]
            if pending:
                work.append({"voice": voice, "ids": [r["id"] for r in batch],
                             "rows": pending})

    jobs = queue.Queue(maxsize=queue_size)
    retries = queue.Queue()
    writes = queue.Queue(maxsize=queue_size)

    def put(q, item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def get(q):
        while not stop.is_set():
            try:
                return q.get(timeout=0.1)
            except queue.Empty:
                continue
        return SENTINEL

    def emit(row, voice, wav, sr, seed, attempt, synth):
        cid = bc.clip_id(row, voice)
        raw = os.path.join(work_dir, f"{cid}-a{attempt}.wav")
        sf.write(raw, wav, sr)
        if attempt == 1:  # a retry re-emits the clip that is already in flight
            inflight.add()
        with lock:
            obs["gen_order"].append(cid)
        put(jobs, {"cid": cid, "row": row, "voice": voice, "seed": seed,
                   "attempt": attempt, "synth": synth, "raw": raw})
        with lock:
            obs["max_jobs_q"] = max(obs["max_jobs_q"], jobs.qsize())

    def generate_batch(item):
        seed = bc.seed_for(item["ids"], item["voice"], 0, base_seed)
        texts = [r["stressed_text"] for r in item["rows"]]
        wavs, sr, wall = teacher.generate(item["voice"], texts, seed)
        with lock:
            obs["gpu_gen_s"] += wall
        per_clip = wall / max(len(wavs), 1)
        print(f"[gen] {item['voice']} {len(wavs)} clips {wall:.1f}s "
              f"({per_clip:.2f}s/clip)", flush=True)
        for row, wav in zip(item["rows"], wavs):
            emit(row, item["voice"], wav, sr, seed, 1, per_clip)

    def generate_retry(job):
        seed = bc.seed_for([job["row"]["id"]], job["voice"], 1, base_seed)
        wavs, sr, wall = teacher.generate(job["voice"], [job["row"]["stressed_text"]], seed)
        with lock:
            obs["gpu_gen_s"] += wall
        print(f"[retry] {job['voice']} 1 clip {wall:.1f}s", flush=True)
        emit(job["row"], job["voice"], wavs[0], sr, seed, 2, wall)

    def stage_a():
        idx = 0
        try:
            while not stop.is_set():
                try:
                    job = retries.get_nowait()
                except queue.Empty:
                    job = None
                if job is not None:
                    generate_retry(job)
                    continue
                if idx < len(work):
                    item = work[idx]
                    idx += 1
                    generate_batch(item)
                    continue
                if inflight.count() == 0:
                    break
                try:
                    job = retries.get(timeout=0.05)
                except queue.Empty:
                    # Lost-job guard: every clip is counted until its row is written.
                    if inflight.idle_for() > 600:
                        raise RuntimeError(
                            f"pipeline stalled: {inflight.count()} clip(s) in flight, "
                            f"nothing screened for 600 s")
                    continue
                generate_retry(job)
        except Exception as exc:  # a dead generator must stop the run, not hang it
            with lock:
                obs["errors"].append(f"stage A: {type(exc).__name__}: {exc}")
            stop.set()
        finally:
            for _ in range(screen_workers):
                put(jobs, SENTINEL)

    def stage_b():
        try:
            while True:
                job = get(jobs)
                if job is SENTINEL:
                    break
                try:
                    wav, sr = sf.read(job["raw"], dtype="float32")
                    result = screener.screen(job["row"], wav, sr, job["voice"],
                                             job["seed"], job["attempt"], job["synth"])
                except Exception as exc:  # one bad clip must not kill the run
                    result = {"ok": False, "id": job["cid"], "seed": job["seed"],
                              "attempt": job["attempt"], "wer": None,
                              "reject_reasons": [f"screening error: "
                                                 f"{type(exc).__name__}: {exc}"]}
                finally:
                    try:
                        os.unlink(job["raw"])
                    except OSError:
                        pass
                if result["ok"] or job["attempt"] >= 2:
                    put(writes, (job, result))
                else:
                    retries.put({"row": job["row"], "voice": job["voice"]})
        except Exception as exc:
            with lock:
                obs["errors"].append(f"stage B: {type(exc).__name__}: {exc}")
            stop.set()

    def stage_c():
        try:
            while True:
                item = get(writes)
                if item is SENTINEL:
                    break
                job, result = item
                if result["ok"]:
                    bc.append_jsonl(manifest, bc.accept(job["row"], result,
                                                       job["voice"], out, adapter))
                    key = "accepted"
                else:
                    bc.append_jsonl(review, bc.review_row(job["row"], result, job["voice"]))
                    key = "dropped"
                with lock:
                    obs[key] += 1
                    obs["clips"] += 1
                inflight.done()
        except Exception as exc:
            with lock:
                obs["errors"].append(f"stage C: {type(exc).__name__}: {exc}")
            stop.set()

    screen_threads = [threading.Thread(target=stage_b, name=f"screen{i}", daemon=True)
                      for i in range(screen_workers)]
    writer = threading.Thread(target=stage_c, name="writer", daemon=True)
    for t in screen_threads:
        t.start()
    writer.start()
    started = time.perf_counter()
    gen = threading.Thread(target=stage_a, name="gen", daemon=True)
    gen.start()
    try:
        gen.join()
        for t in screen_threads:
            t.join()
        put(writes, SENTINEL)
        writer.join()
    except KeyboardInterrupt:
        stop.set()
        for t in [gen, *screen_threads, writer]:
            t.join(timeout=10)
        print("[stop] interrupted; completed clips are in the manifest", flush=True)

    obs["wall_s"] = round(time.perf_counter() - started, 3)
    if obs["wall_s"]:
        obs["gpu_idle_pct"] = round(100 * (1 - obs["gpu_gen_s"] / obs["wall_s"]), 1)
    shutil.rmtree(work_dir, ignore_errors=True)
    if obs["errors"]:
        raise RuntimeError("; ".join(obs["errors"]))
    return obs


# ------------------------------------------------------------------ fake benchmark

class GpuBusy:
    """A shared GPU: one exclusive holder at a time, and it records busy time.

    TTS and Whisper on one card cannot run kernels at the same instant, so a
    benchmark that let their sleeps overlap would flatter the pipeline.  The
    lock serialises them; ``busy`` then gives the honest GPU duty cycle.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._mu = threading.Lock()
        self._busy = 0.0

    def hold(self, seconds):
        with self._lock:
            t = time.perf_counter()
            time.sleep(seconds)
            with self._mu:
                self._busy += time.perf_counter() - t

    @property
    def busy(self):
        with self._mu:
            return self._busy


class FakeTeacher:
    """Synthetic teacher: sleeps ``gen_s`` per clip, returns a short tone.

    Pass ``gpu`` (a ``GpuBusy``) to model the real card, where TTS and Whisper
    kernels serialise on the same device instead of overlapping for free.
    """

    def __init__(self, gen_s=0.0, seconds_per_clip=1.6, gpu=None):
        self.gen_s = gen_s
        self.seconds_per_clip = seconds_per_clip
        self.gpu = gpu
        self.calls = []
        self.times = []

    def _work(self, seconds):
        if self.gpu is None:
            time.sleep(seconds)
        else:
            self.gpu.hold(seconds)

    def generate(self, voice, texts, seed):
        t0 = time.perf_counter()
        self._work(self.gen_s * len(texts))
        rng = np.random.default_rng(seed)
        n = int(self.seconds_per_clip * SR)
        tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)
        wavs = [(tone + 0.01 * rng.standard_normal(n)).astype(np.float32)
                for _ in texts]
        wall = time.perf_counter() - t0
        self.calls.append((voice, len(texts)))
        self.times.append((t0, time.perf_counter()))
        return wavs, SR, wall


class FakeScreener:
    """Synthetic screener: sleeps Whisper (GPU) then meter2 (CPU) per clip.

    ``fail`` holds ``(cid, attempt)`` pairs that are rejected, so tests can
    force the retry -> review path without a model.
    """

    def __init__(self, out, whisper_s=0.0, meter_s=0.0, fail=(), gpu=None):
        self.out = out
        self.whisper_s = whisper_s
        self.meter_s = meter_s
        self.fail = set(fail)
        self.gpu = gpu
        self.calls = []
        self.times = []

    def screen(self, row, wav, sr, voice, seed, attempt, synth_seconds):
        t0 = time.perf_counter()
        if self.gpu is None:
            time.sleep(self.whisper_s)
        else:
            self.gpu.hold(self.whisper_s)
        time.sleep(self.meter_s)
        cid = bc.clip_id(row, voice)
        shaped, stats = bc.prepare_audio(wav, sr)
        rejected = os.path.join(self.out, "rejected", voice, cid + ".flac")
        os.makedirs(os.path.dirname(rejected), exist_ok=True)
        sf.write(rejected, shaped, SR, subtype="PCM_16")
        ok = (cid, attempt) not in self.fail
        self.calls.append((cid, attempt))
        self.times.append((t0, time.perf_counter()))
        return {"ok": ok, "reject_reasons": [] if ok else ["fake reject"],
                "id": cid, "seed": seed, "attempt": attempt, "wer": 0.0,
                "stress_ok": True, "document_seconds": stats["seconds"],
                "duration_ratio": 1.0, "peak": stats["peak"], "rms": stats["rms"],
                "synth_seconds": round(synth_seconds, 3)}


def bench_rows(n):
    rows = []
    for i in range(n):
        text = f"Это тестовое предложение номер {i} для проверки конвейера."
        rows.append({"id": f"bench{i:05d}", "text": text, "plain_text": text,
                     "stressed_text": text, "norm_text": text, "phonemes": [],
                     "category": "prose", "split": "train"})
    return rows


def run_sequential_fake(*, rows, voices, batch_size, base_seed, out, teacher, screener):
    """What ``build_corpus.py`` does today, with the same fake costs."""
    manifest = os.path.join(out, "manifest.jsonl")
    review = os.path.join(out, "review.jsonl")
    os.makedirs(out, exist_ok=True)
    started = time.perf_counter()
    gpu = 0.0
    clips = accepted = dropped = 0
    for voice in voices:
        for batch in bc.make_batches(rows, batch_size):
            ids = [r["id"] for r in batch]
            seed = bc.seed_for(ids, voice, 0, base_seed)
            wavs, sr, wall = teacher.generate(
                voice, [r["stressed_text"] for r in batch], seed)
            gpu += wall
            per = wall / max(len(batch), 1)
            failed = []
            for row, wav in zip(batch, wavs):
                res = screener.screen(row, wav, sr, voice, seed, 1, per)
                clips += 1
                if res["ok"]:
                    bc.append_jsonl(manifest, bc.accept(row, res, voice, out, "fake"))
                    accepted += 1
                else:
                    failed.append(row)
            if failed:
                seed = bc.seed_for([r["id"] for r in failed], voice, 1, base_seed)
                wavs, sr, wall = teacher.generate(
                    voice, [r["stressed_text"] for r in failed], seed)
                gpu += wall
                for row, wav in zip(failed, wavs):
                    res = screener.screen(row, wav, sr, voice, seed, 2, wall)
                    clips += 1
                    if res["ok"]:
                        bc.append_jsonl(manifest, bc.accept(row, res, voice, out, "fake"))
                        accepted += 1
                    else:
                        bc.append_jsonl(review, bc.review_row(row, res, voice))
                        dropped += 1
    wall = time.perf_counter() - started
    return {"clips": clips, "accepted": accepted, "dropped": dropped,
            "wall_s": round(wall, 3), "gpu_gen_s": round(gpu, 3),
            "clips_per_min": round(clips / wall * 60, 1) if wall else 0.0}


def benchmark(*, clips=96, batch_size=8, workers=3, gen_s=0.05, whisper_s=0.015,
              meter_s=0.05, out_root=None, check=True):
    """Sequential vs pipelined at identical per-clip costs.

    The absolute sleeps are scaled-down models of the pod (generation ~1 s/clip,
    Whisper ~0.2 s on GPU, meter2 ~0.7 s on CPU); only the ratios matter for the
    overlap proof.  ``clips`` clips, one voice.
    """
    root = out_root or os.path.join(bc.REPO, "data/bench-pipe")
    shutil.rmtree(root, ignore_errors=True)
    seq_out = os.path.join(root, "seq")
    pipe_out = os.path.join(root, "pipe")
    rows = bench_rows(clips)
    try:
        seq_gpu = GpuBusy()
        seq = run_sequential_fake(
            rows=rows, voices=["f_neutral"], batch_size=batch_size, base_seed=1234,
            out=seq_out, teacher=FakeTeacher(gen_s=gen_s, gpu=seq_gpu),
            screener=FakeScreener(seq_out, whisper_s=whisper_s, meter_s=meter_s,
                                  gpu=seq_gpu))
        seq["gpu_idle_pct"] = round(100 * (1 - seq_gpu.busy / seq["wall_s"]), 1) if seq["wall_s"] else 0.0
        pipe_gpu = GpuBusy()
        obs = run_pipelined(
            rows=rows, voices=["f_neutral"], batch_size=batch_size, base_seed=1234,
            out=pipe_out, teacher=FakeTeacher(gen_s=gen_s, gpu=pipe_gpu),
            screener=FakeScreener(pipe_out, whisper_s=whisper_s, meter_s=meter_s,
                                  gpu=pipe_gpu),
            adapter="fake", queue_size=2 * batch_size, screen_workers=workers)
        pipe = {"clips": obs["clips"], "accepted": obs["accepted"],
                "dropped": obs["dropped"], "wall_s": obs["wall_s"],
                "gpu_busy_s": round(pipe_gpu.busy, 3),
                "gpu_idle_pct": round(100 * (1 - pipe_gpu.busy / obs["wall_s"]), 1)
                if obs["wall_s"] else 0.0,
                "stage_a_idle_pct": obs["gpu_idle_pct"],
                "clips_per_min": round(obs["clips"] / obs["wall_s"] * 60, 1)
                if obs["wall_s"] else 0.0,
                "max_jobs_q": obs["max_jobs_q"]}
        assert obs["clips"] == seq["clips"] == clips, (obs["clips"], seq["clips"], clips)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    speedup = round(pipe["clips_per_min"] / seq["clips_per_min"], 2) if seq["clips_per_min"] else 0.0
    result = {"costs": {"gen_s_per_clip": gen_s, "whisper_s_per_clip": whisper_s,
                        "meter_s_per_clip": meter_s, "batch_size": batch_size,
                        "screen_workers": workers, "clips": clips},
              "sequential": seq, "pipelined": pipe, "speedup": speedup}
    if check:
        assert pipe["gpu_idle_pct"] < GPU_IDLE_MAX * 100, (
            f"GPU stage idle {pipe['gpu_idle_pct']}% >= {GPU_IDLE_MAX * 100}%")
        assert speedup >= SPEEDUP_MIN, f"speedup {speedup} < {SPEEDUP_MIN}"
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-fake", action="store_true",
                    help="prove stage overlap with a stub teacher and screener")
    ap.add_argument("--clips", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gen", type=float, default=0.05, help="stub generation s/clip")
    ap.add_argument("--whisper", type=float, default=0.015, help="stub Whisper s/clip (GPU)")
    ap.add_argument("--meter", type=float, default=0.05, help="stub meter2 s/clip (CPU)")
    args = ap.parse_args(argv)
    if not args.benchmark_fake:
        ap.error("nothing to do; pass --benchmark-fake (real runs: build_corpus.py --pipeline)")
    result = benchmark(clips=args.clips, batch_size=args.batch_size,
                       workers=args.workers, gen_s=args.gen,
                       whisper_s=args.whisper, meter_s=args.meter)
    print(json.dumps(result, indent=1), flush=True)
    print(f"[bench] sequential {result['sequential']['clips_per_min']} clips/min, "
          f"pipelined {result['pipelined']['clips_per_min']} clips/min "
          f"({result['speedup']}x), GPU idle {result['pipelined']['gpu_idle_pct']}%",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
