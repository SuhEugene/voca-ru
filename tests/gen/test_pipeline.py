"""Model-free tests for the staged pipeline (no teacher, no screening models).

    . scripts/cache-env.sh
    PYTHONPATH=training .cache/venvs/r4/bin/python -m unittest discover -s tests/gen
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for path in (os.path.join(REPO, "training"), os.path.join(REPO, "scripts/gen")):
    if path not in sys.path:
        sys.path.insert(0, path)

import build_corpus as bc  # noqa: E402
import pipeline  # noqa: E402

VOICE = "f_neutral"
SEED = 1234


def make_rows(n):
    rows = []
    for i in range(n):
        text = f"Это тестовое предложение номер {i} для проверки конвейера."
        rows.append({"id": f"t{i:03d}", "text": text, "plain_text": text,
                     "stressed_text": text, "norm_text": text, "phonemes": [],
                     "category": "prose", "split": "train"})
    return rows


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp(prefix="pipe-test-")
        self.manifest = os.path.join(self.out, "manifest.jsonl")
        self.review = os.path.join(self.out, "review.jsonl")

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def run_pipe(self, rows, *, fail=(), workers=1, queue_size=8, done=None,
                 whisper_s=0.0, meter_s=0.0):
        teacher = pipeline.FakeTeacher(gen_s=0.0)
        screener = pipeline.FakeScreener(self.out, whisper_s=whisper_s,
                                         meter_s=meter_s, fail=fail)
        obs = pipeline.run_pipelined(
            rows=rows, voices=[VOICE], batch_size=4, base_seed=SEED, out=self.out,
            teacher=teacher, screener=screener, adapter="fake",
            queue_size=queue_size, screen_workers=workers, done=done)
        return obs, teacher, screener

    def test_exactly_one_row_per_clip_and_unique_ids(self):
        rows = make_rows(10)
        obs, _, _ = self.run_pipe(rows, workers=3)
        manifest, review = read(self.manifest), read(self.review)
        ids = [r["id"] for r in manifest] + [r["id"] for r in review]
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(set(ids)), 10)
        self.assertEqual(obs["clips"], 10)
        self.assertEqual(obs["accepted"], 10)
        self.assertEqual(obs["dropped"], 0)
        self.assertFalse(os.path.exists(os.path.join(self.out, ".work")))

    def test_manifest_order_follows_length_buckets_with_one_worker(self):
        rows = list(reversed(make_rows(9)))
        obs, teacher, _ = self.run_pipe(rows, workers=1)
        expected = [bc.clip_id(r, VOICE)
                    for batch in bc.make_batches(rows, 4) for r in batch]
        self.assertEqual(obs["gen_order"], expected)
        self.assertEqual([r["id"] for r in read(self.manifest)], expected)
        self.assertEqual([n for _, n in teacher.calls], [4, 4, 1])

    def test_retry_then_review(self):
        rows = make_rows(4)
        cid = bc.clip_id(rows[0], VOICE)
        obs, teacher, screener = self.run_pipe(rows, fail={(cid, 1), (cid, 2)})
        self.assertEqual(obs["dropped"], 1)
        self.assertEqual(obs["accepted"], 3)
        review = read(self.review)
        self.assertEqual([r["id"] for r in review], [cid])
        self.assertEqual(review[0]["attempt"], 2)
        # generated once in the batch, once as a single-item retry
        self.assertEqual(sum(1 for c, a in screener.calls if c == cid), 2)
        self.assertEqual([n for _, n in teacher.calls], [4, 1])
        self.assertEqual([r["id"] for r in read(self.manifest)], [
            bc.clip_id(r, VOICE) for r in rows[1:]])

    def test_retry_then_accept_uses_attempt_2_seed(self):
        rows = make_rows(3)
        cid = bc.clip_id(rows[1], VOICE)
        obs, _, _ = self.run_pipe(rows, fail={(cid, 1)})
        self.assertEqual(obs["accepted"], 3)
        self.assertEqual(read(self.review), [])
        row = [r for r in read(self.manifest) if r["id"] == cid][0]
        self.assertEqual(row["attempt"], 2)
        self.assertEqual(row["seed"], bc.seed_for([rows[1]["id"]], VOICE, 1, SEED))

    def test_resume_after_torn_manifest_regenerates_only_missing(self):
        rows = make_rows(6)
        self.run_pipe(rows)
        with open(self.manifest) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 6)
        with open(self.manifest, "w") as f:        # a kill mid-append
            f.writelines(lines[:-1])
            f.write(lines[-1][:len(lines[-1]) // 2])
        done = pipeline.repair_outputs(self.manifest, self.review)
        self.assertEqual(len(done), 5)
        obs, teacher, _ = self.run_pipe(rows, done=done)
        self.assertEqual(obs["clips"], 1)
        self.assertEqual([n for _, n in teacher.calls], [1])
        ids = [r["id"] for r in read(self.manifest)]
        self.assertEqual(len(ids), 6)
        self.assertEqual(set(ids), {bc.clip_id(r, VOICE) for r in rows})
        with open(self.manifest, "rb") as f:
            self.assertTrue(f.read().endswith(b"\n"))

    def test_repair_collapses_duplicate_rows(self):
        rows = make_rows(2)
        self.run_pipe(rows)
        with open(self.manifest) as f:
            first = f.readline()
        with open(self.manifest, "a") as f:
            f.write(first)
        done = pipeline.repair_outputs(self.manifest, self.review)
        self.assertEqual(len(done), 2)
        self.assertEqual(len(read(self.manifest)), 2)

    def test_resume_with_everything_done_generates_nothing(self):
        rows = make_rows(3)
        self.run_pipe(rows)
        obs, teacher, _ = self.run_pipe(rows)
        self.assertEqual(obs["clips"], 0)
        self.assertEqual(teacher.calls, [])

    def test_queue_backpressure_is_bounded(self):
        rows = make_rows(16)
        obs, teacher, screener = self.run_pipe(rows, workers=1, queue_size=2,
                                               meter_s=0.01)
        self.assertLessEqual(obs["max_jobs_q"], 2)
        # stages actually overlapped: screening started before generation ended
        gen_end = max(end for _, end in teacher.times)
        screen_start = min(start for start, _ in screener.times)
        self.assertLess(screen_start, gen_end)

    def test_benchmark_proves_overlap(self):
        result = pipeline.benchmark(clips=32, batch_size=4, workers=3,
                                    gen_s=0.02, whisper_s=0.005, meter_s=0.02)
        self.assertEqual(result["pipelined"]["clips"], 32)
        self.assertLess(result["pipelined"]["gpu_idle_pct"], pipeline.GPU_IDLE_MAX * 100)
        self.assertGreater(result["pipelined"]["clips_per_min"],
                           result["sequential"]["clips_per_min"])

    def test_teacher_error_fails_fast_instead_of_hanging(self):
        class Boom(pipeline.FakeTeacher):
            def generate(self, voice, texts, seed):
                raise RuntimeError("boom")

        with self.assertRaises(RuntimeError) as ctx:
            pipeline.run_pipelined(
                rows=make_rows(3), voices=[VOICE], batch_size=4, base_seed=SEED,
                out=self.out, teacher=Boom(), screener=pipeline.FakeScreener(self.out),
                adapter="fake", screen_workers=2)
        self.assertIn("stage A", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
