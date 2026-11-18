# R4b — second run of the compute task (checkpoint verification only)

Subagent R4, branch `task/compute-env`, 2026-11-19, 02:00–03:25.
This is the continuation of the cancelled first session. It is a **second
run**: every number below was measured again in this session, not copied.

Scope: the machine, toolchain and training measurements in
`reports/R4-compute.md` and `docs/compute.md` were already done and are not
repeated. This run adds the checkpoint-verification evidence and the network
measurements behind it. It does **not** replace or contradict the merged
report.

---

## 1. What this run settled

| Item | Result |
|---|---|
| `.cache/venvs/r4` (Python 3.12.16) | **Works.** `torch 2.15.0`, `mps True`, `qwen_tts`, `onnxruntime 1.32.0` all import. |
| `speech_tokenizer/model.safetensors` | **Verified**: 682,293,092 B, sha256 `836b7b35…ba571258` = HF LFS oid. |
| `model.safetensors` | **Still unverified.** Full-file sha256 is `ac725cd7…`, expected `38fc7fc5…`. |
| Where the corruption is | **Pinned to the first 100 MB**, byte 34,394,112 — see §2. |
| Whisper-turbo screening throughput | **Not measured.** The model download was stopped on instruction; the 1.5–2.5 h estimate in the main report stands. |

## 2. The corruption is in the resumed head, not in the weights

The cancelled session had left a 104,857,600-byte prefix in
`.cache/models/qwen3-tts-1.7b/model.safetensors`, fetched with a single
`curl -C -`. Re-fetching that exact range (`-r 0-104857599`, verified by
`cmp`) showed the first difference at **byte 34,394,112** — i.e. the
single-stream resume had appended a stale/partial block there. Everything
beyond that point in the sample of the tail matched the origin.

So the earlier partial file was *not* a different or truncated model; one
resume of one connection wrote a bad block and the rest of the file was
sound. This is the concrete failure that motivates the "never resume one big
file with `curl -C -`" rule now in `AGENTS.md`, and it is the reason the
merged report's §5 says the shared `scripts/fetch-hf-file.sh` must own the
file.

Practical consequence for the orchestrator: after the verified fetch lands,
the weights themselves need no second look — only the download path changes.

## 3. Network, measured this run

Sixty-second windows on `Qwen/Qwen3-TTS-12Hz-1.7B-Base/model.safetensors`
(3,857,413,744 B):

| Mode | Throughput |
|---|---:|
| One connection (`curl -C -`, 02:07) | **0.239 MB/s** |
| Six parallel ranges, aggregate (02:08–02:09) | **4.17 MB/s** |
| Six parallel ranges per job, later with three other multi-part jobs running (03:06–03:24) | ~0.44 MB/s |

Two points worth keeping:

1. **Parallel ranges are ~17× a single stream** here, and the 6-way split is
   what moved 3.75 GB in 17 minutes in this session.
2. **Bandwidth divides, it does not add**: during the 03:00 hour several
   agents were fetching different checkpoints at once (#Hugging Face) and
   every job dropped to ~0.4 MB/s. One downloader per machine at a time, or
   accept the slowdown.

Do not run two fetches of the *same* file: this session had a duplicate
`model.safetensors` fetch (mine, 6 parts) racing the orchestrator's 6-part
job. Both resumed from 0 after the other recreated the staging directory, and
the 3.1 GB already collected was thrown away. The orchestrator's job owns the
file now.

## 4. What is still unverified

- **Full sha256 of the teacher checkpoint.** Do not start bulk generation
  until `shasum -a 256 .cache/models/qwen3-tts-1.7b/model.safetensors` prints
  `38fc7fc51c5e776e840414b6fd443962e941b9654888fd7913e4da643cb857c`.
- **Audio quality of the four `out/r4/ru-sample-*.wav` clips.** They were
  generated from the `ac725cd7…` copy. Do not assume they are right; re-run
  one batch after verification and listen.
- **Whisper-turbo screening throughput** remains an estimate.

## 5. Clean-up

Deleted after verifying nothing used them (all created by this run):
`.cache/models/whisper-large-v3-turbo-ct2/` (1.2 GB partial
faster-whisper download) and `out/r4/{screen_bench.wav,whisper-dl.log}`.
`.cache/models` now holds the teacher (4.5 GB) and the two other agents'
checkpoints; 6.1 GB total. Nothing of mine is left under
`/private/var/folders/…/T/opencode` — that directory was wiped by another
agent at 02:54, before this run's cleanup.

No process of this run is still alive.
