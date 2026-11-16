"""Time the reference implementations on the same inputs as the Rust `bench`.

    DIZIONE_DATA=weights/phonemizer python scripts/bench_python.py pytorch 1 > out/bench/pytorch-1.json
    python scripts/bench_python.py onnx 4 > out/bench/onnx-4.json

Reads the token inputs from out/reference (scripts/dump_reference.py), runs
each default-settings case RUNS times after 3 warm-up runs and prints JSON in
the format of the Rust benchmark. One engine and thread count per process, so
thread pools never interfere.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

engine, threads = sys.argv[1], int(sys.argv[2])
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
reference = Path("out/reference")

cases = []
for path in sorted(reference.glob("*.safetensors")):
    with safe_open(str(path), "np") as f:
        meta = f.metadata()
        if meta["speed"] != "1.0":
            continue
        cases.append((path.stem, f.get_tensor("tokens").astype(np.int64), f.get_tensor("stress").astype(np.int64), int(meta["voice"]),
                      len(f.get_tensor("wave"))))

started = time.perf_counter()
if engine == "pytorch":
    import torch
    from tiny_voca.infer import load_model
    torch.set_num_threads(threads)
    model = load_model("weights/tiny-voca.pt", check_frontend=False)

    def run(tokens, stress, voice):
        t = torch.from_numpy(tokens)[None]
        wave, lengths, _ = model.synthesize(t, torch.from_numpy(stress)[None], torch.tensor([len(tokens)]), torch.tensor([voice]))
        return int(lengths[0])
elif engine == "onnx":
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession("weights/onnx/model.onnx", options, providers=["CPUExecutionProvider"])
    one = lambda value: np.array([value], dtype=np.float32)

    def run(tokens, stress, voice):
        audio, _ = session.run(None, {"tokens": tokens[None], "stress": stress[None], "voice": np.array([voice], dtype=np.int64),
                                      "speed": one(1.0), "pitch_shift": one(0.0), "pitch_range": one(1.0)})
        return audio.shape[1]
else:
    raise SystemExit("engine must be pytorch or onnx")
load_ms = (time.perf_counter() - started) * 1e3

rows = []
for name, tokens, stress, voice, samples in cases:
    times = []
    for index in range(runs + 3):
        t = time.perf_counter()
        produced = run(tokens, stress, voice)
        if index >= 3:
            times.append((time.perf_counter() - t) * 1e3)
    assert produced == samples, (name, produced, samples)
    times.sort()
    rows.append({"case": name, "tokens": len(tokens), "audio_s": round(samples / 24000, 4), "median_ms": round(times[len(times) // 2], 4),
                 "min_ms": round(times[0], 4)})
print(json.dumps({"engine": engine, "threads": threads, "load_ms": round(load_ms, 3), "cases": rows}))
