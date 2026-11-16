"""Wall time of the whole command (new process, text in, WAV out): the
reference `python -m tiny_voca.infer` against the Rust `tiny-voca`."""
import json
import os
import subprocess
import sys
import time

texts = [line.strip() for line in open("bench/sentences.txt", encoding="utf-8") if line.strip()][:3]
env = {**os.environ, "DIZIONE_DATA": "weights/phonemizer"}
commands = {
    "python": lambda text: [sys.executable, "-m", "tiny_voca.infer", "--checkpoint", "weights/tiny-voca.pt", "--text", text, "--output", "out/bench/cli_py.wav"],
    "rust": lambda text: ["./target/release/tiny-voca", "--no-split", "--quiet", "--text", text, "--output", "out/bench/cli_rs.wav"],
}
rows = []
for text in texts:
    row = {"text": text}
    for name, command in commands.items():
        times = []
        for index in range(12):
            started = time.perf_counter()
            subprocess.run(command(text), check=True, env=env, stdout=subprocess.DEVNULL)
            if index >= 2:
                times.append((time.perf_counter() - started) * 1e3)
        times.sort()
        row[name + "_ms"] = round(times[len(times) // 2], 2)
    import soundfile as sf
    row["audio_s"] = round(sf.info("out/bench/cli_rs.wav").duration, 3)
    rows.append(row)
print(json.dumps(rows, ensure_ascii=False))
