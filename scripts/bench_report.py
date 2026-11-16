"""Turn out/bench/*.json into the Markdown tables of BENCHMARK.md."""
import json
import platform
import subprocess
from pathlib import Path

bench = Path("out/bench")
load = lambda name: json.loads((bench / name).read_text()) if (bench / name).exists() else None


def machine():
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = ""
    return f"{cpu or platform.processor()}, {platform.system()} {platform.release()} ({platform.machine()})"


def versions():
    import onnxruntime
    import torch
    rust = subprocess.run([str(Path.home() / ".cargo/bin/rustc"), "--version"], capture_output=True, text=True).stdout.strip()
    return f"PyTorch {torch.__version__}, ONNX Runtime {onnxruntime.__version__}, Python {platform.python_version()}, {rust}"


engines = [("PyTorch (reference)", "pytorch"), ("ONNX Runtime (published model)", "onnx"), ("Rust, pure-Rust matmul", "rustmm"), ("Rust, Accelerate", "rust")]
print("# Speed comparison\n")
print(f"Machine: {machine()}.  ")
print(f"Software: {versions()}.\n")
print("Produced by `bash scripts/benchmark.sh`. Every engine gets the same phoneme ids; times are the median of repeated runs")
print("after warm-up, for one utterance, model only (phonemes in, waveform out, noise generation included).\n")

for threads in (1, 4):
    results = {key: load(f"{key}-{threads}.json") for _, key in engines}
    if not results["pytorch"] or not results["rust"]:
        continue
    cases = [c for c in results["pytorch"]["cases"] if c["case"] in {r["case"] for r in results["rust"]["cases"]}]
    # One row per distinct sentence (voice F2).
    seen, rows = set(), []
    for case in cases:
        if case["tokens"] not in seen:
            seen.add(case["tokens"])
            rows.append(case["case"])
    print(f"## Model only, {threads} thread{'s' if threads > 1 else ''}\n")
    header = ["Audio", "Phonemes"] + [name for name, key in engines if results[key]] + ["Rust vs PyTorch", "Rust vs ONNX"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---:|" * len(header))
    for name in rows:
        pick = {key: next(c for c in results[key]["cases"] if c["case"] == name) for _, key in engines if results[key]}
        audio = pick["pytorch"]["audio_s"]
        cells = [f"{audio:.1f} s", str(pick["pytorch"]["tokens"])]
        cells += [f"{pick[key]['median_ms']:.1f} ms ({audio * 1e3 / pick[key]['median_ms']:.0f}x)" for _, key in engines if key in pick]
        cells.append(f"**{pick['pytorch']['median_ms'] / pick['rust']['median_ms']:.1f}x**")
        cells.append(f"{pick['onnx']['median_ms'] / pick['rust']['median_ms']:.1f}x" if "onnx" in pick else "")
        print("| " + " | ".join(cells) + " |")
    print("\n(n x) is the multiple of real time. Model load: "
          + ", ".join(f"{name} {results[key]['load_ms']:.0f} ms" for name, key in engines if results[key]) + ".\n")

cli = load("cli.json")
if cli:
    print("## Whole command, new process (text in, WAV file out)\n")
    print("`python -m tiny_voca.infer` against `tiny-voca --no-split`, one thread each on the Rust side, median of 10 runs.\n")
    print("| Audio | Python | Rust | Speed-up |\n|---:|---:|---:|---:|")
    for row in cli:
        print(f"| {row['audio_s']:.1f} s | {row['python_ms']:.0f} ms | {row['rust_ms']:.0f} ms | **{row['python_ms'] / row['rust_ms']:.0f}x** |")
    print()

phonemizer = load("phonemizer.json")
if phonemizer:
    p, r = phonemizer["python"], phonemizer["rust"]
    print("## Phonemizer only (dizione)\n")
    print(f"{phonemizer['sentences']} test texts (heavy in numbers, dates and made-up words); {phonemizer['mismatches']} differ between the two implementations.\n")
    print("| | Python | Rust | Speed-up |\n|---|---:|---:|---:|")
    for label, key, unit in (("Load the dictionary and tables", "load_ms", "ms"), ("Per text, first pass (cold word cache)", "first_pass_ms_per_line", "ms"),
                             ("Per text, warm", "warm_ms_per_line", "ms")):
        print(f"| {label} | {p[key]:.3g} {unit} | {r[key]:.3g} {unit} | **{p[key] / r[key]:.1f}x** |")
    print()
