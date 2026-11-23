"""One-clip smoke test through the real corpus pipeline.

Runs ``scripts/gen/build_corpus.py --limit 1`` (Base + mandatory LoRA, F3 text
with stress marks, Whisper WER, meter2 stress) into a throwaway directory and
checks the result. ``--check DIR`` validates an existing smoke output without a
GPU, which is what the local tests use.

    python remote/lib/smoke.py            # after bootstrap
    python remote/lib/smoke.py --check out/smoke
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys


def find_row(manifest_path):
    if not os.path.isfile(manifest_path):
        return None
    rows = []
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows[0] if rows else None


def check(smoke_dir):
    """Return (ok, lines). Pure file inspection; no GPU needed."""
    lines = []
    ok = True

    def fail(message):
        nonlocal ok
        ok = False
        lines.append(f"FAIL {message}")

    row = find_row(os.path.join(smoke_dir, "manifest.jsonl"))
    if row is None:
        fail("manifest.jsonl has no accepted clip")
        return ok, lines
    lines.append(f"PASS clip {row.get('id')} accepted")

    adapter = str(row.get("teacher_adapter") or "")
    if "lora" not in adapter.lower() or "none" in adapter.lower():
        fail(f"teacher_adapter is not the stress LoRA: {adapter!r}")
    else:
        lines.append(f"PASS stress adapter loaded: {adapter}")

    stressed = row.get("stressed_text") or ""
    marks = stressed.count("\u0301")
    if marks < 2:
        fail(f"stress marks missing from prompt text ({marks} found)")
    else:
        lines.append(f"PASS stress marks in prompt text: {marks}")

    wer = row.get("wer")
    if not isinstance(wer, (int, float)):
        fail("no WER recorded for the clip")
    else:
        lines.append(f"PASS WER {100 * float(wer):.2f} %")

    if row.get("stress_ok") is None:
        fail("stress check did not run")
    else:
        lines.append(
            f"PASS meter2 stress: ok={row['stress_ok']} "
            f"checked={row.get('stress_checked')} errors={row.get('stress_errors')}"
        )

    path = os.path.join(smoke_dir, str(row.get("audio")))
    if not os.path.isfile(path):
        fail(f"audio file missing: {path}")
        return ok, lines
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.samplerate != 24000 or info.channels != 1:
            fail(f"audio is {info.samplerate} Hz / {info.channels} ch, expected 24000/mono")
        elif info.subtype != "PCM_16":
            fail(f"audio subtype {info.subtype}, expected PCM_16")
        else:
            lines.append(f"PASS FLAC {info.samplerate} Hz mono PCM16, {info.duration:.2f} s")
    except Exception as exc:
        fail(f"cannot read audio: {type(exc).__name__}: {exc}")
    return ok, lines


def run_smoke(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    python = os.path.join(root, ".venv-remote", "bin", "python")
    if not os.path.isfile(python):
        python = sys.executable
    command = [
        python, os.path.join(root, "scripts/gen/build_corpus.py"),
        "--limit", "1", "--voices", "f_neutral", "--batch-size", "1",
        "--split", "all", "--out", out_dir, "--require-lora",
        "--asr-device", os.environ.get("VOCA_ASR_DEVICE", "cuda"),
        "--asr-compute-type", "float16",
    ]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", os.path.join(root, "training"))
    print("[smoke] " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=root, env=env)
    if result.returncode != 0:
        print(f"FAIL build_corpus exited {result.returncode}", flush=True)
        return False
    return check(out_dir)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..")))
    parser.add_argument("--out", default=None, help="smoke output dir")
    parser.add_argument("--check", default=None, help="validate an existing smoke dir")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    if args.check:
        ok, lines = check(os.path.abspath(args.check))
        print("\n".join(lines))
        print("SMOKE PASS" if ok else "SMOKE FAIL")
        return 0 if ok else 1

    out_dir = args.out or os.path.join(args.root, "out/remote-smoke")
    if os.path.isdir(out_dir) and not args.keep:
        shutil.rmtree(out_dir)
    ok = run_smoke(args.root, out_dir)
    _ok, lines = check(out_dir)
    print("\n".join(lines))
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
