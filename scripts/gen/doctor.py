"""`make doctor` — device, package and model-cache check for teacher generation."""
import importlib.metadata as md
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_git = os.popen(f"git -C {REPO} rev-parse --path-format=absolute --git-common-dir").read().strip()
SHARED = os.path.dirname(os.path.abspath(_git)) if _git else REPO
MODELS = os.path.join(SHARED, ".cache/models")

print(f"python       {sys.version.split()[0]}")
for p in ["torch", "qwen-tts", "transformers", "soundfile", "librosa", "numpy"]:
    try:
        print(f"{p:<12} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p:<12} MISSING")
print(f"mps          {torch.backends.mps.is_available()}")
for name in ["qwen3-tts-1.7b", "qwen3-tts-1.7b-voicedesign"]:
    path = os.path.join(MODELS, name)
    weights = os.path.join(path, "model.safetensors")
    ok = os.path.isfile(weights)
    size = os.path.getsize(weights) / 1024**3 if ok else 0
    print(f"model {name:<30} {'ok' if ok else 'MISSING'} {size:.2f} GiB  {path}")
