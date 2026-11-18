"""Five tone/noise fixtures (no speech) for the harness self-check."""

from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

# id, expected text, seconds, kind
FIXTURES = [
    ("tone_exact", "один два три", 0.5, "tone"),
    ("tone_other", "четыре пять шесть семь", 0.5, "tone"),
    ("noise_exact", "сегодня хорошая погода", 0.6, "noise"),
    ("noise_other", "завтра будет дождь", 0.6, "noise"),
    ("mixed_exact", "цена пятьсот рублей", 0.7, "mixed"),
]

RATE = 24_000


def _samples(kind: str, seconds: float) -> list[int]:
    n = int(RATE * seconds)
    rng = random.Random({"tone": 1, "noise": 2, "mixed": 3}[kind])
    out = []
    for i in range(n):
        if kind == "tone":
            value = 0.4 * math.sin(2 * math.pi * 440 * i / RATE)
        elif kind == "noise":
            value = 0.2 * rng.uniform(-1, 1)
        else:
            value = (0.25 * math.sin(2 * math.pi * 220 * i / RATE)
                     + 0.1 * rng.uniform(-1, 1))
        out.append(int(max(-1.0, min(1.0, value)) * 32767))
    return out


def make_fixtures(root: str | Path) -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, _text, seconds, kind in FIXTURES:
        path = root / f"{name}.wav"
        samples = _samples(kind, seconds)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        paths.append(path)
    return paths
