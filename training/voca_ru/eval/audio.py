"""Audio metadata without a heavy dependency: stdlib wave, ffprobe fallback."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path


def audio_info(path: str | Path) -> tuple[float, int]:
    """(duration_seconds, sample_rate). Raises if the file cannot be read."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate, rate
    except (wave.Error, EOFError):
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=sample_rate", "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot read audio {path}: {exc}") from exc
    meta = json.loads(out.stdout)
    duration = float(meta["format"]["duration"])
    streams = meta.get("streams") or [{}]
    return duration, int(streams[0].get("sample_rate") or 0)
