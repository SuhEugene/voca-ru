"""ASR judges for round-trip intelligibility.

Primary: faster-whisper large-v3-turbo (CTranslate2).  Optional: GigaAM-v3.
CTranslate2 has no Metal backend, so on this machine "MPS/CPU" resolves to CPU;
asking for ``--device mps`` fails instead of silently ignoring it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Protocol


class ASRBackend(Protocol):
    name: str

    def transcribe(self, audio_path: str) -> str: ...


def default_model_dir() -> Path:
    cache = Path(os.environ.get("VOCA_CACHE", Path(__file__).resolve().parents[3] / ".cache"))
    return cache / "models" / "whisper-large-v3-turbo"


class FasterWhisperASR:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ru",
        beam_size: int = 5,
    ) -> None:
        if device == "mps":
            raise ValueError(
                "faster-whisper/CTranslate2 has no MPS backend; use --device cpu"
            )
        model_dir = Path(model_dir) if model_dir else default_model_dir()
        if not (model_dir / "model.bin").is_file():
            raise FileNotFoundError(
                f"ASR model missing: {model_dir / 'model.bin'} "
                "(expected a CTranslate2 faster-whisper model directory)"
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # loud: the judge is not optional
            raise RuntimeError(
                "faster-whisper is not installed in this interpreter"
            ) from exc
        self.name = f"faster-whisper:{model_dir.name}:{compute_type}"
        self.language = language
        self.beam_size = beam_size
        try:
            self._model = WhisperModel(
                str(model_dir), device=device, compute_type=compute_type
            )
        except Exception as exc:
            raise RuntimeError(f"cannot load ASR model {model_dir}: {exc}") from exc

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self._model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class GigaAMASR:
    """Optional cross-check. Install with ``uv pip install gigaam``.

    GigaAM wants 16 kHz mono; the model wrapper handles resampling.
    """

    def __init__(self, model_name: str = "v3_e2e_rnnt", device: str = "cpu") -> None:
        try:
            import gigaam
        except ImportError as exc:
            raise RuntimeError(
                "GigaAM backend requested but `gigaam` is not installed"
            ) from exc
        self.name = f"gigaam:{model_name}"
        self._model = gigaam.load_model(model_name, device=device)

    def transcribe(self, audio_path: str) -> str:
        return self._model.transcribe(audio_path).strip()


class FakeASR:
    """Deterministic backend for the self-check: path -> transcript map."""

    def __init__(self, transcripts: dict[str, str]) -> None:
        self.name = "fake"
        self._by_path = {str(Path(k).resolve()): v for k, v in transcripts.items()}

    def transcribe(self, audio_path: str) -> str:
        key = str(Path(audio_path).resolve())
        if key not in self._by_path:
            raise KeyError(f"fake ASR has no transcript for {key}")
        return self._by_path[key]


def make_backend(kind: str, **kwargs) -> ASRBackend:
    if kind in ("faster-whisper", "whisper"):
        return FasterWhisperASR(**kwargs)
    if kind == "gigaam":
        allowed = {"model_name": kwargs.get("gigaam_model", "v3_e2e_rnnt"),
                   "device": kwargs.get("device", "cpu")}
        return GigaAMASR(**allowed)
    if kind == "fake":
        return FakeASR(json.loads(Path(kwargs["fake_transcripts"]).read_text("utf-8")))
    raise ValueError(f"unknown ASR backend: {kind}")


def timed_transcribe(backend: ASRBackend, audio_path: str) -> tuple[str, float]:
    start = time.perf_counter()
    text = backend.transcribe(audio_path)
    return text, time.perf_counter() - start
