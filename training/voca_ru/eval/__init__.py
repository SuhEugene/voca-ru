"""Round-trip intelligibility evaluation for Russian voca (F5).

    PYTHONPATH=training python -m voca_ru.eval --manifest m.jsonl \
        --audio-dir audio --out report.json --md report.md

Public API::

    from voca_ru.eval import evaluate, make_backend, wer, cer, score_text
"""

from .asr import FakeASR, FasterWhisperASR, GigaAMASR, make_backend
from .metrics import cer, edit_counts, wer
from .runner import evaluate, load_manifest, to_markdown
from .scoring import implementation, score_text
from .stress import evaluate_stress, load_stress_set

__all__ = [
    "FakeASR", "FasterWhisperASR", "GigaAMASR", "make_backend",
    "cer", "edit_counts", "wer", "evaluate", "load_manifest", "to_markdown",
    "implementation", "score_text", "evaluate_stress", "load_stress_set",
]
