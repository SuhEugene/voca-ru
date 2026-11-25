"""Export a voca-ru inference checkpoint to ONNX.

    python scripts/export_onnx.py inference.pt out/voca-ru.onnx

The graph takes phoneme ids and stress for one utterance and returns the
waveform plus per-token frame durations. Text stays outside the graph: the
Russian frontend (normalise -> RUAccent stress -> G2P -> ids) runs in Python or
Rust before the call.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from voca_ru.text.symbols_ru import SYMBOLS  # noqa: E402
from voca_ru.train.config import Config  # noqa: E402
from voca_ru.train.model import TinyVoca  # noqa: E402
from voca_ru.train.onnx_export import export  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--external-noise", action="store_true")
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = Config.from_dict(state["config"])
    model = TinyVoca(config).eval()
    model.load_state_dict(state["model"], strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(model, args.output, args.external_noise, args.opset, args.optimized)
    meta = {"sample_rate": config.sample_rate, "hop": config.hop, "voices": list(config.speakers),
            "symbols": list(SYMBOLS), "max_tokens": config.max_tokens, "max_frames": config.max_frames,
            "step": state.get("step"), "frontend": "voca-ru-text", "language": "ru",
            "inputs": ["tokens", "stress", "voice", "speed", "pitch_shift", "pitch_range"],
            "outputs": ["audio", "durations"]}
    args.output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
