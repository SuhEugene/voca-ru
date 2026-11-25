"""Synthesise Russian text with a trained checkpoint.

    python scripts/say.py inference.pt "Привет!" --voice M2 --out out/say

Voice names come from the checkpoint's ``config["speakers"]`` (F2 / M2 here).
"""

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from voca_ru.text import analyze  # noqa: E402
from voca_ru.train.config import Config  # noqa: E402
from voca_ru.train.model import TinyVoca  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--voice", default=None, help="speaker name; default is every voice")
    parser.add_argument("--out", type=Path, default=Path("out/say"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pitch-shift", type=float, default=0.0,
                        help="log-F0 offset; 0.058 is about a semitone, negative is lower")
    parser.add_argument("--pitch-range", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = Config.from_dict(state["config"])
    model = TinyVoca(config).eval()
    model.load_state_dict(state["model"], strict=True)
    speakers = list(config.speakers)
    chosen = [speakers.index(args.voice)] if args.voice else range(len(speakers))
    args.out.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for index, text in enumerate(args.texts, start=1):
            fields = analyze(text)
            phones = fields["tokens"]
            levels = fields["stress"]
            tokens = torch.tensor([phones])
            stress = torch.tensor([levels])
            for voice in chosen:
                wave, lengths, _ = model.synthesize(
                    tokens, stress, torch.tensor([len(phones)]), torch.tensor([voice]), seed=args.seed,
                    pitch_shift=args.pitch_shift, pitch_range=args.pitch_range, speed=args.speed)
                path = args.out / f"{index:02d}-{speakers[voice]}.wav"
                sf.write(path, wave[0, : int(lengths[0])].float().numpy(), config.sample_rate)
                print(json.dumps({"file": str(path), "voice": speakers[voice], "text": text,
                                  "seconds": round(int(lengths[0]) / config.sample_rate, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
