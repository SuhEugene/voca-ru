"""Same two phrases from three checkpoints (+0 / +30k / +60k), both voices."""

import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, "training")

from voca_ru.text import analyze
from voca_ru.train.config import Config
from voca_ru.train.model import TinyVoca

CHECKPOINTS = [
    ("base", ".cache/models/voca-ru-run/inference.pt"),
    ("plus30k", ".cache/models/voca-ru-ft/stage1/inference.pt"),
    ("plus60k", ".cache/models/voca-ru-ft/stage2/inference.pt"),
]
TEXTS = [
    ("01", "Мой коллега уходит в отпуск на две недели."),
    ("02", "Ну, Котик, сегодня ты играла, а завтра будет новый день."),
]
OUT = Path("out/ab-60k")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for label, path in CHECKPOINTS:
        state = torch.load(path, map_location="cpu", weights_only=True)
        config = Config.from_dict(state["config"])
        model = TinyVoca(config).eval()
        model.load_state_dict(state["model"], strict=True)
        for voice_index, voice in enumerate(config.speakers):
            for name, text in TEXTS:
                features = analyze(text)
                with torch.inference_mode():
                    audio, _, _ = model.synthesize(
                        torch.tensor([features["tokens"]]),
                        torch.tensor([features["stress"]]),
                        torch.tensor([len(features["tokens"])]),
                        torch.tensor([voice_index]),
                        seed=7,
                    )
                target = OUT / f"{name}-{label}-{voice}.wav"
                sf.write(target, audio[0].numpy(), 24000)
                print(f"{target.name}  {len(audio[0]) / 24000:.2f}s")


if __name__ == "__main__":
    main()
