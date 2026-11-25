"""Verify an exported ONNX graph against the PyTorch checkpoint it came from."""

import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "training")

from voca_ru.text import analyze
from voca_ru.train.config import Config
from voca_ru.train.model import TinyVoca

TEXT = "Мой коллега уходит в отпуск на две недели."
one = lambda value: np.array([value], dtype=np.float32)


def main() -> None:
    checkpoint, onnx_path = sys.argv[1], sys.argv[2]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = Config.from_dict(state["config"])
    model = TinyVoca(config).eval()
    model.load_state_dict(state["model"], strict=True)

    features = analyze(TEXT)
    tokens = torch.tensor([features["tokens"]])
    stress = torch.tensor([features["stress"]])
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(1, config.hop * config.max_frames, generator=generator)

    with torch.inference_mode():
        audio, length, detail = model.synthesize(
            tokens, stress, torch.tensor([len(features["tokens"])]), torch.tensor([0]), seed=7
        )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {
        "tokens": tokens.numpy().astype(np.int64),
        "stress": stress.numpy().astype(np.int64),
        "voice": np.array([0], dtype=np.int64),
        "speed": one(1.0),
        "pitch_shift": one(0.0),
        "pitch_range": one(1.0),
    }
    # the shipped graph draws its own noise, so only the seeded export takes this input
    if any(item.name == "noise" for item in session.get_inputs()):
        feed["noise"] = noise.numpy()
    exported, durations = session.run(None, feed)

    n = min(exported.shape[1], int(length[0]))
    peak = float(np.abs(exported[:, :n]).max())
    count = len(features["tokens"])
    print(json_line := {
        "torch_samples": int(length[0]),
        "onnx_samples": int(exported.shape[1]),
        "same_length": exported.shape[1] == int(length[0]),
        "durations_match": bool((durations[0][:count] == detail["durations"][0].numpy()[:count]).all()),
        "max_rel_diff": round(float(np.abs(exported[0, :n] - audio[0, :n].numpy()).max()) / peak, 6),
    })


if __name__ == "__main__":
    main()
