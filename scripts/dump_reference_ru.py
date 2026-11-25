"""Dump reference inputs and outputs of the PyTorch Russian model for the Rust parity test.

    PYTHONPATH=training .cache/venvs/r4/bin/python scripts/dump_reference_ru.py \
        .cache/models/voca-ru-run/inference.pt bench/sentences.txt out/reference-ru

One safetensors file per case: tokens, stress, the exact noise torch drew,
durations, prosody, f0 and the waveform. Mirrors scripts/dump_reference.py for
the Russian checkpoint and frontend.
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from voca_ru.text import analyze  # noqa: E402
from voca_ru.train.config import Config  # noqa: E402
from voca_ru.train.model import TinyVoca  # noqa: E402

checkpoint, sentences, target = sys.argv[1], sys.argv[2], Path(sys.argv[3])
target.mkdir(parents=True, exist_ok=True)
state = torch.load(checkpoint, map_location="cpu", weights_only=True)
config = Config.from_dict(state["config"])
model = TinyVoca(config).eval()
model.load_state_dict(state["model"], strict=True)
texts = [line.strip() for line in open(sentences, encoding="utf-8") if line.strip()]
settings = [dict(speed=1.0, pitch_shift=0.0, pitch_range=1.0, seed=0),
            dict(speed=1.3, pitch_shift=0.12, pitch_range=1.4, seed=3)]
case = 0
with torch.inference_mode():
    for index, text in enumerate(texts):
        fields = analyze(text)
        tokens = torch.tensor([fields["tokens"]])
        stress = torch.tensor([fields["stress"]])
        lengths = torch.tensor([len(fields["tokens"])])
        for voice in range(len(config.speakers)):
            for s in (settings if index < 2 else settings[:1]):
                wave, samples, details = model.synthesize(
                    tokens, stress, lengths, torch.tensor([voice]), speed=s["speed"],
                    pitch_shift=s["pitch_shift"], seed=s["seed"], pitch_range=s["pitch_range"])
                samples = int(samples[0])
                noise = torch.randn(samples, generator=torch.Generator().manual_seed(s["seed"]))
                f0 = model.frequency(details["prosody"][..., 0] + s["pitch_shift"])
                save_file(
                    {"tokens": tokens[0].int(), "stress": stress[0].int(), "noise": noise,
                     "durations": details["durations"][0].int(), "prosody": details["prosody"][0].contiguous(),
                     "f0": f0[0].contiguous(), "wave": wave[0, :samples].contiguous()},
                    str(target / f"case{case:02d}.safetensors"),
                    metadata={"text": text, "voice": str(voice), **{k: repr(v) for k, v in s.items()}})
                case += 1
print(f"wrote {case} cases to {target}")
