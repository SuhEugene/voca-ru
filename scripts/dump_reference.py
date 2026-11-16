"""Dump reference inputs and outputs of the PyTorch model for the Rust parity test.

    DIZIONE_DATA=weights/phonemizer python scripts/dump_reference.py weights/tiny-voca.pt bench/sentences.txt out/reference

One safetensors file per case: tokens, stress, the exact noise torch drew,
durations, prosody, f0 and the waveform.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tiny_voca.infer import load_model, match_frontend
from tiny_voca.text import phonemize

checkpoint, sentences, target = sys.argv[1], sys.argv[2], Path(sys.argv[3])
target.mkdir(parents=True, exist_ok=True)
match_frontend(checkpoint)
model = load_model(checkpoint)
texts = [line.strip() for line in open(sentences, encoding="utf-8") if line.strip()]
settings = [dict(speed=1.0, pitch_shift=0.0, pitch_range=1.0, seed=0), dict(speed=1.3, pitch_shift=0.12, pitch_range=1.4, seed=3),
            dict(speed=0.7, pitch_shift=-0.1, pitch_range=0.6, seed=11)]
case = 0
for index, text in enumerate(texts):
    phones = phonemize(text)
    for voice in range(len(model.config.speakers)):
        for s in (settings if index < 2 else settings[:1]):
            tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
            wave, lengths, details = model.synthesize(tokens, stress, torch.tensor([len(phones)]), torch.tensor([voice]),
                                                      s["speed"], s["pitch_shift"], s["seed"], s["pitch_range"])
            samples = int(lengths[0])
            noise = torch.randn(samples, generator=torch.Generator().manual_seed(s["seed"]))
            f0 = model.frequency(details["prosody"][..., 0] + s["pitch_shift"])
            save_file({"tokens": tokens[0].int(), "stress": stress[0].int(), "noise": noise, "durations": details["durations"][0].int(),
                       "prosody": details["prosody"][0].contiguous(), "f0": f0[0].contiguous(), "wave": wave[0, :samples].contiguous()},
                      str(target / f"case{case:02d}.safetensors"),
                      metadata={"text": text, "voice": str(voice), **{k: repr(v) for k, v in s.items()}})
            case += 1
print(f"wrote {case} cases to {target}")
