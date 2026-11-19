"""Model and training smoke tests (run: python -m unittest discover -s tests/train)."""

import math
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

import torch  # noqa: E402

from voca_ru.train.cli import TrainingSystem, WEIGHTS, optimize_step  # noqa: E402
from voca_ru.train.config import Config  # noqa: E402
from voca_ru.train.losses import Discriminators  # noqa: E402
from voca_ru.train.model import TinyVoca  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def synthetic_batch(config, tokens=24, frames=120, rows=2):
    """Random tensors with the real shapes and dtypes the dataset produces."""
    generator = torch.Generator().manual_seed(7)
    counts = torch.full((rows, tokens), frames // tokens, dtype=torch.long)
    counts[:, 0] += frames - (frames // tokens) * tokens
    voiced = torch.randint(0, 2, (rows, frames), generator=generator).float()
    return {
        "tokens": torch.randint(1, config.vocab_size, (rows, tokens), generator=generator).to(DEVICE),
        "stress": torch.randint(0, 3, (rows, tokens), generator=generator).to(DEVICE),
        "durations": counts.to(DEVICE),
        "token_lengths": torch.full((rows,), tokens, device=DEVICE),
        "frame_lengths": torch.full((rows,), frames, device=DEVICE),
        "speakers": torch.arange(rows, device=DEVICE) % len(config.speakers),
        "wave": torch.randn(rows, frames * config.hop, generator=generator).to(DEVICE) * 0.1,
        "mel": torch.randn(rows, frames, config.n_mels, generator=generator).to(DEVICE) * 0.1,
        "prosody": torch.stack((torch.randn(rows, frames, generator=generator) * 0.2,
                                voiced, torch.randn(rows, frames, generator=generator) * 0.2), dim=-1).to(DEVICE),
        "pitch_trusted": voiced.clone().to(DEVICE),
    }


class ModelTest(unittest.TestCase):
    def test_parameter_count(self):
        count = sum(p.numel() for p in TinyVoca(Config()).parameters())
        self.assertEqual(count, 4_537_291)
        self.assertTrue(4_300_000 <= count <= 4_700_000)

    def test_shapes(self):
        config = Config()
        model = TinyVoca(config).to(DEVICE)
        batch = synthetic_batch(config)
        with torch.inference_mode():
            hidden, mask = model.encode(batch["tokens"], batch["stress"], batch["token_lengths"], batch["speakers"])
            self.assertEqual(hidden.shape, (2, 24, config.channels))
            self.assertEqual(model.duration(hidden, mask).shape, (2, 24, 2))
            expanded, frame_mask, lengths = model.expand(hidden, batch["durations"], mask, batch["speakers"])
            self.assertEqual(expanded.shape, (2, 120, config.channels))
            self.assertTrue(torch.equal(lengths, batch["frame_lengths"]))
            _, prosody = model.predict_prosody(expanded, frame_mask)
            self.assertEqual(prosody.shape, (2, 120, 3))
            acoustic = model.decode(expanded, model.hard_prosody(prosody), frame_mask)
            wave = model.vocoder(acoustic, model.frequency(prosody[..., 0]), lengths)
            self.assertEqual(wave.shape, (2, 120 * config.hop))
            self.assertTrue(torch.isfinite(wave).all())

    def test_checkpoint_round_trip(self):
        config = Config()
        source = TinyVoca(config)
        weights = {key: value.clone() for key, value in source.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "inference.pt"
            torch.save({"format_version": 4, "config": config.to_dict(), "model": weights}, path)
            state = torch.load(path, map_location="cpu", weights_only=True)
            target = TinyVoca(Config.from_dict(state["config"]))
            target.load_state_dict(state["model"])
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)


class ConfigTest(unittest.TestCase):
    def test_configs(self):
        from voca_ru.train.cli import load_config
        config, base = load_config(ROOT / "configs" / "base.yaml")
        self.assertEqual(config.vocab_size, 76)
        self.assertEqual(base["steps"], 104_000)
        self.assertEqual(base["gan_start"], 8_000)
        self.assertEqual(base["gan_ratio"], 0.7)
        _, smoke = load_config(ROOT / "configs" / "smoke.yaml")
        self.assertLessEqual(smoke["steps"], 100)


class PrepareTest(unittest.TestCase):
    def test_extract_shapes(self):
        import numpy as np
        from voca_ru.train.prepare import extract
        config = Config()
        time = np.arange(int(1.2 * config.sample_rate)) / config.sample_rate
        wave = (0.3 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)
        item = extract(wave, config)
        frames = math.ceil(len(wave) / config.hop)
        self.assertEqual(tuple(item["wave"].shape), (frames * config.hop,))
        self.assertEqual(tuple(item["mel"].shape), (frames, config.n_mels))
        self.assertEqual(tuple(item["prosody"].shape), (frames, 3))


class TrainingTest(unittest.TestCase):
    def test_three_steps(self):
        config = Config()
        system = TrainingSystem(config).to(DEVICE)
        critics = Discriminators().to(DEVICE)
        optimizer = torch.optim.AdamW(system.parameters(), lr=3e-4, betas=(0.8, 0.99), weight_decay=0.01)
        critic_optimizer = torch.optim.AdamW(critics.parameters(), lr=3e-4, betas=(0.8, 0.99))
        recipe = {
            "weights": WEIGHTS, "learning_rate": 3e-4, "warmup": 1, "lr_half_life": 100000,
            "teacher_hold": 0, "teacher_ramp": 1, "teacher_final": 0.0,
            "gan_start": 2, "gan_ratio": 0.7, "gan_ramp": 1, "feature_matching": 30.0,
            "gan_scale_max": 2000.0, "crop_samples": 4096, "crops": 1, "grad_clip": 10.0,
            "branched_from": None,
        }
        batch = synthetic_batch(config)
        losses = []
        for step in range(3):
            metrics = optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe, amp=True)
            losses.append(metrics["loss"])
            self.assertTrue(math.isfinite(metrics["loss"]), metrics)
        self.assertEqual(len(losses), 3)
        # 3 steps include one adversarial step (gan_start=2)
        self.assertIn("gan_scale", metrics)


if __name__ == "__main__":
    unittest.main()
