"""Round-trip WER for a checkpoint on a slice of the held-out test split."""

import json
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, "training")

from voca_ru.eval import evaluate, make_backend
from voca_ru.train.config import Config
from voca_ru.train.data import AnnotatedDataset
from voca_ru.train.model import TinyVoca


def main() -> None:
    checkpoint, limit = Path(sys.argv[1]), int(sys.argv[2])
    root = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/ru-corpus")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = Config.from_dict(state["config"])
    model = TinyVoca(config).eval()
    model.load_state_dict(state["model"], strict=True)

    dataset = AnnotatedDataset(root / "annotations/test.jsonl", config,
                               acoustic_root=root / "prepared", cache=False)
    rows = dataset.rows[:limit]
    out = Path("/tmp/wer-samples")
    out.mkdir(exist_ok=True)
    audio_rows = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            for voice_index, voice in enumerate(config.speakers):
                tokens = torch.tensor([row["tokens"]])
                stress = torch.tensor([row["stress"]])
                wave, lengths, _ = model.synthesize(tokens, stress, torch.tensor([len(row["tokens"])]),
                                                    torch.tensor([voice_index]))
                path = out / f"{index:04d}-{voice}.wav"
                sf.write(path, wave[0, : int(lengths[0])].numpy(), config.sample_rate)
                audio_rows.append({"id": f"{index:04d}-{voice}", "text": row["text"], "audio": str(path)})

    report = evaluate(audio_rows, make_backend("whisper", device="cpu"))
    print(json.dumps({"clips": len(audio_rows), "wer": report["overall"]["wer"],
                      "cer": report["overall"]["cer"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
