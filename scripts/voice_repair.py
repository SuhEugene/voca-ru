"""Re-clone one candidate with new seeds until it produces real audio.

VoiceDesign sometimes emits a reference the teacher cannot clone (a silent
0.1-0.2 s take). This retries the clone path with the next seed and keeps the
first take whose RMS is above -40 dBFS, so the folder always holds 12 usable
voices.

    .cache/venvs/r4/bin/python scripts/voice_repair.py fb01_soft_light [out_dir]
"""

import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, ".cache/models")
MERGED = os.path.join(CACHE, "qwen3-tts-1.7b-ru-stress-merged")
LANG = "Russian"
CLONE_TEXT = "Мне очень нравятся большие... контекстные окна. Ты! Видел это? Нравится"
FLOOR = 0.01  # RMS below this is silence, not speech


def main() -> None:
    name = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "out/voice-femboy")
    candidates = {row["name"]: row for row in json.load(open(os.path.join(out, "candidates.json")))}
    voice = candidates[name]
    reference = os.path.join(out, voice["ref"])
    reference_text = open(os.path.join(out, f"{name}.txt"), encoding="utf-8").read().strip()

    from qwen_tts import Qwen3TTSModel

    teacher = Qwen3TTSModel.from_pretrained(MERGED, device_map="mps", dtype=torch.bfloat16,
                                            attn_implementation="sdpa")
    prompt = teacher.create_voice_clone_prompt(ref_audio=reference, ref_text=reference_text,
                                               x_vector_only_mode=False)
    for attempt, seed in enumerate(range(voice["seed"], voice["seed"] + 6)):
        torch.manual_seed(seed)
        wavs, rate = teacher.generate_voice_clone(text=[CLONE_TEXT], language=[LANG],
                                                  voice_clone_prompt=prompt)
        audio = wavs[0]
        rms = float(np.sqrt((np.asarray(audio) ** 2).mean()))
        seconds = len(audio) / rate
        print(f"seed {seed}: {seconds:.2f} s rms {rms:.3f}", flush=True)
        if seconds > 1.0 and rms >= FLOOR:
            sf.write(os.path.join(out, voice["clone"]), audio, rate)
            voice["seed"] = seed
            voice["seconds"] = round(seconds, 2)
            json.dump(list(candidates.values()), open(os.path.join(out, "candidates.json"), "w"),
                      ensure_ascii=False, indent=1)
            print(f"kept seed {seed} for {name}")
            return
    raise SystemExit(f"no usable take for {name}")


if __name__ == "__main__":
    main()
