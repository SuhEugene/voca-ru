"""Seven lower-pitched takes on the f_storyteller character.

Same VoiceDesign paragraph and the same cloning path as voice_candidates.py; the
instructions below only pull the register down, in steps, so the choice is about
how low to go rather than about a different character.

    .cache/venvs/r4/bin/python scripts/voice_variants.py [out_dir]
"""

import json
import os
import sys
import time

import soundfile as sf
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(REPO, ".cache/models")
VD_DIR = os.path.join(CACHE, "qwen3-tts-1.7b-voicedesign")
MERGED = os.path.join(CACHE, "qwen3-tts-1.7b-ru-stress-merged")
LANG = "Russian"

REFERENCE_TEXT = (
    "Осенью лес меняется до неузнаваемости: листья становятся золотыми, воздух "
    "пахнет сыростью, а по утрам над рекой стоит густой туман."
)
CLONE_TEXT = "Привет! Мне нравится эта маленькая книга — правда, щиплет глаза?"

BASE = ("A friendly adult female voice, Russian native speaker, telling a story to "
        "one listener: warm and smiling, lively but tasteful intonation, soft "
        "consonants, conversational pace, natural rather than theatrical. ")

VOICES = [
    {
        "name": "s1_slightly_lower",
        "seed": 7001,
        "short": "чуть ниже оригинала",
        "instruct": BASE + "Her pitch sits slightly lower than usual for a woman, "
                    "in a warm chest register. Studio quality.",
    },
    {
        "name": "s2_alto",
        "seed": 7002,
        "short": "альт, тёплый средний-низкий",
        "instruct": BASE + "A warm alto: medium-low female pitch, smooth and rounded, "
                    "no breathy edges. Studio quality.",
    },
    {
        "name": "s3_lower_register",
        "seed": 7003,
        "short": "низкий регистр, грудной",
        "instruct": BASE + "Clearly low-pitched for a woman, with strong chest "
                    "resonance and a soft, grounded delivery. Studio quality.",
    },
    {
        "name": "s4_velvet",
        "seed": 7004,
        "short": "бархатный низкий",
        "instruct": BASE + "A low, velvet voice with a slight smoky edge, unhurried "
                    "and confiding, like reading aloud late in the evening. Studio quality.",
    },
    {
        "name": "s5_radio_low",
        "seed": 7005,
        "short": "ночной радиоэфир, низкий",
        "instruct": BASE + "A late-night radio storyteller: low warm voice, close to "
                    "the microphone, calm and steady, almost no high frequencies. Studio quality.",
    },
    {
        "name": "s6_gentle_low",
        "seed": 7006,
        "short": "мягкий низкий, с улыбкой",
        "instruct": BASE + "Gentle and soft-spoken with a low pitch, a smile in the "
                    "voice, very kind and unhurried. Studio quality.",
    },
    {
        "name": "s7_mature_low",
        "seed": 7007,
        "short": "взрослый низкий, ~40 лет",
        "instruct": BASE + "A mature woman around forty with a low, calm, well-placed "
                    "voice; assured and warm, slightly slower than average. Studio quality.",
    },
]


def load(path):
    from qwen_tts import Qwen3TTSModel

    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        path, device_map="mps", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    print(f"[load] {os.path.basename(path)} {time.perf_counter() - started:.1f} s", flush=True)
    return model


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "out/voice-variants")
    os.makedirs(out, exist_ok=True)

    designer = load(VD_DIR)
    references = {}
    for voice in VOICES:
        torch.manual_seed(voice["seed"])
        wavs, rate = designer.generate_voice_design(
            text=REFERENCE_TEXT, instruct=voice["instruct"], language=LANG,
        )
        references[voice["name"]] = os.path.join(out, f"{voice['name']}.ref.wav")
        sf.write(references[voice["name"]], wavs[0], rate)
        with open(os.path.join(out, f"{voice['name']}.txt"), "w", encoding="utf-8") as file:
            file.write(REFERENCE_TEXT + "\n")
        print(f"[design] {voice['name']} {len(wavs[0]) / rate:.2f} s", flush=True)

    del designer
    torch.mps.empty_cache()

    teacher = load(MERGED)
    summary = []
    for voice in VOICES:
        prompt = teacher.create_voice_clone_prompt(
            ref_audio=references[voice["name"]], ref_text=REFERENCE_TEXT,
            x_vector_only_mode=False,
        )
        torch.manual_seed(voice["seed"])
        wavs, rate = teacher.generate_voice_clone(
            text=[CLONE_TEXT], language=[LANG], voice_clone_prompt=prompt,
        )
        clone = os.path.join(out, f"{voice['name']}.clone.wav")
        sf.write(clone, wavs[0], rate)
        summary.append({**voice, "ref": os.path.basename(references[voice["name"]]),
                        "clone": os.path.basename(clone),
                        "seconds": round(len(wavs[0]) / rate, 2)})
        print(f"[clone] {voice['name']} {summary[-1]['seconds']} s", flush=True)

    with open(os.path.join(out, "variants.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
