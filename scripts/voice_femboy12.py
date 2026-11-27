"""Twelve soft androgynous ("femboy") candidates: VoiceDesign invents them, the
merged Russian teacher clones each one the way the corpus would be generated.

    .cache/venvs/r4/bin/python scripts/voice_femboy12.py [out_dir]
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

# The line the user asked for; the same one for every candidate.
CLONE_TEXT = "Мне очень нравятся большие... контекстные окна. Ты! Видел это? Нравится"

BASE = ("A soft, gentle young man's voice, Russian native speaker, light and "
        "airy rather than deep; warm, shy and friendly, slightly feminine timbre "
        "without sounding like a child or a woman; clear articulation, "
        "conversational pace, close to the microphone. ")

VOICES = [
    {"name": "fb01_soft_light", "seed": 8101,
     "short": "мягкий светлый, почти шёпот",
     "instruct": BASE + "He speaks quietly and gently, with a lot of breath in the voice."},
    {"name": "fb02_airy_youth", "seed": 8102,
     "short": "воздушный юношеский",
     "instruct": BASE + "Youthful and airy, as if smiling while talking, very light delivery."},
    {"name": "fb03_shy", "seed": 8103,
     "short": "застенчивый, тёплый",
     "instruct": BASE + "Shy and warm, hesitating slightly at commas, kind and disarming."},
    {"name": "fb04_androgynous", "seed": 8104,
     "short": "андрогинный, нейтральный",
     "instruct": BASE + "Androgynous: neither clearly male nor female to the listener, "
                        "smooth and even, calm."},
    {"name": "fb05_velvet_high", "seed": 8105,
     "short": "бархатный, но высокий",
     "instruct": BASE + "A high, velvet timbre with soft edges, unhurried and confiding."},
    {"name": "fb06_playful", "seed": 8106,
     "short": "игривый, с улыбкой",
     "instruct": BASE + "Playful and smiling, a little teasing, still soft and gentle."},
    {"name": "fb07_whisper_close", "seed": 8107,
     "short": "вполголоса у самого уха",
     "instruct": BASE + "Almost a whisper, very close to the microphone, intimate and calm."},
    {"name": "fb08_gentle_smile", "seed": 8108,
     "short": "нежный, улыбчивый",
     "instruct": BASE + "Tender and smiling, warm low-volume delivery, no sharp consonants."},
    {"name": "fb09_clear_soft", "seed": 8109,
     "short": "чёткий, но мягкий",
     "instruct": BASE + "Clean, well-articulated but soft, like a careful audiobook reader."},
    {"name": "fb10_dreamy", "seed": 8110,
     "short": "мечтательный, медленный",
     "instruct": BASE + "Dreamy and slow, slightly detached, gentle melodic intonation."},
    {"name": "fb11_bright_kid", "seed": 8111,
     "short": "звонкий, но не детский",
     "instruct": BASE + "Brighter and lighter than average, energetic yet soft; explicitly "
                        "an adult, not a child."},
    {"name": "fb12_low_soft", "seed": 8112,
     "short": "низкий, но нежный",
     "instruct": BASE + "Lower than the others but still soft and tender, calm and grounded."},
]

REFERENCE_TEXT = (
    "Осенью лес меняется до неузнаваемости: листья становятся золотыми, воздух "
    "пахнет сыростью, а по утрам над рекой стоит густой туман."
)


def load(path):
    from qwen_tts import Qwen3TTSModel

    started = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        path, device_map="mps", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    print(f"[load] {os.path.basename(path)} {time.perf_counter() - started:.1f} s", flush=True)
    return model


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "out/voice-femboy")
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

    with open(os.path.join(out, "candidates.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
