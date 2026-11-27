"""Five female narrator candidates: VoiceDesign invents them, the merged Russian
teacher (Base + stress LoRA) clones them the way the corpus is generated.

    .cache/venvs/r4/bin/python scripts/voice_candidates.py [out_dir]

Writes <name>.ref.wav (the VoiceDesign reference), <name>.clone.wav (the same
voice through the cloning path the corpus uses) and README.md.
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

# The paragraph the designer reads. The same text for all five keeps the
# comparison to timbre alone.
REFERENCE_TEXT = (
    "Осенью лес меняется до неузнаваемости: листья становятся золотыми, воздух "
    "пахнет сыростью, а по утрам над рекой стоит густой туман."
)
# What every candidate is then cloned onto: a plain sentence and a question,
# with ы, щ and a soft рь.
CLONE_TEXT = "Привет! Мне нравится эта маленькая книга — правда, щиплет глаза?"

VOICES = [
    {
        "name": "f_audiobook",
        "seed": 1101,
        "short": "тёплая дикторская, аудиокниги",
        "instruct": "A warm, expressive adult female voice in her thirties, Russian "
                    "native speaker with no accent. Audiobook narrator: low, soft "
                    "chest register, unhurried measured pace, gentle breathiness, "
                    "long phrase endings that trail off slightly. Intimate, confiding, "
                    "never sharp; studio recording, no reverb.",
    },
    {
        "name": "f_contralto",
        "seed": 2202,
        "short": "низкий контральто, спокойствие",
        "instruct": "A mature female contralto around forty-five, Russian native "
                    "speaker with no accent. Deep, velvety and calm, like a late-night "
                    "radio host reading prose: slow steady rhythm, dark timbre, very "
                    "little pitch movement, warm and grounded. Studio quality.",
    },
    {
        "name": "f_storyteller",
        "seed": 3303,
        "short": "сказочница, живая интонация",
        "instruct": "A friendly adult female voice in her thirties, Russian native "
                    "speaker, telling a story to one listener. Warm and smiling, with "
                    "lively but tasteful intonation, slight melodic rise at commas, "
                    "soft consonants, conversational pace. Natural, not theatrical.",
    },
    {
        "name": "f_soft",
        "seed": 4404,
        "short": "тихая, вполголоса, близко к микрофону",
        "instruct": "A soft-spoken young adult female voice, Russian native speaker "
                    "with no accent, recorded close to the microphone. Quiet and "
                    "breathy, almost whispered warmth, very gentle, intimate ASMR-like "
                    "delivery, clear articulation but low volume. Studio quality.",
    },
    {
        "name": "f_clear_warm",
        "seed": 5505,
        "short": "чёткая, но тёплая — золотая середина",
        "instruct": "A clear yet warm adult female voice in her thirties, Russian "
                    "native speaker with no accent. Professional voice actor for "
                    "documentaries: well-articulated, even and precise, but with a "
                    "soft warm edge and friendly tone; steady moderate pace, bright "
                    "but not sharp. Studio quality.",
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
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "out/voice-candidates")
    os.makedirs(out, exist_ok=True)

    designer = load(VD_DIR)
    references = {}
    for voice in VOICES:
        torch.manual_seed(voice["seed"])
        started = time.perf_counter()
        wavs, rate = designer.generate_voice_design(
            text=REFERENCE_TEXT, instruct=voice["instruct"], language=LANG,
        )
        path = os.path.join(out, f"{voice['name']}.ref.wav")
        sf.write(path, wavs[0], rate)
        text_path = os.path.join(out, f"{voice['name']}.txt")
        with open(text_path, "w", encoding="utf-8") as file:
            file.write(REFERENCE_TEXT + "\n")
        references[voice["name"]] = path
        print(f"[design] {voice['name']} {len(wavs[0]) / rate:.2f} s "
              f"in {time.perf_counter() - started:.1f} s", flush=True)

    del designer
    torch.mps.empty_cache()

    teacher = load(MERGED)
    summary = []
    for voice in VOICES:
        prompt = teacher.create_voice_clone_prompt(
            ref_audio=references[voice["name"]],
            ref_text=REFERENCE_TEXT,
            x_vector_only_mode=False,
        )
        torch.manual_seed(voice["seed"])
        started = time.perf_counter()
        wavs, rate = teacher.generate_voice_clone(
            text=[CLONE_TEXT], language=[LANG], voice_clone_prompt=prompt,
        )
        path = os.path.join(out, f"{voice['name']}.clone.wav")
        sf.write(path, wavs[0], rate)
        summary.append({**voice, "ref": os.path.basename(references[voice["name"]]),
                        "clone": os.path.basename(path),
                        "clone_seconds": round(len(wavs[0]) / rate, 2),
                        "clone_wall_s": round(time.perf_counter() - started, 2)})
        print(f"[clone] {voice['name']} {summary[-1]['clone_seconds']} s", flush=True)

    with open(os.path.join(out, "candidates.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=1)

    lines = [
        "# Пять женских голосов-кандидатов",
        "",
        "Слева направо по списку — от тёплого дикторского к мягкому вполголоса.",
        "Для каждого голоса два файла:",
        "",
        "- `*.ref.wav` — как голос придумал VoiceDesign (эталон для клонирования)",
        "- `*.clone.wav` — та же фраза, но через тот путь, которым генерируется корпус:",
        f"  «{CLONE_TEXT}»",
        "",
        "Слушать надо `*.clone.wav`: именно этот голос попадёт в модель.",
        "",
        "| голос | характер |",
        "|---|---|",
    ]
    for voice in VOICES:
        lines.append(f"| `{voice['name']}` | {voice['short']} |")
    lines += ["", "## Инструкции, которыми они заданы", ""]
    for voice in VOICES:
        lines.append(f"**{voice['name']}** — {voice['instruct']}")
        lines.append("")
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
