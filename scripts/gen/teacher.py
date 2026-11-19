"""G1 teacher gate: Qwen3-TTS on MPS (Apple M5 Pro).

Subcommands
  smoke   Base checkpoint, transcript-free clone of clone_ref.wav, 6 Russian
          sentences from batch1 -> out/g1/base/<id>.wav + metrics.json
  design  VoiceDesign checkpoint, 4 candidate voices -> out/g1/voices/
  batch   Base checkpoint, 8-clip batch with a designed reference

Run with the repo venv:
  . scripts/cache-env.sh
  .cache/venvs/r4/bin/python scripts/gen/teacher.py smoke

All paths are repo-relative; audio is written under out/ (git-ignored).
"""
import json
import os
import re
import resource
import sys
import time

import numpy as np
import soundfile as sf
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# The cache is shared by every worktree and lives at the main repo root.
# `scripts/cache-env.sh` (git rev-parse --show-toplevel) would point it at the
# worktree, so resolve through the common git dir instead.
_git = os.popen(f"git -C {REPO} rev-parse --path-format=absolute --git-common-dir").read().strip()
SHARED = os.path.dirname(os.path.abspath(_git)) if _git else REPO
CACHE = os.path.join(SHARED, ".cache/models")
BASE_DIR = os.path.join(CACHE, "qwen3-tts-1.7b")
VD_DIR = os.path.join(CACHE, "qwen3-tts-1.7b-voicedesign")
REF_WAV = os.path.join(BASE_DIR, "clone_ref.wav")
OUT = os.path.join(REPO, "out/g1")
LANG = "Russian"
SEED = 1234


def load(model_dir):
    from qwen_tts import Qwen3TTSModel

    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        model_dir, device_map="mps", dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    print(f"[load] {os.path.basename(model_dir)} {time.perf_counter()-t0:.1f} s", flush=True)
    mem("after load")
    return model


def mem(tag):
    # macOS reports ru_maxrss in bytes, Linux in KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2
    print(
        f"[mem] {tag}: peak RSS {rss:.2f} GiB, "
        f"MPS driver {torch.mps.driver_allocated_memory()/1024**3:.2f} GiB",
        flush=True,
    )


def audio_stats(wav, sr):
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(a**2) + 1e-12))
    peak = float(np.abs(a).max())
    # longest all-quiet run, as a share of the clip
    quiet = np.abs(a) < 0.01
    longest = run = 0
    for q in quiet:
        run = run + 1 if q else 0
        longest = max(longest, run)
    return {
        "duration_s": round(len(a) / sr, 3),
        "peak": round(peak, 4),
        "rms": round(rms, 5),
        "silence_share": round(longest / max(len(a), 1), 3),
        "clipped": bool(peak >= 0.999),
    }


def save(wav, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.asarray(wav, dtype=np.float32), sr, subtype="PCM_16")


def batch1(n=8000):
    rows = []
    with open(os.path.join(REPO, "data/scripts-ru/batch1.jsonl")) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


# The six required categories. batch1 is digit-free by design (numbers are
# spelled out), so "number" means a spelled-out numeral.
SMOKE = [
    ("number", "Он приехал на встречу на десять минут раньше."),
    ("date", "Мы уедем в отпуск в конце июля."),
    ("question", "Как бы в такой ситуации поступила Организация Объединенных Наций?"),
    ("long", "Каждый год приезжал я в деревню на несколько недель и с каждым годом "
             "находил моего отца все более и более грустным, в себя углубленным, "
             "задумчивым до робости."),
    ("name", "Ну, Котик, сегодня ты играла, как никогда, — сказал Иван Петрович "
             "со слезами на глазах, когда его дочь кончила и встала."),
    ("plain", "Мой коллега уходит в отпуск на две недели."),
]


def cmd_smoke():
    model = load(BASE_DIR)
    t0 = time.perf_counter()
    prompt = model.create_voice_clone_prompt(ref_audio=REF_WAV, x_vector_only_mode=True)
    print(f"[prompt] x_vector_only, {time.perf_counter()-t0:.2f} s", flush=True)

    rows = batch1()
    clips = []
    for i, (cat, text) in enumerate(SMOKE):
        ids = [r["id"] for r in rows if r["text"] == text]
        torch.manual_seed(SEED + i)
        t0 = time.perf_counter()
        wavs, sr = model.generate_voice_clone(
            text=text, language=LANG, voice_clone_prompt=prompt,
        )
        dt = time.perf_counter() - t0
        wav = wavs[0]
        stats = audio_stats(wav, sr)
        chars = len(text)
        expected = chars / 14.0  # ~14 chars/s is normal Russian narration
        rec = {
            "category": cat, "text": text, "text_id": ids[0] if ids else None,
            "seed": SEED + i, "chars": chars, "wall_s": round(dt, 3),
            "chars_per_s": round(chars / dt, 2),
            "rtf": round(dt / stats["duration_s"], 3),
            "expected_s": round(expected, 2),
            "duration_ratio": round(stats["duration_s"] / expected, 2),
            **stats,
        }
        path = os.path.join(OUT, "base", f"{i}_{cat}.wav")
        save(wav, sr, path)
        rec["file"] = os.path.relpath(path, REPO)
        clips.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        mem(f"after clip {i} ({cat})")

    total_audio = sum(c["duration_s"] for c in clips)
    total_wall = sum(c["wall_s"] for c in clips)
    summary = {
        "device": "mps", "dtype": "bfloat16", "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "reference": os.path.relpath(REF_WAV, REPO), "ref_mode": "x_vector_only",
        "clips": clips, "sr": sr,
        "total_audio_s": round(total_audio, 2), "total_wall_s": round(total_wall, 2),
        "rtf": round(total_wall / total_audio, 3),
        "x_realtime": round(total_audio / total_wall, 2),
        "peak_mps_gib": round(torch.mps.driver_allocated_memory() / 1024**3, 2),
    }
    with open(os.path.join(OUT, "base-metrics.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("[summary]", json.dumps({k: v for k, v in summary.items() if k != "clips"}),
          flush=True)


# --- VoiceDesign -----------------------------------------------------------

VOICES = [
    {
        "name": "f_warm",
        "gender": "female",
        "seed": 101,
        "instruct": "A warm, natural adult female voice in her thirties speaking "
                    "Russian as a native speaker, with no foreign accent. Calm, "
                    "friendly and clear, standard Russian pronunciation, moderate "
                    "pace, gentle intonation, studio quality.",
        "text": "Знаешь, иногда мне кажется, что самое важное в жизни — это просто "
                "остановиться и внимательно послушать тишину вокруг себя.",
    },
    {
        "name": "f_neutral",
        "gender": "female",
        "seed": 202,
        "instruct": "A clear, neutral adult female voice in her forties speaking "
                    "Russian as a native speaker, with no accent. Professional, "
                    "even and articulate, like a careful newsreader; standard "
                    "pronunciation, steady pace, studio quality.",
        "text": "Летним утром город просыпается медленно: открываются кафе, "
                "дворники поливают цветы, а на остановке собираются первые "
                "пассажиры.",
    },
    {
        "name": "m_warm",
        "gender": "male",
        "seed": 303,
        "instruct": "A warm, natural adult male voice in his thirties speaking "
                    "Russian as a native speaker, with no foreign accent. Friendly "
                    "and relaxed, conversational, standard Russian pronunciation, "
                    "moderate pace, studio quality.",
        "text": "Когда я вернулся домой после долгой дороги, первое, что я "
                "почувствовал, — это запах чая и спокойный голос мамы на кухне.",
    },
    {
        "name": "m_neutral",
        "gender": "male",
        "seed": 404,
        "instruct": "A clear, neutral adult male voice in his forties speaking "
                    "Russian as a native speaker, with no accent. Professional and "
                    "steady, like an audiobook narrator; standard pronunciation, "
                    "measured pace, studio quality.",
        "text": "Осенью лес меняется до неузнаваемости: листья становятся "
                "золотыми, воздух пахнет сыростью, а по утрам над рекой стоит "
                "густой туман.",
    },
]


def cmd_design():
    model = load(VD_DIR)
    os.makedirs(os.path.join(OUT, "voices"), exist_ok=True)
    table = []
    for v in VOICES:
        torch.manual_seed(v["seed"])
        t0 = time.perf_counter()
        wavs, sr = model.generate_voice_design(
            text=v["text"], instruct=v["instruct"], language=LANG,
        )
        dt = time.perf_counter() - t0
        wav = wavs[0]
        stats = audio_stats(wav, sr)
        path = os.path.join(OUT, "voices", v["name"] + ".wav")
        save(wav, sr, path)
        with open(os.path.join(OUT, "voices", v["name"] + ".txt"), "w") as f:
            f.write(v["text"] + "\n")
        rec = {**v, **stats, "sr": sr, "wall_s": round(dt, 3),
               "chars_per_s": round(len(v["text"]) / dt, 2),
               "rtf": round(dt / stats["duration_s"], 3),
               "file": os.path.relpath(path, REPO)}
        table.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        mem(f"after {v['name']}")
    with open(os.path.join(OUT, "voices-metrics.json"), "w") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)


def cmd_batch():
    name = sys.argv[2] if len(sys.argv) > 2 else "f_warm"
    mode = sys.argv[3] if len(sys.argv) > 3 else "as-is"
    wav = os.path.join(OUT, "voices", name + ".wav")
    txt = os.path.join(OUT, "voices", name + ".txt")
    ref_text = open(txt).read().strip()
    model = load(BASE_DIR)
    t0 = time.perf_counter()
    prompt = model.create_voice_clone_prompt(
        ref_audio=wav, ref_text=ref_text, x_vector_only_mode=False,
    )
    print(f"[prompt] {name} with transcript, {time.perf_counter()-t0:.2f} s", flush=True)

    rows = batch1()
    if mode == "uniform":
        # Length-bucketed batch: no clip is padded to a 179-char outlier.
        cand = [r["text"] for r in rows if 55 <= len(r["text"]) <= 75]
        texts = cand[:8]
    else:
        texts = [r["text"] for r in rows[:8]]
    torch.manual_seed(SEED)
    runs = {}
    wavs = sr = None
    for tag in ("cold", "warm"):
        t0 = time.perf_counter()
        wavs, sr = model.generate_voice_clone(
            text=texts, language=[LANG] * len(texts), voice_clone_prompt=prompt,
        )
        runs[tag] = time.perf_counter() - t0
    dt = runs["cold"]
    recs = []
    for i, (w, txt_i) in enumerate(zip(wavs, texts)):
        stats = audio_stats(w, sr)
        path = os.path.join(OUT, "batch8" if mode == "as-is" else "batch8-uniform", f"{i}.wav")
        save(w, sr, path)
        recs.append({"i": i, "text": txt_i, "chars": len(txt_i),
                     "wall_s": round(dt / len(texts), 2), **stats})
    audio = sum(r["duration_s"] for r in recs)
    summary = {
        "voice": name, "batch": len(texts), "mode": mode, "device": "mps", "dtype": "bfloat16",
        "total_audio_s": round(audio, 2),
        "wall_cold_s": round(dt, 2), "wall_warm_s": round(runs["warm"], 2),
        "rtf_cold": round(dt / audio, 3), "rtf_warm": round(runs["warm"] / audio, 3),
        "x_realtime_cold": round(audio / dt, 2),
        "x_realtime_warm": round(audio / runs["warm"], 2),
        "chars_per_s_cold": round(sum(r["chars"] for r in recs) / dt, 2),
        "s_per_clip": round(dt / len(texts), 2),
        "peak_mps_gib": round(torch.mps.driver_allocated_memory() / 1024**3, 2),
        "clips": recs,
    }
    with open(os.path.join(OUT, "batch8-metrics.json" if mode == "as-is" else "batch8-uniform-metrics.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("[batch]", json.dumps({k: v for k, v in summary.items() if k != "clips"}),
          flush=True)


if __name__ == "__main__":
    {"smoke": cmd_smoke, "design": cmd_design, "batch": cmd_batch}[sys.argv[1]]()
