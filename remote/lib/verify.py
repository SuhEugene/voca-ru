"""Corpus verification + dataset card, shared by publish.sh and the tests.

``verify`` reads the manifest, the FLAC files and ``review.jsonl`` and returns
one dict with every error and the statistics the dataset card needs.  It never
writes inside the corpus; ``write_dataset_card`` is the only function that does,
and only when publish.sh asks for it.

    python remote/lib/verify.py --corpus data/corpus-ru
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

SR = 24000
REQUIRED = ("id", "text", "audio", "seconds", "voice")
MAX_PRINTED_ERRORS = 40


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _rel_audio_paths(audio_dir):
    found = []
    for root, _dirs, files in os.walk(audio_dir):
        for name in files:
            if name.lower().endswith(".flac"):
                found.append(os.path.relpath(os.path.join(root, name), audio_dir))
    return found


def _decode_check(path, row, errors, tolerance):
    try:
        import soundfile as sf

        info = sf.info(path)
    except Exception as exc:  # unreadable file is a hard failure
        errors.append(f"{row['id']}: FLAC unreadable ({type(exc).__name__}: {exc})")
        return None
    if info.samplerate != SR:
        errors.append(f"{row['id']}: sample rate {info.samplerate} != {SR}")
    if info.channels != 1:
        errors.append(f"{row['id']}: {info.channels} channels != mono")
    if info.subtype != "PCM_16":
        errors.append(f"{row['id']}: subtype {info.subtype} != PCM_16")
    seconds = row.get("seconds")
    if not isinstance(seconds, (int, float)):
        errors.append(f"{row['id']}: manifest seconds is not a number")
    elif abs(info.duration - float(seconds)) > max(tolerance, 0.02 * float(seconds)):
        errors.append(
            f"{row['id']}: duration {info.duration:.3f}s != manifest {seconds:.3f}s"
        )
    return info.duration


def verify(corpus_dir, *, max_rows=0, check_audio=True):
    """Verify a finished corpus directory. Returns {ok, errors, stats}."""
    corpus_dir = os.path.abspath(corpus_dir)
    manifest_path = os.path.join(corpus_dir, "manifest.jsonl")
    audio_dir = os.path.join(corpus_dir, "audio")
    errors = []

    if not os.path.isfile(manifest_path):
        return {"ok": False, "errors": [f"missing {manifest_path}"], "stats": {}}
    if not os.path.isdir(audio_dir):
        return {"ok": False, "errors": [f"missing {audio_dir}"], "stats": {}}

    rows = load_jsonl(manifest_path)
    if max_rows:
        rows = rows[:max_rows]
    if not rows:
        return {"ok": False, "errors": ["manifest is empty"], "stats": {}}

    seen = set()
    wer_values = []
    stress_checked = stress_errors = 0
    voices = Counter()
    categories = Counter()
    total_seconds = 0.0
    referenced = set()

    for index, row in enumerate(rows, 1):
        missing = [key for key in REQUIRED if key not in row]
        if missing:
            errors.append(f"row {index}: missing {', '.join(missing)}")
            continue
        cid = row["id"]
        if cid in seen:
            errors.append(f"{cid}: duplicate id")
        seen.add(cid)
        rel = str(row["audio"])
        if os.path.isabs(rel) or rel.startswith(".."):
            errors.append(f"{cid}: audio path escapes the corpus: {rel}")
            continue
        if not rel.startswith("audio/"):
            errors.append(f"{cid}: audio path outside audio/: {rel}")
        path = os.path.join(corpus_dir, rel)
        referenced.add(os.path.relpath(path, audio_dir))
        if not os.path.isfile(path):
            errors.append(f"{cid}: audio file missing: {rel}")
            continue
        if row.get("voice") and not rel.startswith(f"audio/{row['voice']}/"):
            errors.append(f"{cid}: audio {rel} not under audio/{row['voice']}/")
        if check_audio:
            _decode_check(path, row, errors, 0.05)
        voices[row.get("voice", "?")] += 1
        categories[row.get("category") or "uncategorized"] += 1
        try:
            total_seconds += float(row.get("seconds") or 0.0)
        except (TypeError, ValueError):
            errors.append(f"{cid}: seconds is not numeric")
        if isinstance(row.get("wer"), (int, float)):
            wer_values.append(float(row["wer"]))
        stress_checked += int(row.get("stress_checked") or 0)
        stress_errors += int(row.get("stress_errors") or 0)

    on_disk = set(_rel_audio_paths(audio_dir))
    orphans = sorted(on_disk - referenced)
    for rel in orphans[:MAX_PRINTED_ERRORS]:
        errors.append(f"audio file without a manifest row: audio/{rel}")
    if len(orphans) > MAX_PRINTED_ERRORS:
        errors.append(f"... and {len(orphans) - MAX_PRINTED_ERRORS} more orphan FLACs")

    rejected = load_jsonl(os.path.join(corpus_dir, "review.jsonl"))
    reject_reasons = Counter()
    for row in rejected:
        for reason in row.get("reject_reasons") or ["unknown"]:
            reject_reasons[str(reason).split(" ")[0]] += 1

    wer_sorted = sorted(wer_values)
    stats = {
        "clips": len(rows),
        "hours": round(total_seconds / 3600.0, 3),
        "audio_seconds": round(total_seconds, 2),
        "voices": dict(sorted(voices.items())),
        "categories": dict(sorted(categories.items())),
        "wer_clips": len(wer_values),
        "wer_mean": round(sum(wer_values) / len(wer_values), 4) if wer_values else None,
        "wer_p95": _percentile(wer_sorted, 0.95),
        "wer_max": round(wer_sorted[-1], 4) if wer_sorted else None,
        "stress_checked": stress_checked,
        "stress_errors": stress_errors,
        "stress_error_pct": round(100.0 * stress_errors / stress_checked, 3)
        if stress_checked
        else None,
        "accepted": len(rows),
        "dropped": len(rejected),
        "drop_rate_pct": round(100.0 * len(rejected) / (len(rows) + len(rejected)), 2)
        if rows
        else None,
        "reject_reasons": dict(reject_reasons.most_common(12)),
        "sr": SR,
    }
    return {"ok": not errors, "errors": errors, "stats": stats}


def _percentile(sorted_values, q):
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return round(sorted_values[index], 4)


def _table(mapping):
    return "\n".join(f"| `{key}` | {value} |" for key, value in mapping.items()) or "| — | — |"


def dataset_card(stats, meta):
    """Dataset card for the HF repo. ``meta`` carries dates, SHAs and paths."""
    voices = stats.get("voices", {})
    voice_rows = "\n".join(
        f"| `{name}` | {count} |" for name, count in voices.items()
    ) or "| — | — |"
    categories = _table(stats.get("categories", {}))
    rejects = _table(stats.get("reject_reasons", {})) if stats.get("reject_reasons") else "| none | 0 |"
    wer_mean = stats.get("wer_mean")
    wer_p95 = stats.get("wer_p95")
    stress_pct = stats.get("stress_error_pct")
    fmt = lambda value, unit="": "n/a" if value is None else f"{value}{unit}"
    return f"""---
license: other
license_name: research-only
language:
- ru
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- audio
- russian
- synthetic
- text-to-speech
- qwen3-tts
pretty_name: voca-ru synthetic Russian speech corpus
---

# voca-ru synthetic Russian speech corpus

Russian utterances synthesised with **Qwen3-TTS-1.7B Base** plus
the **Russian stress LoRA** `{meta["adapter_repo"]}`. Two designed voices
(`f_neutral`, `m_neutral`) read every sentence of the frozen script list, so the
same text appears in both voices. This corpus is the teacher for the 4.5M-param
`voca` Russian student model; it is **not** a recording of a human speaker.

**Licence: research-only. See [Licence](#licence).**

## At a glance

| | |
|---|---|
| Clips | {stats.get("clips", 0)} |
| Audio | {stats.get("hours", 0)} h ({stats.get("audio_seconds", 0)} s) |
| Format | {stats.get("sr", 24000)} Hz mono FLAC, PCM 16-bit, peak 0.95 |
| Voices | {", ".join(voices) or "n/a"} |
| Sentences | {meta.get("sentences", "n/a")} (each read by both voices) |
| Generated | {meta.get("generated_on", "n/a")} |
| Dropped by screening | {stats.get("dropped", 0)} ({fmt(stats.get("drop_rate_pct"), " %")}) |

## Screening

Every clip was transcribed with **Whisper large-v3-turbo** and kept only if the
normalised Russian WER was ≤ 15 %, the duration ratio and speaking rate were
plausible, the clip was not silent or clipped, and audio stress detection
(`meter2`, Apache-2.0) agreed with the RUAccent stress marks in the prompt. A
failing clip was regenerated once and then dropped with a reason in
`review.jsonl` (not shipped).

| Metric | Value |
|---|---:|
| WER mean (all clips) | {fmt(wer_mean and round(100 * wer_mean, 3), " %")} |
| WER p95 | {fmt(wer_p95 and round(100 * wer_p95, 3), " %")} |
| WER max | {fmt(stats.get("wer_max") and round(100 * stats.get("wer_max"), 3), " %")} |
| Stress words checked | {stats.get("stress_checked", 0)} |
| Stress disagreements | {stats.get("stress_errors", 0)} ({fmt(stress_pct, " %")}) |

## Layout

```
manifest.jsonl          one row per accepted clip
audio/<voice>/<id>.flac
```

`manifest.jsonl` row fields: `id`, `text_id`, `text`, `norm_text` /
`stressed_text` (RUAccent marks, U+0301), `phonemes` (76-symbol F3 inventory),
`voice`, `audio`, `seconds`, `wer`, `stress_ok`, `stress_checked`,
`stress_errors`, `seed`, `attempt`, `category`, `split`, `origin`,
`source_license`, `source_url`, `teacher_model`, `teacher_adapter`.

Splits are by **sentence**, not by clip: both voices of a sentence share a
split, so validation/test stay clean for training.

### Voices

| Voice | Clips |
|---|---:|
{voice_rows}

### Categories

| Category | Clips |
|---|---:|
{categories}

## How it was made

1. Text: `data/scripts-ru/batch1.jsonl` + `batch2.jsonl`, normalised and
   stress-marked by the project frontend (`training/voca_ru/text`): RUAccent
   dictionary + small ONNX model, marks written as U+0301 after the stressed
   vowel, then eSpeak NG Russian G2P for phonemes.
2. Teacher: `{meta["teacher_repo"]}` bfloat16, batch {meta.get("batch_size", "n/a")},
   length-bucketed, one designed reference prompt per voice (reference WAV +
   exact transcript), deterministic per-batch seed.
3. Adapter: `{meta["adapter_repo"]}` LoRA (r=16, alpha=32) injected into
   `talker.model` with `peft`. **Mandatory**: Base without it misplaces stress.
4. Audio: trimmed, peak-normalised to 0.95, written as 24 kHz mono FLAC PCM16.

## Licences and provenance

- **This dataset: research-only.** Redistribution and use are limited to
  non-commercial research. The adapter's card states that part of its training
  data is unlicensed or non-commercial (Silero), which is acceptable only while
  this corpus and everything trained on it stay research-only.
- Teacher model: `{meta["teacher_repo"]}` — Apache-2.0.
- Adapter and `meter2`: `{meta["adapter_repo"]}` — Apache-2.0, with the
  research-only caveat above.
- Stress dictionary: RUAccent — MIT (used to produce text marks, not shipped).
- Text scripts: CC0 / public domain / project-own (F2); per-row provenance is in
  `origin`, `source_license` and `source_url`.
- Audio is synthetic. No human speaker's voice or personality rights are
  involved; the two voices were designed with Qwen3-TTS VoiceDesign.

## Exact model files

SHA-256 of the checkpoints used for generation:

| Asset | Repository | SHA-256 |
|---|---|---|
| Teacher weights | `{meta["teacher_repo"]}` | `{meta.get("teacher_sha", "n/a")}` |
| Speech tokenizer | `{meta["teacher_repo"]}` | `{meta.get("tokenizer_sha", "n/a")}` |
| Stress LoRA | `{meta["adapter_repo"]}` | `{meta.get("adapter_sha", "n/a")}` |
| meter2 stress head | `{meta["adapter_repo"]}` | `{meta.get("head_sha", "n/a")}` |
| meter2 CTC encoder (`jonatasgrosman/wav2vec2-large-xlsr-53-russian`) | `jonatasgrosman/wav2vec2-large-xlsr-53-russian` | `{meta.get("xlsr_sha", "n/a")}` |
| ASR (`Systran/faster-whisper-large-v3-turbo`) | `Systran/faster-whisper-large-v3-turbo` | `{meta.get("whisper_sha", "n/a")}` |

## Citation

```bibtex
@misc{{voca_ru_corpus_{meta.get("year", "")},
  title  = {{voca-ru synthetic Russian speech corpus}},
  author = {{SuhEugene}},
  year   = {{{meta.get("year", "")}}},
  note   = {{Synthetic Russian TTS corpus generated with Qwen3-TTS-1.7B Base +
            Russian stress LoRA. Research-only.}},
  url    = {{https://huggingface.co/datasets/{meta.get("repo", "")}}}
}}
```

Please also credit the teacher and adapter:
- Qwen3-TTS: <https://huggingface.co/{meta["teacher_repo"]}>
- Russian stress adapter: <https://huggingface.co/{meta["adapter_repo"]}>
- Upstream `tiny-voca` recipe: n4ze3m, <https://github.com/n4ze3m/tiny-voca>

## Limitations

- One synthetic female and one synthetic male voice; no speaker diversity.
- Teacher errors are baked in. Screening removes gross failures, not accent or
  prosody artefacts.
- Stress screening only covers words `meter2` is confident about.
- Research-only; do not use for commercial products or voice impersonation.
"""


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.corpus)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        for error in result["errors"][:MAX_PRINTED_ERRORS]:
            print("FAIL:", error)
        if len(result["errors"]) > MAX_PRINTED_ERRORS:
            print(f"FAIL: ... and {len(result['errors']) - MAX_PRINTED_ERRORS} more")
        print(json.dumps(result["stats"], ensure_ascii=False, indent=1))
        print("PASS" if result["ok"] else "FAIL")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
