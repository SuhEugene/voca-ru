"""Evaluation runner: manifest -> ASR -> WER/CER + timing -> JSON/Markdown."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .asr import ASRBackend, timed_transcribe
from .audio import audio_info
from .metrics import cer, edit_counts
from .scoring import score_text

CATEGORIES = ("number", "date", "money", "name", "question",
              "reading", "conversation", "template", "uncategorized")


def load_manifest(path: str | Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(Path(path).read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not {"id", "text", "audio"} <= row.keys():
            raise ValueError(f"{path}:{lineno}: row needs id, text, audio")
        rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def corpus_index(paths: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    by_id: dict[str, str] = {}
    by_text: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "category" not in row:
                continue
            by_id[row["id"]] = row["category"]
            by_text[score_text(row["text"])] = row["category"]
    return by_id, by_text


def resolve_category(row: dict, by_id: dict[str, str], by_text: dict[str, str]) -> str:
    if row.get("category"):
        return row["category"]
    row_id = str(row["id"])
    for key in (row_id, row_id.rsplit("_", 1)[0]):
        if key in by_id:
            return by_id[key]
    return by_text.get(score_text(row["text"]), "uncategorized")


def evaluate(
    rows: list[dict],
    backend: ASRBackend,
    audio_dir: str | Path | None = None,
    corpora: list[Path] | None = None,
) -> dict:
    audio_dir = Path(audio_dir) if audio_dir else None
    by_id, by_text = corpus_index(corpora or [])
    out_rows = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        audio = Path(row["audio"])
        if not audio.is_absolute() and audio_dir is not None:
            audio = audio_dir / audio
        if not audio.is_file():
            raise FileNotFoundError(f"audio not found for {row['id']}: {audio}")
        duration, rate = audio_info(audio)
        text, asr_seconds = timed_transcribe(backend, str(audio))
        errors, ref_len, subs, ins = edit_counts(row["text"], text)
        out_rows.append({
            "id": row["id"],
            "category": resolve_category(row, by_id, by_text),
            "text": row["text"],
            "asr": text,
            "audio": str(audio),
            "audio_seconds": round(duration, 4),
            "sample_rate": rate,
            "asr_seconds": round(asr_seconds, 4),
            "errors": errors,
            "ref_words": ref_len,
            "substitutions": subs,
            "insertions": ins,
            "wer": errors / ref_len if ref_len else float(bool(errors)),
            "cer": cer(row["text"], text),
            "synth_seconds": row.get("synth_seconds"),
            "first_audio_seconds": row.get("first_audio_seconds"),
        })
        print(f"[{index}/{len(rows)}] {row['id']}: WER {out_rows[-1]['wer']:.1%}", flush=True)
    wall = time.perf_counter() - started

    total_errors = sum(r["errors"] for r in out_rows)
    total_words = sum(r["ref_words"] for r in out_rows)
    total_audio = sum(r["audio_seconds"] for r in out_rows)
    total_asr = sum(r["asr_seconds"] for r in out_rows)
    overall = {
        "rows": len(out_rows),
        "audio_seconds": round(total_audio, 3),
        "asr_seconds": round(total_asr, 3),
        "wer": total_errors / total_words if total_words else None,
        "cer": _mean(r["cer"] for r in out_rows),
        "asr_rtf": total_asr / total_audio if total_audio else None,
        "asr_speed_x": total_audio / total_asr if total_asr else None,
        "wall_seconds": round(wall, 3),
        "sample_rates": sorted({r["sample_rate"] for r in out_rows}),
    }
    categories = {}
    for name in CATEGORIES:
        rows_c = [r for r in out_rows if r["category"] == name]
        if not rows_c:
            continue
        errs = sum(r["errors"] for r in rows_c)
        words = sum(r["ref_words"] for r in rows_c)
        categories[name] = {
            "rows": len(rows_c),
            "wer": errs / words if words else None,
            "cer": _mean(r["cer"] for r in rows_c),
        }
    synth = [r for r in out_rows
             if r["synth_seconds"] is not None and r["first_audio_seconds"] is not None]
    synthesis = None
    if synth:
        synth_seconds = sum(r["synth_seconds"] for r in synth)
        firsts = [r["first_audio_seconds"] for r in synth]
        synthesis = {
            "rows": len(synth),
            "synth_seconds": round(synth_seconds, 3),
            "rtf": synth_seconds / total_audio if total_audio else None,
            "first_audio_seconds_mean": round(statistics.mean(firsts), 4),
            "first_audio_seconds_p95": round(_p95(firsts), 4),
            "first_audio_seconds_max": round(max(firsts), 4),
        }
    return {
        "overall": overall,
        "categories": categories,
        "synthesis": synthesis,
        "rows": out_rows,
    }


def _mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _p95(values: list[float]) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))]


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def _pct(value) -> str:
    return "—" if value is None else f"{value:.2%}"


def to_markdown(report: dict) -> str:
    o = report["overall"]
    lines = [
        "# Round-trip evaluation",
        "",
        f"- ASR: `{report['asr_backend']}`",
        f"- Scoring text: `{report['scoring_implementation']}`",
        f"- Rows: {o['rows']}, audio {o['audio_seconds']:.1f} s",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| WER | {_fmt(o['wer'])} |",
        f"| CER | {_fmt(o['cer'])} |",
        f"| ASR RTF | {_fmt(o['asr_rtf'])} |",
        f"| ASR speed (audio s / wall s) | {_fmt(o['asr_speed_x'])} |",
        f"| sample rates | {', '.join(map(str, o['sample_rates']))} |",
        "",
        "## By category",
        "",
        "| category | rows | WER | CER |",
        "|---|---:|---:|---:|",
    ]
    for name, c in report["categories"].items():
        lines.append(f"| {name} | {c['rows']} | {_pct(c['wer'])} | {_pct(c['cer'])} |")
    if report.get("synthesis"):
        s = report["synthesis"]
        lines += [
            "",
            "## Synthesis timing (from manifest fields)",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| rows | {s['rows']} |",
            f"| RTF | {_fmt(s['rtf'])} |",
            f"| first audio mean (s) | {_fmt(s['first_audio_seconds_mean'])} |",
            f"| first audio p95 (s) | {_fmt(s['first_audio_seconds_p95'])} |",
        ]
    stress = report.get("stress")
    if stress:
        lines += [
            "",
            "## Stress audit",
            "",
            f"- items: {stress['items']}, with audio: {stress['with_audio']}",
            f"- word miss rate: {_fmt(stress['word_miss_rate'])}",
            f"- stress error rate: {_fmt(stress['stress_error_rate'])}",
            f"- detector: {stress['detector'] or 'none'}",
        ]
    lines += ["", "## Rows", "", "| id | category | WER | CER | ASR |", "|---|---|---:|---:|---|"]
    for r in report["rows"]:
        lines.append(
            f"| {r['id']} | {r['category']} | {_pct(r['wer'])} | "
            f"{_pct(r['cer'])} | {r['asr'][:80].replace('|', '/')} |"
        )
    if report.get("warnings"):
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in report["warnings"]]
    return "\n".join(lines) + "\n"

