"""CLI: python -m voca_ru.eval --manifest m.jsonl --audio-dir d --out r.json --md r.md"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import scoring
from .asr import make_backend, timed_transcribe
from .runner import evaluate, load_manifest, to_markdown
from .stress import DEFAULT_SET, evaluate_stress, load_stress_set

ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m voca_ru.eval", description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="JSONL rows {id,text,audio}")
    src.add_argument("--audio", help="single WAV/FLAC file; requires --text")
    p.add_argument("--text", help="reference text for --audio")
    p.add_argument("--audio-dir", help="base directory for relative manifest paths")
    p.add_argument("--out", required=True, help="JSON report path")
    p.add_argument("--md", help="Markdown report path")
    p.add_argument("--backend", default="faster-whisper",
                   choices=["faster-whisper", "gigaam", "fake"])
    p.add_argument("--model-path", help="CTranslate2 model directory")
    p.add_argument("--gigaam-model", default="v3_e2e_rnnt")
    p.add_argument("--device", default="cpu", help="cpu (CTranslate2 has no MPS backend)")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--fake-transcripts", help="JSON {audio_path: text} for --backend fake")
    p.add_argument("--corpus", action="append", default=[],
                   help="script JSONL for category lookup (repeatable)")
    p.add_argument("--limit", type=int)
    p.add_argument("--stress-set", nargs="?", const=str(DEFAULT_SET),
                   help="run the stress audit; optional path (default: built-in 200 items)")
    p.add_argument("--stress-audio-dir", help="<item id>.wav files for the stress set")
    p.add_argument("--category", default="uncategorized", help="category for --audio")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.audio and not args.text:
        raise SystemExit("--audio requires --text")
    if args.audio:
        rows = [{"id": Path(args.audio).stem, "text": args.text,
                 "audio": args.audio, "category": args.category}]
    else:
        rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    corpora = [Path(p) for p in args.corpus]
    if not corpora:
        corpora = sorted((ROOT / "data" / "scripts-ru").glob("*.jsonl"))

    backend = make_backend(
        args.backend,
        model_dir=args.model_path,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        fake_transcripts=args.fake_transcripts,
        gigaam_model=args.gigaam_model,
    )
    report = evaluate(rows, backend, args.audio_dir, corpora)
    report["asr_backend"] = backend.name
    report["scoring_implementation"] = scoring.implementation
    report["warnings"] = []
    if args.backend == "fake":
        report["warnings"].append("fake ASR backend: numbers test the harness, not speech")
    if args.backend == "gigaam":
        report["warnings"].append("GigaAM is a cross-check, not the primary judge")
    if scoring.implementation.startswith("eval.shim"):
        report["warnings"].append("scoring uses the F5 shim (TODO F3)")
    if any(rate != 24000 for rate in report["overall"]["sample_rates"]):
        report["warnings"].append(
            f"non-24 kHz audio: {report['overall']['sample_rates']} (contract §2)"
        )

    if args.stress_set:
        items = load_stress_set(args.stress_set)
        transcripts: dict[str, str] = {}
        if args.stress_audio_dir:
            audio_dir = Path(args.stress_audio_dir)
            for item in items:
                path = audio_dir / f"{item['id']}.wav"
                if path.is_file():
                    transcripts[item["id"]], _ = timed_transcribe(backend, str(path))
        report["stress"] = evaluate_stress(items, transcripts)
        if not transcripts:
            report["warnings"].append(
                "stress set loaded but no audio found; set --stress-audio-dir"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    if args.md:
        md = Path(args.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(to_markdown(report), "utf-8")
    o = report["overall"]
    speed = f"{o['asr_speed_x']:.1f}x" if o["asr_speed_x"] else "n/a"
    print(f"WER {o['wer']:.2%}  CER {o['cer']:.2%}  "
          f"rows {o['rows']}  ASR {speed} audio time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
