"""Verify the finished corpus, write the dataset card, upload to HF.

    python remote/lib/publish.py --check                     # verify + stats only
    python remote/lib/publish.py --dry-run                   # + write README.md
    HF_TOKEN=... python remote/lib/publish.py                 # + upload
    HF_TOKEN=... python remote/lib/publish.py --allow-partial # partial corpus

Refuses to upload when verification fails, when HF_TOKEN is unset, or when the
corpus is obviously incomplete (override with --allow-partial).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import verify as verify_mod  # noqa: E402

TEACHER = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ADAPTER = "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf"
XLSR = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
WHISPER = "Systran/faster-whisper-large-v3-turbo"

SHAS = {
    "teacher_sha": ".cache/models/qwen3-tts-1.7b/model.safetensors",
    "tokenizer_sha": ".cache/models/qwen3-tts-1.7b/speech_tokenizer/model.safetensors",
    "adapter_sha": ".cache/models/qwen3-tts-ru-stress/lora/adapter.safetensors",
    "head_sha": ".cache/models/qwen3-tts-ru-stress/meter/head.pt",
    "xlsr_sha": ".cache/models/wav2vec2-xlsr-ru/pytorch_model.bin",
    "whisper_sha": ".cache/models/whisper-large-v3-turbo/model.bin",
}


def read_lock(root):
    table = {}
    path = os.path.join(root, "remote/assets.lock")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            repo, repo_path, dest, size, sha = line.split("|")
            table[dest] = {"repo": repo, "path": repo_path, "bytes": int(size), "sha256": sha}
    return table


def expected_sentences(root):
    total = 0
    script_dir = os.path.join(root, "data/scripts-ru")
    for name in sorted(os.listdir(script_dir)):
        if name.endswith(".jsonl"):
            with open(os.path.join(script_dir, name), encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
    return total


def build_meta(root, corpus_dir, repo):
    lock = read_lock(root)
    meta = {
        "repo": repo,
        "teacher_repo": TEACHER,
        "adapter_repo": ADAPTER,
        "generated_on": time.strftime("%Y-%m-%d", time.gmtime()),
        "year": time.strftime("%Y", time.gmtime()),
    }
    for key, dest in SHAS.items():
        meta[key] = lock.get(dest, {}).get("sha256", "n/a")
    stats_path = os.path.join(corpus_dir, "run-stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path, encoding="utf-8") as handle:
            run_stats = json.load(handle)
        meta["batch_size"] = run_stats.get("batch_size")
    rows = verify_mod.load_jsonl(os.path.join(corpus_dir, "manifest.jsonl"))
    meta["sentences"] = len({row.get("text_id") for row in rows})
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.abspath(os.path.join(HERE, "..", "..")))
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--repo", default=os.environ.get("HF_REPO", "SuhEugene/voca-ru-corpus"))
    parser.add_argument("--check", action="store_true", help="verify only")
    parser.add_argument("--dry-run", action="store_true", help="verify + write the card")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--public", action="store_true",
                        help="create the repo public instead of private")
    parser.add_argument("--max-rows", type=int, default=0, help="debug: verify a prefix")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    corpus_dir = os.path.abspath(args.corpus or os.path.join(root, "data/corpus-ru"))
    print(f"[publish] corpus {corpus_dir}", flush=True)
    if not os.path.isdir(corpus_dir):
        print(f"FAIL corpus directory missing: {corpus_dir}", flush=True)
        return 1

    result = verify_mod.verify(corpus_dir, max_rows=args.max_rows)
    for error in result["errors"][:verify_mod.MAX_PRINTED_ERRORS]:
        print("FAIL:", error, flush=True)
    if len(result["errors"]) > verify_mod.MAX_PRINTED_ERRORS:
        print(f"FAIL: ... and {len(result['errors']) - verify_mod.MAX_PRINTED_ERRORS} more",
              flush=True)
    if not result["ok"]:
        print(f"FAIL verification: {len(result['errors'])} problem(s); nothing uploaded",
              flush=True)
        return 1
    stats = result["stats"]
    expected = expected_sentences(root) * len(stats.get("voices") or {"f_neutral": 1})
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)
    print(f"[publish] {stats['clips']}/{expected} clips, {stats['hours']} h", flush=True)
    if stats["clips"] < expected and not (args.allow_partial or args.max_rows):
        print(f"FAIL corpus incomplete ({stats['clips']} < {expected}); "
              "rerun generate.sh or pass --allow-partial", flush=True)
        return 1

    meta = build_meta(root, corpus_dir, args.repo)
    card = verify_mod.dataset_card(stats, meta)
    card_path = os.path.join(corpus_dir, "README.md")
    if args.check:
        print(f"[publish] verification PASS; card would be {len(card)} bytes", flush=True)
        return 0
    with open(card_path, "w", encoding="utf-8") as handle:
        handle.write(card)
    print(f"[publish] wrote {card_path}", flush=True)
    if args.dry_run:
        print("[publish] dry run: not uploading", flush=True)
        return 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("FAIL HF_TOKEN is unset; refusing to upload", flush=True)
        return 1

    hf = os.path.join(root, ".venv-remote", "bin", "hf")
    if not os.path.isfile(hf):
        hf = "hf"
    visibility = "--public" if args.public else "--private"
    print(f"[publish] hf repo create {args.repo} (dataset, {visibility})", flush=True)
    subprocess.run(
        [hf, "repo", "create", args.repo, "--repo-type", "dataset", visibility, "--exist-ok"],
        check=True,
    )
    print(f"[publish] uploading {corpus_dir} -> {args.repo}", flush=True)
    subprocess.run(
        [hf, "upload", args.repo, corpus_dir, ".", "--repo-type", "dataset",
         "--commit-message", f"voca-ru corpus {meta['generated_on']}",
         "--exclude", "rejected/*", "--exclude", "run.log",
         "--exclude", "health.log", "--exclude", "status.json",
         "--exclude", "status-history.jsonl", "--exclude", "*.tmp"],
        check=True,
    )
    print(f"[publish] done: https://huggingface.co/datasets/{args.repo}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
