"""Canonical file list of the remote bundle; used by pack.sh and the tests.

    python remote/lib/bundle_files.py --check-stage out/remote-bundle
"""

from __future__ import annotations

import argparse
import os
import sys

REQUIRED = [
    "README.md",
    "remote/README.md",
    "remote/bootstrap.sh",
    "remote/generate.sh",
    "remote/watch.sh",
    "remote/publish.sh",
    "remote/pack.sh",
    "remote/assets.lock",
    "remote/requirements-remote.txt",
    "remote/lib/verify.py",
    "remote/lib/health.py",
    "remote/lib/smoke.py",
    "remote/lib/publish.py",
    "remote/lib/bundle_files.py",
    "scripts/gen/build_corpus.py",
    "scripts/gen/screen.py",
    "scripts/gen/README.md",
    "training/voca_ru/__init__.py",
    "training/voca_ru/text/__init__.py",
    "training/voca_ru/text/pipeline.py",
    "training/voca_ru/text/normalize.py",
    "training/voca_ru/text/numbers.py",
    "training/voca_ru/text/stress.py",
    "training/voca_ru/text/g2p.py",
    "training/voca_ru/text/symbols_ru.py",
    "training/voca_ru/eval/__init__.py",
    "training/voca_ru/eval/asr.py",
    "training/voca_ru/eval/audio.py",
    "training/voca_ru/eval/metrics.py",
    "training/voca_ru/eval/runner.py",
    "training/voca_ru/eval/scoring.py",
    "training/voca_ru/eval/stress.py",
    "data/scripts-ru/batch1.jsonl",
    "data/scripts-ru/batch2.jsonl",
    "assets/voices/f_neutral.wav",
    "assets/voices/f_neutral.txt",
    "assets/voices/m_neutral.wav",
    "assets/voices/m_neutral.txt",
    "assets/voices/README.md",
    "configs/base.yaml",
    "configs/smoke.yaml",
]

# Nothing from these belongs in a tarball that ships to a rented box.
FORBIDDEN_PREFIXES = (".cache/", "out/", "data/corpus-ru/", "worktrees/", ".venv-remote/")


def check_stage(stage_dir):
    missing = [rel for rel in REQUIRED if not os.path.isfile(os.path.join(stage_dir, rel))]
    forbidden = []
    for root, _dirs, files in os.walk(stage_dir):
        rel_root = os.path.relpath(root, stage_dir)
        for name in files:
            rel = os.path.normpath(os.path.join(rel_root, name))
            if rel.startswith(FORBIDDEN_PREFIXES):
                forbidden.append(rel)
    return missing, forbidden


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-stage", required=True)
    args = parser.parse_args(argv)
    missing, forbidden = check_stage(args.check_stage)
    for rel in missing:
        print(f"MISSING {rel}")
    for rel in forbidden[:20]:
        print(f"FORBIDDEN {rel}")
    if missing or forbidden:
        print(f"BUNDLE FAIL ({len(missing)} missing, {len(forbidden)} forbidden)")
        return 1
    print(f"BUNDLE PASS ({len(REQUIRED)} required files present, no caches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
