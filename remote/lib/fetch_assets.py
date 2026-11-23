"""Download every pinned asset and verify its SHA-256.

Reads ``remote/assets.lock`` (``repo|repo_path|dest|bytes|sha256``).  Entries
with repo ``bundle`` are shipped in the tarball and are only verified.  Files
already present with the right size and hash are not downloaded again, so
bootstrap is idempotent.

    python remote/lib/fetch_assets.py --check      # verify only
    python remote/lib/fetch_assets.py              # download + verify
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

LOCAL_DIRS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": ".cache/models/qwen3-tts-1.7b",
    "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf": ".cache/models/qwen3-tts-ru-stress",
    "jonatasgrosman/wav2vec2-large-xlsr-53-russian": ".cache/models/wav2vec2-xlsr-ru",
    "Systran/faster-whisper-large-v3-turbo": ".cache/models/whisper-large-v3-turbo",
    "ruaccent/accentuator": ".cache/models/ruaccent",
}


def read_lock(root):
    entries = []
    with open(os.path.join(root, "remote/assets.lock"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            repo, repo_path, dest, size, sha = line.split("|")
            entries.append({
                "repo": repo, "repo_path": repo_path, "dest": dest,
                "bytes": int(size), "sha256": sha,
            })
    return entries


def sha256_file(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def is_good(root, entry):
    path = os.path.join(root, entry["dest"])
    if not os.path.isfile(path) or os.path.getsize(path) != entry["bytes"]:
        return False
    return sha256_file(path) == entry["sha256"]


def download_group(hf, root, repo, entries):
    local_dir = os.path.join(root, LOCAL_DIRS[repo])
    command = [hf, "download", repo, "--repo-type", "model",
               "--local-dir", local_dir]
    for entry in entries:
        command += ["--include", entry["repo_path"]]
    print("[assets] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def download_and_verify(root, hf):
    entries = read_lock(root)
    by_repo = {}
    for entry in entries:
        if entry["repo"] != "bundle":
            by_repo.setdefault(entry["repo"], []).append(entry)

    for repo, group in by_repo.items():
        if repo not in LOCAL_DIRS:
            print(f"FAIL unknown repository in lock: {repo}", flush=True)
            return 1
        missing = [entry for entry in group if not is_good(root, entry)]
        if missing:
            print(f"[assets] {repo}: {len(group) - len(missing)}/{len(group)} cached",
                  flush=True)
            download_group(hf, root, repo, missing)

    failures = []
    print(f"\n{'asset':<62} {'size (MB)':>10}  status", flush=True)
    print("-" * 84, flush=True)
    for entry in entries:
        good = is_good(root, entry)
        status = "ok" if good else "MISMATCH/MISSING"
        print(f"{entry['dest']:<62} {entry['bytes'] / 1e6:>10.1f}  {status}", flush=True)
        if not good:
            failures.append(entry["dest"])
    if failures:
        print(f"FAIL {len(failures)} asset(s) failed verification", flush=True)
        return 1
    print(f"PASS {len(entries)} assets verified (SHA-256)", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..")))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    hf = os.path.join(args.root, ".venv-remote/bin/hf")
    if not os.path.isfile(hf):
        hf = "hf"
    if args.check:
        failures = [e["dest"] for e in read_lock(args.root)
                    if not is_good(args.root, e)]
        for dest in failures:
            print(f"FAIL {dest}")
        print("ASSETS PASS" if not failures else f"ASSETS FAIL ({len(failures)})")
        return 0 if not failures else 1
    return download_and_verify(args.root, hf)


if __name__ == "__main__":
    sys.exit(main())
