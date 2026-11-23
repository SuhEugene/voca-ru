"""The bundle's file list and a real pack.sh run (no models, no caches)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "lib"))

import bundle_files  # noqa: E402


class BundleListTests(unittest.TestCase):
    def test_every_required_file_has_a_source_in_the_repo(self):
        git_common = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        shared = os.path.dirname(os.path.abspath(git_common)) if git_common else REPO
        missing = []
        for rel in bundle_files.REQUIRED:
            candidates = [rel]
            if rel == "README.md":
                candidates = ["remote/bundle-README.md"]
            elif rel.endswith(".wav"):
                candidates = [
                    rel,
                    os.path.join("out/g1/voices", os.path.basename(rel)),
                    os.path.join(shared, "out/g1/voices", os.path.basename(rel)),
                ]
            if not any(os.path.isfile(os.path.join(REPO, c)) for c in candidates):
                missing.append(rel)
        self.assertEqual(missing, [], f"no source for bundle files: {missing}")

    def test_forbidden_prefixes_are_not_required(self):
        for rel in bundle_files.REQUIRED:
            self.assertFalse(rel.startswith(bundle_files.FORBIDDEN_PREFIXES), rel)


@unittest.skipUnless(shutil.which("zstd"), "zstd is not installed")
class PackScriptTests(unittest.TestCase):
    def test_pack_builds_a_small_valid_bundle(self):
        result = subprocess.run(
            ["bash", os.path.join(REPO, "remote/pack.sh")],
            cwd=REPO, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PACK PASS", result.stdout)

        archive = os.path.join(REPO, "out/remote-bundle.tar.zst")
        self.assertLess(os.path.getsize(archive), 50 * 1024 * 1024)

        listing = subprocess.run(
            ["bash", "-c", f"zstd -dc {archive} | tar -tf -"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        names = {name.lstrip("./") for name in listing}
        for rel in ("README.md", "remote/bootstrap.sh", "remote/assets.lock",
                    "scripts/gen/build_corpus.py", "training/voca_ru/text/pipeline.py",
                    "data/scripts-ru/batch1.jsonl", "assets/voices/f_neutral.wav"):
            self.assertIn(rel, names)
        self.assertFalse(any(name.startswith(("./.cache", ".cache", "./out", "out/"))
                             for name in names))

        staged = os.path.join(REPO, "out/remote-bundle")
        missing, forbidden = bundle_files.check_stage(staged)
        self.assertEqual(missing, [])
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
