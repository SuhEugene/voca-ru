"""Asset lock: six HF sources, pinned sizes/hashes, paths where the code looks."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "remote", "lib"))

import fetch_assets  # noqa: E402

EXPECTED_REPOS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf",
    "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "Systran/faster-whisper-large-v3-turbo",
    "ruaccent/accentuator",
    "bundle",
}


class AssetLockTests(unittest.TestCase):
    def setUp(self):
        self.entries = fetch_assets.read_lock(REPO)

    def test_six_sources_present(self):
        self.assertEqual({entry["repo"] for entry in self.entries}, EXPECTED_REPOS)

    def test_every_line_is_well_formed(self):
        self.assertEqual(len(self.entries), 33)
        dests = [entry["dest"] for entry in self.entries]
        self.assertEqual(len(dests), len(set(dests)), "duplicate destination")
        for entry in self.entries:
            self.assertGreater(entry["bytes"], 0, entry)
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$", entry)
            if entry["repo"] != "bundle":
                self.assertTrue(entry["dest"].startswith(".cache/models/"), entry)
            else:
                self.assertTrue(entry["dest"].startswith("assets/voices/"), entry)

    def test_paths_match_what_the_pipeline_reads(self):
        dests = {entry["dest"] for entry in self.entries}
        for required in (
            ".cache/models/qwen3-tts-1.7b/model.safetensors",
            ".cache/models/qwen3-tts-1.7b/speech_tokenizer/model.safetensors",
            ".cache/models/qwen3-tts-ru-stress/lora/adapter.safetensors",
            ".cache/models/qwen3-tts-ru-stress/meter/head.pt",
            ".cache/models/wav2vec2-xlsr-ru/pytorch_model.bin",
            ".cache/models/whisper-large-v3-turbo/model.bin",
            ".cache/models/ruaccent/dictionary/accents.json.gz",
        ):
            self.assertIn(required, dests)

    def test_recorded_hashes_match_local_small_files(self):
        """Spot-check the digest of small pinned files against the shared cache."""
        git_common = os.popen(
            f"git -C {REPO} rev-parse --path-format=absolute --git-common-dir"
        ).read().strip()
        shared = os.path.dirname(os.path.abspath(git_common)) if git_common else REPO
        checked = 0
        for entry in self.entries:
            if entry["repo"] == "bundle" or entry["bytes"] > 5 * 1024 * 1024:
                continue
            path = os.path.join(shared, entry["dest"])
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            self.assertEqual(digest, entry["sha256"], entry["dest"])
            checked += 1
        self.assertGreater(checked, 5)

    def test_local_dirs_are_declared_for_every_hf_repo(self):
        for entry in self.entries:
            if entry["repo"] != "bundle":
                self.assertIn(entry["repo"], fetch_assets.LOCAL_DIRS)


if __name__ == "__main__":
    unittest.main()
