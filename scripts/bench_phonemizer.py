"""Phonemizer parity and timing (Python dizione vs the Rust port) as JSON,
from the output of scripts/dizione_parity.py --time."""
import json
import re
import subprocess
import sys

out = subprocess.run([sys.executable, "scripts/dizione_parity.py", "--time"], capture_output=True, text=True, check=True).stdout
summary = re.search(r"(\d+) texts .*?(\d+) mismatches", out)
python = json.loads(re.search(r"^python:\s*(\{.*\})", out, re.M)[1])
rust = json.loads(re.search(r"^rust:\s*(\{.*\})", out, re.M)[1])
print(json.dumps({"sentences": int(summary[1]), "mismatches": int(summary[2]), "python": python, "rust": rust}))
