import json
import sys

path = "/workspace/train.log"
lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
for i, line in enumerate(lines[-8:], start=len(lines) - 7):
    line = line.strip()
    if line.startswith("{"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(i, "unparsed:", line[:120])
            continue
        keys = [k for k in ("step", "eval_error", "stop", "error", "reason", "message") if k in row]
        detail = {k: (str(row[k])[:80]) for k in keys}
        print(i, len(row), "keys:", list(row)[:6] if not keys else detail)
    else:
        print(i, line[:160])
