#!/usr/bin/env python3
"""Live view of every running Hugging Face download: progress, speed, ETA, bandwidth.

Auto-detects `scripts/fetch-hf-file.sh <url> <out> <sha> [parts]` jobs, sums their
range parts, asks each URL once for the real file size, and shows what this machine
is pulling right now (per job, total, and the interface counter as a cross-check).

  scripts/download-watch.py              # refresh every second until all jobs finish
  scripts/download-watch.py --once       # single snapshot (one 1 s sample)
  scripts/download-watch.py 0.5          # custom refresh interval
  scripts/download-watch.py <file> <bytes>   # watch one file by hand
"""
import os
import re
import subprocess
import sys
import time
import urllib.request

RE = re.compile(r"fetch-hf-file\.sh\s+(\S+)\s+(\S+)")


def ps_jobs():
    out = subprocess.run(["ps", "-Ao", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    procs = []
    for line in out.splitlines():
        m = re.search(r"^\s*(\d+)\s+(\d+)\s+.*fetch-hf-file\.sh\s+(\S+)\s+(\S+)", line)
        if m:
            procs.append((int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)))
    pids = {p[0] for p in procs}
    jobs = {}
    for pid, ppid, url, dest in procs:
        dest = os.path.abspath(os.path.expanduser(dest))
        url0, n = jobs.get(dest, (url, 0))
        jobs[dest] = (url0, n + (0 if ppid in pids else 1))   # count launchers, not part subshells
    return jobs


def head_size(url, cache):
    if url not in cache:
        size = None
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as r:
                size = int(r.headers.get("Content-Length") or 0) or None
        except Exception:
            pass
        cache[url] = size
    return cache[url]


def parts_dir(dest):
    return os.path.join(os.path.dirname(dest), "parts-" + os.path.basename(dest))


def tree_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def downloaded(dest):
    p = parts_dir(dest)
    n = tree_size(p) if os.path.isdir(p) else 0
    if n == 0 and os.path.exists(dest):
        n = os.path.getsize(dest)
    return n


def net_bytes():
    # en0 is listed once per address family; count the Link# row only.
    try:
        out = subprocess.run(["netstat", "-ibn"], capture_output=True, text=True).stdout
    except OSError:
        return 0
    total = 0
    for line in out.splitlines():
        f = line.split()
        if f and f[0] == "en0" and "Link#" in line and f[6].isdigit():
            total += int(f[6])
    return total


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def bar(pct, width=30):
    pct = max(0.0, min(pct, 100.0))
    fill = int(pct / 100 * width)
    return "#" * fill + "-" * (width - fill)


def eta(sec):
    if sec is None:
        return "?"
    sec = int(sec)
    if sec >= 3600:
        return f"{sec // 3600}h{sec % 3600 // 60:02d}m"
    return f"{sec // 60}m{sec % 60:02d}s"


def label(dest):
    parts = dest.split(os.sep)
    return "/".join(parts[-2:])


def main():
    args = sys.argv[1:]
    once = "--once" in args
    args = [a for a in args if a != "--once"]
    interval, manual = 1.0, None
    if len(args) == 2:
        manual = (os.path.abspath(os.path.expanduser(args[0])), int(args[1]))
    elif len(args) == 1:
        interval = float(args[0])

    cache = {}
    tty = sys.stdout.isatty()
    start = time.monotonic()

    while True:
        jobs = ps_jobs()
        if manual:
            jobs[manual[0]] = (None, 1)
        if not jobs:
            print("no active downloads")
            return 0

        t0 = time.monotonic()
        base = {d: downloaded(d) for d in jobs}
        net0 = net_bytes()
        time.sleep(interval)
        rows = []
        for dest, (url, writers) in sorted(jobs.items()):
            size = manual[1] if manual and dest == manual[0] else head_size(url, cache)
            rows.append((dest, downloaded(dest), size, writers))
        dt = max(time.monotonic() - t0, 1e-6)
        net_rate = max(net_bytes() - net0, 0) / dt

        display, total_rate = [], 0.0
        for dest, have, size, writers in rows:
            rate = max(have - base[dest], 0) / dt
            total_rate += rate
            pct = 100.0 * have / size if size else 0.0
            left = max(size - have, 0) if size else 0
            display.append((label(dest), have, size, pct, rate,
                            left / rate if rate > 0 else None, writers))

        if tty:
            print("\033[2J\033[H", end="")
        print(f"downloads: {len(display)}   elapsed {eta(time.monotonic() - start)}")
        print()
        for name, have, size, pct, rate, left, writers in display:
            print(name + (f"   [{writers} writers!]" if writers > 1 else ""))
            note = "  (parts exceed size: stale parts?)" if size and have > size * 1.05 else ""
            print(f"  [{bar(pct)}] {pct:5.1f}%  {human(have)} / {human(size) if size else '?'}{note}")
            print(f"  {human(rate)}/s   ETA {eta(left)}")
        print()
        print(f"total {human(total_rate)}/s   interface in {human(net_rate)}/s")

        if once or all(size and have >= size for _, have, size, _, _, _, _ in display):
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
