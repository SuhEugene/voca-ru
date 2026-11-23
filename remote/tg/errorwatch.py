#!/usr/bin/env python3
"""Tail pipeline logs and send one Telegram alert per new error.

Started by remote/tg/watch-errors.sh (and by the hook in generate.sh /
bootstrap.sh).  Survives restarts and log rotation: the byte offset and inode
of each file are persisted, and a rotated/shrunk file is read from its start.

Detected (see tgcore.detect_error): tracebacks, ERROR, FAIL*, non-zero exit,
OOM / CUDA out of memory, disk full, Killed.  Repeated identical errors are
coalesced for --dedupe seconds.  Every alert carries the last 15 lines.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tgcore  # noqa: E402

MAX_CHUNK = 256 * 1024
CONTEXT_LINES = 15
MAX_SIGNATURES = 200


class WatchState(tgcore.Store):
    def __init__(self, path):
        super().__init__(path)
        data = self.load({})
        data.setdefault("logs", {})
        data.setdefault("signatures", {})
        self.data = data

    def offset(self, path):
        return (self.data["logs"].get(path) or {})

    def set_offset(self, path, inode, offset, head=None):
        record = {"inode": inode, "offset": offset}
        if head is not None:
            record["head"] = head
        self.data["logs"][path] = record

    def save(self):
        if len(self.data["signatures"]) > MAX_SIGNATURES:
            keep = sorted(self.data["signatures"].items(),
                          key=lambda item: item[1].get("last", 0))[-MAX_SIGNATURES:]
            self.data["signatures"] = dict(keep)
        super().save(self.data)


def read_new_lines(path, state, max_bytes=MAX_CHUNK):
    """New complete lines since the saved offset; updates state in memory.

    Rotation and in-place truncation are both detected: a new inode, a file
    shorter than the offset, or a changed 256-byte header restarts at zero.
    """
    try:
        with open(path, "rb") as handle:
            stat = os.fstat(handle.fileno())
            head = hashlib.sha1(handle.read(256)).hexdigest()
            saved = state.offset(path)
            offset = int(saved.get("offset", 0))
            if (saved.get("inode") != stat.st_ino or stat.st_size < offset
                    or (saved.get("head") and saved["head"] != head)):
                offset = 0
            if stat.st_size == offset:
                state.set_offset(path, stat.st_ino, offset, head)
                return []
            handle.seek(offset)
            chunk = handle.read(max_bytes)
    except OSError:
        return []
    end = chunk.rfind(b"\n")
    if end < 0:
        return []  # wait for the writer to finish the line
    lines = chunk[:end].decode("utf-8", "replace").splitlines()
    state.set_offset(path, stat.st_ino, offset + end + 1, head)
    return lines


class ErrorWatcher:
    def __init__(self, paths, *, state_path, dedupe_s=600, now=time.time):
        self.paths = [os.path.abspath(path) for path in paths]
        self.state = WatchState(state_path)
        self.dedupe_s = dedupe_s
        self.now = now
        self.context = {path: deque(maxlen=CONTEXT_LINES) for path in self.paths}
        self.notifier = tgcore.Notifier(now=now)
        self.host = socket.gethostname()

    def consume(self, path, lines):
        """Feed new lines for one log; return alerts that were sent."""
        sent = []
        for line in lines:
            if not line.strip():
                continue
            self.context[path].append(line)
            category = tgcore.detect_error(line)
            if not category:
                continue
            if self._is_duplicate(line):
                continue
            level = "WARN" if category == "alarm" else "ERROR"
            summary = line.strip()[:300]
            context = "\n".join(self.context[path])
            ok = self.notifier.send(
                summary, level,
                title=f"tg-watch \u00b7 {os.path.basename(path)}",
                host=self.host, context=context,
                key=tgcore.error_signature(line),
            )
            if ok:
                sent.append(summary)
        return sent

    def _is_duplicate(self, line):
        now = self.now()
        sig = tgcore.error_signature(line)
        record = self.state.data["signatures"].get(sig)
        if record and now - float(record.get("last", 0)) < self.dedupe_s:
            record["count"] = int(record.get("count", 1)) + 1
            record["last"] = now
            self.state.save()
            return True
        self.state.data["signatures"][sig] = {"last": now, "count": 1}
        self.state.save()
        return False

    def poll(self):
        sent = []
        for path in self.paths:
            lines = read_new_lines(path, self.state)
            if lines:
                sent.extend(self.consume(path, lines))
            self.state.save()
        return sent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", help="log files to watch (missing files are waited for)")
    parser.add_argument("--state", default=None,
                        help="offset/dedupe state (default: <TG_STATE_DIR>/watch-state.json)")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--dedupe", type=float, default=600.0)
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    args = parser.parse_args(argv)

    env = os.environ
    state_path = args.state or os.path.join(tgcore.state_dir(env), "watch-state.json")
    watcher = ErrorWatcher(args.logs, state_path=state_path, dedupe_s=args.dedupe)
    if not watcher.notifier.enabled:
        print("[tg-watch] disabled: no chat id / token", flush=True)
        return 0

    print(f"[tg-watch] watching {', '.join(watcher.paths)} "
          f"(interval {args.interval:g}s)", flush=True)
    while True:
        try:
            for summary in watcher.poll():
                print(f"[tg-watch] alert: {summary}", flush=True)
        except Exception as exc:  # never die on a bad line or a filesystem race
            print(f"[tg-watch] poll error: {exc}", flush=True)
        if args.once:
            return 0
        time.sleep(max(args.interval, 0.5))


if __name__ == "__main__":
    sys.exit(main())
