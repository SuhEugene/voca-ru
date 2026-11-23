"""Shared, standard-library-only logic for the Telegram alerts and status bot.

Everything that can be tested without the network lives here: escaping,
truncation, rate limiting, state persistence, error-line detection, status
rendering and the widget payload/HTML.  ``notify.py``, ``errorwatch.py`` and
``status-bot.py`` are thin command-line shells around this module.

Docs this follows (https://core.telegram.org/bots/api and /bots/widgets):
  * sendMessage text: 1-4096 characters after entities parsing; parse_mode HTML
    needs ``&``, ``<`` and ``>`` escaped.
  * sendWidget creates a widget; with ``is_shared`` the first send creates a
    shared widget whose ``widget.id`` is reused by every later message.
  * editSharedWidgetData(widget_id, public_data, cache_time) changes the shared
    state; a shared widget takes at most one edit per second and ``cache_time``
    is 5-3600 s (60 s here).  public_data is at most 1024 UTF-8 bytes.
  * Widget HTML is at most 32768 bytes and 4096 bytes after zlib-9 compression.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

MSG_LIMIT = 4096          # sendMessage, characters after entities parsing
CAPTION_LIMIT = 1024
WIDGET_STATE_LIMIT = 1024  # public_data, UTF-8 bytes
WIDGET_HTML_LIMIT = 32768
WIDGET_CACHE_TIME = 60    # seconds; docs allow 5-3600
REFRESH_SECONDS = 60

LEVEL_ICON = {"CRIT": "\U0001f525", "ERROR": "\u274c", "WARN": "\u26a0\ufe0f",
              "INFO": "\u2139\ufe0f"}

_ENTITY_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


# --------------------------------------------------------------------- text

def escape_html(text):
    """Escape the three characters Telegram's HTML parser treats as markup."""
    return "".join(_ENTITY_ESCAPES.get(ch, ch) for ch in text)


def truncate(text, limit=MSG_LIMIT, ellipsis="\u2026"):
    """Cut at ``limit`` characters (code points, as Telegram counts them)."""
    if len(text) <= limit:
        return text
    if limit <= len(ellipsis):
        return ellipsis[:limit]
    return text[: limit - len(ellipsis)].rstrip() + ellipsis


def truncate_markup(text, limit=MSG_LIMIT, ellipsis="\u2026"):
    """Cut already-escaped HTML without splitting an entity or a tag."""
    if len(text) <= limit:
        return text
    cut = text[: limit - len(ellipsis)]
    cut = re.sub(r"&[a-zA-Z#0-9]{0,8}$", "", cut)
    cut = re.sub(r"<[^<>]*$", "", cut)
    return cut.rstrip() + ellipsis


def format_alert(text, level="INFO", *, title=None, host=None, context=None,
                 limit=MSG_LIMIT):
    """One alert as HTML.  ``context`` is shown in a <pre> block.

    Dynamic parts are escaped; the layout tags stay literal.  The 4096-character
    limit is counted after entities are parsed, so a cut never leaves a half
    entity or tag behind.
    """
    level = (level or "INFO").upper()
    head = f"{LEVEL_ICON.get(level, LEVEL_ICON['INFO'])} <b>{escape_html(level)}</b>"
    if title:
        head += f" \u00b7 {escape_html(str(title))}"
    if host:
        head += f" @ {escape_html(str(host))}"
    parts = [head]
    if text:
        parts.append(escape_html(str(text)))
    if context:
        parts.append(f"<pre>{escape_html(str(context))}</pre>")
    return truncate_markup("\n".join(parts), limit)


# -------------------------------------------------------------------- config

def state_path(env=None):
    env = env or os.environ
    return env.get("TG_STATE") or os.path.join(HERE, "state.json")


def state_dir(env=None):
    env = env or os.environ
    path = env.get("TG_STATE_DIR") or os.path.join(HERE, "state")
    os.makedirs(path, exist_ok=True)
    return path


def data_dir(env=None):
    env = env or os.environ
    root = env.get("VOCA_ROOT") or REPO
    return env.get("TG_CORPUS_DIR") or os.path.join(root, "data", "corpus-ru")


def read_token(env=None):
    env = env or os.environ
    token = (env.get("TG_TOKEN") or "").strip()
    if not token:
        path = env.get("TG_TOKEN_FILE") or os.path.expanduser("~/.secrets/telegram-bot-token")
        try:
            with open(path, encoding="utf-8") as handle:
                token = handle.read().strip()
        except OSError:
            token = ""
    return token


def read_chat_id(env=None):
    """$TG_CHAT_ID first, then the id the bot saved when someone sent /start."""
    env = env or os.environ
    raw = (env.get("TG_CHAT_ID") or "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            return raw
    try:
        with open(state_path(env), encoding="utf-8") as handle:
            value = json.load(handle).get("chat_id")
    except (OSError, ValueError):
        return None
    return value


def mask_secret(text, token=""):
    """Never let a bot token reach a log line."""
    text = str(text)
    if token:
        text = text.replace(token, "<token>")
    return re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot<token>", text)


def log_error(path, line, token=""):
    """Append one line to tg-errors.log; never raise."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {mask_secret(line, token)}\n")
    except OSError:
        pass


# -------------------------------------------------------------------- state

def _atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


class Store:
    """Tiny JSON-file store with an exclusive lock for cross-process writers."""

    def __init__(self, path):
        self.path = path
        self._lock_path = path + ".lock"

    @contextlib.contextmanager
    def locked(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            import fcntl
        except ImportError:  # pragma: no cover - not on Linux/macOS
            yield
            return
        with open(self._lock_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load(self, default=None):
        return load_json(self.path, default if default is not None else {})

    def save(self, data):
        _atomic_write_json(self.path, data)


class BotState(Store):
    """remote/tg/state.json: chat id, update offset, one widget per chat."""

    def __init__(self, path=None):
        super().__init__(path or state_path())
        data = self.load({})
        data.setdefault("chat_id", None)
        data.setdefault("offset", None)
        data.setdefault("widgets", {})
        self.data = data

    def save(self, data=None):
        if data is not None:
            self.data = data
        super().save(self.data)

    @property
    def chat_id(self):
        return self.data.get("chat_id")

    def set_chat_id(self, chat_id):
        self.data["chat_id"] = chat_id
        self.save()

    @property
    def offset(self):
        return self.data.get("offset")

    def advance(self, update_id):
        if update_id is None:
            return
        self.data["offset"] = int(update_id) + 1
        self.save()

    def widget(self, chat_id):
        return (self.data.get("widgets") or {}).get(str(chat_id))

    def set_widget(self, chat_id, record):
        self.data.setdefault("widgets", {})[str(chat_id)] = record
        self.save()

    def drop_widget(self, chat_id):
        (self.data.get("widgets") or {}).pop(str(chat_id), None)
        self.save()


# ------------------------------------------------------------- rate limiting

class Decision:
    __slots__ = ("send", "reason", "suppressed")

    def __init__(self, send, reason, suppressed=0):
        self.send = send
        self.reason = reason
        self.suppressed = suppressed

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Decision(send={self.send!r}, reason={self.reason!r}, suppressed={self.suppressed})"


def _signature(text, key=None):
    raw = key if key is not None else " ".join(str(text).split())
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class RateLimiter:
    """At most N messages per rolling minute; the same text is sent once per window."""

    def __init__(self, path, *, max_per_min=8, dedupe_s=300, now=time.time):
        self.store = Store(path)
        self.max_per_min = max_per_min
        self.dedupe_s = dedupe_s
        self.now = now

    def check(self, text, key=None, now=None):
        now = self.now() if now is None else now
        sig = _signature(text, key)
        with self.store.locked():
            data = self.store.load({})
            sent = [float(t) for t in data.get("sent", []) if now - float(t) < 60]
            last = {k: float(v) for k, v in (data.get("last") or {}).items()
                    if now - float(v) < max(self.dedupe_s, 60)}
            suppressed = int(data.get("suppressed", 0))
            previous = last.get(sig)
            if previous is not None and now - previous < self.dedupe_s:
                data.update(sent=sent, last=last, suppressed=suppressed)
                self.store.save(data)
                return Decision(False, "duplicate", suppressed)
            if len(sent) >= self.max_per_min:
                data.update(sent=sent, last=last, suppressed=suppressed + 1)
                self.store.save(data)
                return Decision(False, "throttled", suppressed + 1)
            sent.append(now)
            last[sig] = now
            data.update(sent=sent, last=last, suppressed=0)
            self.store.save(data)
            return Decision(True, "send", suppressed)


# --------------------------------------------------------------------- HTTP

class TelegramError(RuntimeError):
    def __init__(self, method, error_code, description, retry_after=None):
        super().__init__(f"{method}: {error_code} {description}")
        self.method = method
        self.error_code = error_code
        self.description = description
        self.retry_after = retry_after


class TelegramClient:
    """Minimal Bot API client with 429 handling.  ``transport`` enables tests."""

    def __init__(self, token, *, base_url="https://api.telegram.org",
                 transport=None, sleep=time.sleep, timeout=30):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _urllib_post
        self.sleep = sleep
        self.timeout = timeout

    def call(self, method, params=None, *, attempts=3):
        url = f"{self.base_url}/bot{self.token}/{method}"
        body = json.dumps(params or {}, ensure_ascii=False).encode("utf-8")
        last = None
        for attempt in range(max(1, attempts)):
            try:
                status, text = self.transport(url, body, self.timeout)
            except Exception as exc:  # network down: fail soft for the caller
                raise TelegramError(method, 0, str(exc)) from exc
            try:
                payload = json.loads(text) if text else {}
            except ValueError:
                payload = {}
            if payload.get("ok"):
                return payload.get("result")
            code = int(payload.get("error_code") or status or 0)
            description = payload.get("description") or str(text)[:200]
            retry_after = (payload.get("parameters") or {}).get("retry_after")
            last = TelegramError(method, code, description, retry_after)
            if code == 429 and attempt + 1 < attempts:
                self.sleep(min(float(retry_after or 1), 30))
                continue
            if code >= 500 and attempt + 1 < attempts:
                self.sleep(1 + attempt)
                continue
            raise last
        raise last  # pragma: no cover - attempts >= 1 always raises above

    # -- methods used by the bot ------------------------------------------
    def send_message(self, chat_id, text, **extra):
        params = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
        params.update(extra)
        return self.call("sendMessage", params)

    def edit_message_text(self, chat_id, message_id, text, **extra):
        params = {"chat_id": chat_id, "message_id": message_id, "text": text}
        params.update(extra)
        return self.call("editMessageText", params)

    def send_widget(self, chat_id, html, **extra):
        params = {"chat_id": chat_id, "html": html}
        params.update(extra)
        return self.call("sendWidget", params)

    def edit_shared_widget_data(self, widget_id, public_data, **extra):
        params = {"widget_id": widget_id, "public_data": public_data}
        params.update(extra)
        return self.call("editSharedWidgetData", params)

    def get_updates(self, offset=None, timeout=8, **extra):
        params = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            params["offset"] = offset
        params.update(extra)
        return self.call("getUpdates", params)


def _urllib_post(url, body, timeout):
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


class Notifier:
    """One-shot alert path: dedupe + throttle + send; failures never raise."""

    def __init__(self, *, env=None, client=None, now=time.time):
        env = env or os.environ
        self.env = env
        self.token = read_token(env)
        self.chat_id = read_chat_id(env)
        self.now = now
        self.dir = state_dir(env)
        self.error_log = env.get("TG_ERROR_LOG") or os.path.join(self.dir, "tg-errors.log")
        try:
            per_min = int(env.get("TG_MAX_PER_MIN", "8"))
        except ValueError:
            per_min = 8
        try:
            dedupe = int(env.get("TG_DEDUPE_SECONDS", "300"))
        except ValueError:
            dedupe = 300
        self.limiter = RateLimiter(os.path.join(self.dir, "notify-state.json"),
                                   max_per_min=per_min, dedupe_s=dedupe, now=now)
        self.client = client or (
            TelegramClient(self.token, base_url=self.env.get("TG_API_BASE")
                           or "https://api.telegram.org", sleep=self._sleep)
            if self.token else None)
        self.enabled = bool(self.chat_id and self.token)

    def _sleep(self, seconds):
        time.sleep(seconds)

    def send(self, text, level="INFO", *, key=None, title=None, host=None,
             context=None, parse_mode="HTML"):
        if not self.enabled:
            return False
        body = format_alert(text, level, title=title, host=host, context=context)
        decision = self.limiter.check(body, key=key)
        if not decision.send:
            return False
        try:
            self.client.send_message(self.chat_id, body, parse_mode=parse_mode)
        except TelegramError as exc:
            log_error(self.error_log, str(exc), self.token)
            return False
        return True


# ------------------------------------------------------------ error scanning

_ERROR_PATTERNS = (
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("oom", re.compile(r"CUDA out of memory|OutOfMemoryError|OutOfMemory")),
    ("disk", re.compile(r"No space left on device|Disk quota exceeded")),
    ("killed", re.compile(r"(?:^|\s)Killed(?:\s|$)|OOMKilled|Segmentation fault")),
    ("error", re.compile(r"(?:^|[\s\[:])ERROR(?:\s|[:.\]])")),
    ("fail", re.compile(r"(?:^|[\s\[:])FAIL(?:ED|URE)?(?:\s|[:.\]])")),
    ("exit", re.compile(r"exited (?:with )?(?:code |status )?[1-9]\d*"
                        r"|Exit code[: ]+[1-9]\d*|returncode[=: ]+[1-9]\d*")),
    ("alarm", re.compile(r"^ALARM:")),
)


def detect_error(line):
    """Category of an alarming log line, or None.  ``Traceback`` outranks ERROR."""
    for category, pattern in _ERROR_PATTERNS:
        if pattern.search(line):
            return category
    return None


def error_signature(line):
    """Strip the parts that change between repeats so dedupe sees one error."""
    text = re.sub(r"\d{4}-\d{2}-\d{2}[T ][\d:.]+Z?", "<ts>", line)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<n>", text)
    return text.strip()[:240]


def format_error_alert(category, summary, context, *, log_name="run.log", host=None):
    """Body sent by errorwatch: summary line plus the last lines as context."""
    text = f"{summary}\ncategory: {category} \u00b7 {log_name}"
    return text, context


# ------------------------------------------------------------------- status

def parse_iso(text):
    if not text:
        return None
    text = str(text).strip().replace("Z", "+00:00")
    try:
        import datetime
        return datetime.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def human_duration(seconds):
    if seconds is None:
        return "\u2014"
    seconds = max(0, int(round(float(seconds))))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def progress_bar(pct, width=12):
    pct = 0.0 if pct is None else max(0.0, min(100.0, float(pct)))
    filled = int(round(width * pct / 100.0))
    return "\u2593" * filled + "\u2591" * (width - filled)


def _g(status, *names):
    for name in names:
        value = status.get(name)
        if value is not None:
            return value
    return None


def spend(pod, status=None, now=None):
    """(dollars_per_hour, hours, dollars) from .cache/runpod-pod.json, or Nones."""
    now = now if now is not None else time.time()
    if not pod:
        return None, None, None
    try:
        rate = float(pod.get("cost"))
    except (TypeError, ValueError):
        return None, None, None
    started = parse_iso(pod.get("created"))
    if started is None and status:
        started = parse_iso(status.get("run_started_at"))
    hours = max(0.0, (now - started) / 3600.0) if started else None
    dollars = rate * hours if hours is not None else None
    return rate, hours, dollars


def format_status(status, *, stats=None, pod=None, now=None):
    """Compact plain-text dashboard for a phone (fallback when widgets fail)."""
    now = now if now is not None else time.time()
    stats = stats or {}
    status = status or {}
    done = _g(status, "clips_done")
    if done is None:
        done = stats.get("clips_accepted") or 0
    total = _g(status, "clips_expected") or 0
    pct = _g(status, "progress_pct")
    if pct is None and total:
        pct = 100.0 * done / total
    rate = _g(status, "clips_per_min")
    if rate is None:
        rate = stats.get("clips_per_min")
    updated = parse_iso(status.get("updated_at"))
    status_age = None if updated is None else max(0.0, now - updated)
    stale = status_age is not None and status_age > 600

    warnings = []
    age = _g(status, "last_clip_age_s")
    disk = _g(status, "disk_free_gb")
    temp = _g(status, "gpu_temp_c")
    if age is not None and age > 20 * 60:
        warnings.append("no new clips")
    if disk is not None and disk < 10:
        warnings.append("low disk")
    if temp is not None and temp > 85:
        warnings.append("hot GPU")
    if stale:
        warnings.append("stale status")
    icon = "\U0001f534" if stale else ("\U0001f7e0" if warnings else "\U0001f7e2")

    clips_line = f"{done:,} / {total:,} clips" if total else f"{done:,} clips"
    if rate is not None:
        clips_line += f" \u00b7 {rate}/min"

    lines = [
        f"{icon} corpus \u00b7 {time.strftime('%H:%M', time.localtime(now))} local",
        f"{progress_bar(pct)} {pct:.1f}%" if pct is not None else f"{progress_bar(0)} \u2014",
        clips_line,
        f"ETA {human_duration((total - done) / rate * 60)} \u00b7 last clip {human_duration(age)}"
        if rate and total > done else f"last clip {human_duration(age)}",
    ]
    wer = _g(status, "wer_mean")
    p95 = _g(status, "wer_p95")
    stress = _g(status, "stress_fail_pct")
    dropped = _g(status, "dropped")
    quality = []
    if wer is not None:
        quality.append(f"WER {100 * wer:.2f}%")
    if p95 is not None:
        quality.append(f"p95 {100 * p95:.2f}%")
    if stress is not None:
        quality.append(f"stress {stress:.2f}%")
    if dropped is not None:
        quality.append(f"dropped {dropped}")
    if quality:
        lines.append(" \u00b7 ".join(quality))
    gpu = []
    if temp is not None:
        gpu.append(f"{temp}\u00b0C")
    used, total_vram = _g(status, "vram_used_gb"), _g(status, "vram_total_gb")
    if used is not None:
        gpu.append(f"{used}/{total_vram} GB" if total_vram else f"{used} GB")
    if disk is not None:
        gpu.append(f"disk {disk:.0f} GB")
    if gpu:
        lines.append(" \u00b7 ".join(gpu))
    rate_usd, hours, dollars = spend(pod, status, now)
    if dollars is not None:
        lines.append(f"${rate_usd:.2f}/h \u00d7 {human_duration(hours * 3600)} = ${dollars:.2f}")
    if warnings:
        lines.append("warning: " + ", ".join(warnings))
    return "\n".join(lines)


def widget_public_data(status, *, stats=None, pod=None, now=None):
    """Compact JSON for ediSharedWidgetData; never above 1024 UTF-8 bytes."""
    now = now if now is not None else time.time()
    stats = stats or {}
    status = status or {}
    done = _g(status, "clips_done")
    if done is None:
        done = stats.get("clips_accepted") or 0
    total = _g(status, "clips_expected") or 0
    pct = _g(status, "progress_pct")
    if pct is None and total:
        pct = 100.0 * done / total
    rate = _g(status, "clips_per_min") or stats.get("clips_per_min")
    data = {
        "t": time.strftime("%H:%M", time.localtime(now)),
        "p": round(float(pct), 1) if pct is not None else None,
        "d": done,
        "n": total,
        "r": f"{rate:.1f}" if rate is not None else None,
        "e": human_duration((total - done) / rate * 60) if rate else None,
        "a": human_duration(_g(status, "last_clip_age_s")),
    }
    wer = _g(status, "wer_mean")
    if wer is not None:
        data["w"] = f"{100 * wer:.2f}"
    p95 = _g(status, "wer_p95")
    if p95 is not None:
        data["w9"] = f"{100 * p95:.2f}"
    stress = _g(status, "stress_fail_pct")
    if stress is not None:
        data["s"] = f"{stress:.2f}"
    dropped = _g(status, "dropped")
    if dropped is not None:
        data["dr"] = dropped
    temp = _g(status, "gpu_temp_c")
    used = _g(status, "vram_used_gb")
    total_vram = _g(status, "vram_total_gb")
    gpu = []
    if temp is not None:
        gpu.append(f"{temp}\u00b0C")
    if used is not None:
        gpu.append(f"{used}/{total_vram}GB" if total_vram else f"{used}GB")
    if gpu:
        data["g"] = " ".join(gpu)
    disk = _g(status, "disk_free_gb")
    if disk is not None:
        data["df"] = f"{disk:.0f}GB"
    rate_usd, hours, dollars = spend(pod, status, now)
    if dollars is not None:
        data["c"] = f"${rate_usd:.2f}/h \u00b7 {human_duration(hours * 3600)} \u00b7 ${dollars:.2f}"
    updated = parse_iso(status.get("updated_at"))
    if updated is not None and now - updated > 600:
        data["st"] = "stale"
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if len(text.encode("utf-8")) > WIDGET_STATE_LIMIT:  # drop optional fields
        for key in ("s", "dr", "df", "w9", "g", "c", "st"):
            data.pop(key, None)
            text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            if len(text.encode("utf-8")) <= WIDGET_STATE_LIMIT:
                break
    return text


WIDGET_HTML = """<html><body>
<div class="hdr"><span id="t">\u2014</span> <span class="dim" id="st">waiting</span></div>
<div class="bar"><div id="fill"></div></div>
<div class="big"><span id="p">\u2014</span><span class="dim" id="clips"></span></div>
<div class="grid">
<div><span class="dim">rate</span> <b id="r">\u2014</b></div>
<div><span class="dim">eta</span> <b id="e">\u2014</b></div>
<div><span class="dim">wer</span> <b id="w">\u2014</b></div>
<div><span class="dim">p95</span> <b id="w9">\u2014</b></div>
<div><span class="dim">stress</span> <b id="s">\u2014</b></div>
<div><span class="dim">dropped</span> <b id="dr">\u2014</b></div>
<div><span class="dim">gpu</span> <b id="g">\u2014</b></div>
<div><span class="dim">disk</span> <b id="df">\u2014</b></div>
<div><span class="dim">last clip</span> <b id="a">\u2014</b></div>
<div><span class="dim">spend</span> <b id="c">\u2014</b></div>
</div>
<script>
var W=window.TelegramWidget||{};
function el(i){return document.getElementById(i);}
function put(i,v){if(v!==null&&v!==undefined&&v!=="")el(i).textContent=v;}
function draw(){
  var d={};try{d=JSON.parse(W.publicData||"{}");}catch(e){}
  put("t",d.t);put("st",d.st||"live");
  el("fill").style.width=Math.max(0,Math.min(100,Number(d.p)||0))+"%";
  put("p",d.p!==null&&d.p!==undefined?d.p+"%":"\\u2014");
  put("clips",d.d!==null&&d.d!==undefined?"  "+d.d+" / "+d.n:"");
  put("r",d.r?d.r+"/min":null);put("e",d.e);put("w",d.w?d.w+"%":null);
  put("w9",d.w9?d.w9+"%":null);put("s",d.s?d.s+"%":null);put("dr",d.dr);
  put("g",d.g);put("df",d.df);put("a",d.a);put("c",d.c);
}
if(W.onEvent){W.onEvent("publicDataChanged",draw);}
if(W.watchPublicData){W.watchPublicData();}
draw();
</script>
</body></html>"""

WIDGET_CSS = """
body{margin:2px 0;padding:0;font-size:15px}
.hdr{font-size:13px;margin-bottom:4px}
.bar{height:8px;border-radius:4px;background:rgba(127,127,127,.25);overflow:hidden}
#fill{height:100%;width:0;border-radius:4px;background:var(--tg-theme-button-color,#3390ec);transition:width .3s}
.big{font-size:22px;font-weight:600;margin:4px 0 6px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1px 10px;font-size:13px}
.dim{color:var(--tg-theme-hint-color,#8a8a8a);font-weight:400}
b{font-weight:600}
"""


def widget_html():
    """Inline the CSS because a widget allows no external stylesheet."""
    return WIDGET_HTML.replace("<html><body>", "<html><head><style>" + WIDGET_CSS + "</style></head><body>")


def compress_bound(html=None):
    """Bytes after zlib level 9, the second widget size limit."""
    import zlib
    return len(zlib.compress((html or widget_html()).encode("utf-8"), 9))
