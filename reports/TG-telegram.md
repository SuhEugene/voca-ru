# TG — Telegram alerts and a live corpus widget

Branch `task/tg-bot`, worktree `worktrees/tg-bot`, commits prefixed `TG:`.
Deliberately **not** rebased or merged into `main` (human instruction): the
branch stays as it is.

## 1. What the official docs allow (fetched 2026-11-24)

From <https://core.telegram.org/bots/api> and
<https://core.telegram.org/bots/widgets>:

* Widget creation: **`sendWidget`**. With `is_shared: true` the first send
  creates a shared widget — "one object on the server ... any number of messages
  show it by its identifier". The identifier is `Message.widget.id`.
* Widget update: **`editSharedWidgetData`** (`widget_id`, `public_data`,
  `cache_time`) for a shared widget; **`editMessageWidgetData`** for a static
  one. There is **no method that replaces a widget's HTML** ("no method replaces
  it"); only its state changes.
* Re-sending an existing shared widget: **`sendSharedWidget`** (not needed here).
* A widget is anchored to **a message** (chat state) while its shared state
  belongs to the widget object, not to a chat.
* Rates: a shared widget takes **at most one edit per second**; an edit counts
  as one sent message. `cache_time` (refresh period) is **5–3600 s**, default
  30 s. We use 60 s.
* Sizes: widget HTML **32768 bytes**, and **4096 bytes after zlib-9
  compression**; `public_data` **1024 UTF-8 bytes**; `sendMessage` text
  **1–4096 characters after entities parsing**.
* Production availability: widgets work **only in the private chat with the
  bot** (groups, inline and forwards are test-server only). A bot cannot post
  widgets in channels.
* 429 responses carry `parameters.retry_after`.

So the shared-widget flow the brief asked for **is supported** and is what was
implemented; the message-edit fallback exists only for servers/apps where
`sendWidget` is rejected.

## 2. What was implemented

| Path | Purpose |
|---|---|
| `remote/tg/tgcore.py` | stdlib-only core: escaping, truncation, rate limit, state, HTTP client with 429, error detection, status rendering, widget payload + HTML |
| `remote/tg/notify.sh` / `notify.py` | `notify.sh "text" [LEVEL]`; HTML escaping, 4096-char cut, identical-alert dedupe (default 5 min), max 8/min, HTTP errors to `state/tg-errors.log`, always exit 0; `--ping` optional |
| `remote/tg/watch-errors.sh` / `errorwatch.py` | tails `run.log` + `health.log` (or args); detects traceback, `ERROR`, `FAIL`, non-zero exit, OOM/`CUDA out of memory`, disk-full, `Killed`, health `ALARM:`; 15-line context, hostname, 10-min dedupe; persisted byte offset + 256-byte head hash survive restarts and rotation |
| `remote/tg/status-bot.py` | long-polling bot: `/start`, `/chatid`, `/status`, `/stop`, `/help`; one widget per chat, 60 s refresh, `sendWidget(is_shared=True)` → `editSharedWidgetData(cache_time=60)`, automatic `editMessageText` fallback, widget id persisted, one call to `deleteWebhook` at startup |
| `remote/tg/hook.sh` | sourced by pipeline scripts; starts the watcher and reports a non-zero exit once; no-op unless `TG_CHAT_ID`, `python3` and the watcher exist |
| `remote/generate.sh` | 8-line guarded hook after `mkdir -p "$OUT"` |
| `remote/bootstrap.sh` | 8-line guarded hook; tees output to `.cache/logs/bootstrap.log` so the watcher can see it |
| `remote/tg/README.md` | pod setup, chat id, alert meanings, env vars, exact API limits |
| `tests/tg/` | 76 tests, no network |

Dashboard content (status message and widget): clips done/total, %, clips/min,
ETA, WER mean/p95, stress-failure %, dropped, GPU temp/VRAM (when
`status.json` has them), disk free, last-clip age, and spend from
`.cache/runpod-pod.json` (`cost × hours since created`). Header uses the pod's
local time. One widget per chat; ids live in `remote/tg/state.json`.

Runtime state is git-ignored: `remote/tg/state.json`, `remote/tg/state/`
(rate limit, offsets, `tg-errors.log`). The token is read from `$TG_TOKEN` /
`$TG_TOKEN_FILE` / `~/.secrets/telegram-bot-token`, is never committed or
printed, and is masked in every log line.

## 3. Tests

```sh
cd worktrees/tg-bot
python3 -m unittest discover -s tests/tg -v     # 76 passed
python3 -m unittest tests.remote.test_shell -v  # 6 passed (pipeline untouched)
```

Covered: escaping/truncation at 4096 (multibyte), alert layout, duplicate and
N/min throttling with a fake clock, state persistence across instances, error
detection on real D2/remote log shapes plus negative cases, tail offsets across
restart, in-place truncation and rotation, partial lines, dedupe expiry, 15-line
context, 429 `retry_after` (and the 30 s cap and give-up path), token masking,
widget payload ≤1024 bytes and HTML within both size limits, status rendering
from the real `status.json`/`run-stats.json`/`runpod-pod.json` shapes, widget
creation/persistence/refresh, `/stop`, deleted-widget resend, message fallback.

A local HTTP server was used once by hand (not in the suite) to confirm
`notify.sh` → JSON → HTML escaping end to end on `127.0.0.1`; no external
Telegram request was made.

## 4. Not verified — needs a live chat

* A real `sendWidget` / `editSharedWidgetData` round trip, and how the widget
  actually renders on iOS/Android/Desktop. The docs' production rules were
  followed; if the token's server rejects widgets, the bot automatically uses
  the edited-message fallback (tested).
* `--ping` was **not** run: `remote/tg/state.json` does not exist here, so there
  is no saved chat id; per the brief it was skipped.
* `hook.sh` was syntax-tested and its functions are unit-shaped through the
  watcher tests, but `bootstrap.sh`/`generate.sh` were not run on a CUDA box
  with `TG_CHAT_ID` set. On this Mac `bootstrap.sh` stops at preflight as before.
* Spend assumes `.cache/runpod-pod.json` is the file `scripts/runpod-grab.sh`
  writes (`cost` is $/h, `created` ISO-8601); the shape was checked against the
  real local file.
* Telegram message limits other than those quoted (e.g. per-chat app-level
  throttling) are delegated to the 429 handler.

## 5. Run on the pod

```sh
# once: token outside the repo
install -m 600 /dev/null ~/.secrets/telegram-bot-token
printf '%s' '<token>' > ~/.secrets/telegram-bot-token

# status bot; then send /start to it, which saves the chat id
tmux new -s tgbot 'python3 remote/tg/status-bot.py'
# or: nohup python3 remote/tg/status-bot.py >> .cache/logs/tg-bot.log 2>&1 &

# export the saved id for alerts in the generation shell
export TG_CHAT_ID="$(python3 -c 'import json;print(json.load(open("remote/tg/state.json"))["chat_id"])')"

# generation: starts the error watcher itself when TG_CHAT_ID is set
tmux new -s gen 'bash remote/generate.sh'

# if you prefer a separate watcher (do not run both):
nohup bash remote/tg/watch-errors.sh >> .cache/logs/tg-watch.log 2>&1 &

# health loop, unchanged
tmux new -s watch 'bash remote/watch.sh'
```

Optional connectivity check, only with a saved chat id:

```sh
bash remote/tg/notify.sh --ping
```

## 6. Open issues / decisions

* One widget per chat, not a global shared widget broadcast to many chats: the
  brief says "one widget per chat"; `sendSharedWidget` could fan one object out
  later if ever needed.
* `status.json` from `remote/lib/health.py` has no `vram_total_gb`, so VRAM is
  shown as used-only ("21.3 GB"). No change to the health script was made.
* The watcher and the pipeline's EXIT hook can both report the same fatal run
  (different messages, both rate-limited); the watcher's dedupe covers repeats
  of the log line, not the hook summary. Deliberate: a stopped run should be
  loud.
