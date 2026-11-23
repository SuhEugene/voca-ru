# TG — Telegram alerts and a live corpus widget

Branch `task/tg-bot`, worktree `worktrees/tg-bot`, commits prefixed `TG:`.
Deliberately **not** rebased or merged into `main` (human instruction): the
branch stays as it is.

## 1. Widgets in the docs: methods, parameters, mechanism

Sources re-read on 2026-11-24: the Bot API page and the individual method/type
pages, plus the 10.4 changelog entry. `https://core.telegram.org/bots/widgets`
is a **documentation page**, not a running widget instance; nothing below is
evidence that widgets are live on the production server. The changelog states
the methods were added in **Bot API 10.4 (November 9, 2026)**; the API page
currently reports 10.5 (November 22, 2026).

### 1.1 Exact methods and documented parameter names

All four exist on the API pages; nothing implemented here is invented, and
nothing had to be deleted for being absent.

| Method | Documented parameters (as written on the page) | Used here |
|---|---|---|
| `sendWidget` | `chat_id` (Integer or String, required), `message_thread_id`, `html` (String, required, 1–32768 bytes, ≤4096 compressed zlib max), `width`, `height`, `public_data` (String, 0–1024 bytes), `chat_data` (0–1024 bytes), `caption` (0–1024 chars), `parse_mode`, `caption_entities`, `show_caption_above_media`, `allow_queries`, `is_shared`, `disable_notification`, `protect_content`, `allow_paid_broadcast`, `message_effect_id`, `reply_parameters`, `reply_markup` | `chat_id`, `html`, `width=0`, `height=0`, `public_data`, `is_shared=True`, `caption` |
| `sendSharedWidget` | `chat_id`, `message_thread_id`, `widget_id` (String, required), `chat_data`, `caption`, `parse_mode`, `caption_entities`, `show_caption_above_media`, `disable_notification`, `protect_content`, `allow_paid_broadcast`, `message_effect_id`, `reply_parameters`, `reply_markup` | not used (one widget per chat) |
| `editSharedWidgetData` | `widget_id` (String, required), `public_data` (String, required, 0–1024 bytes; empty string removes it), `cache_time` (Integer, optional, **5–3600**, default 30) | `widget_id`, `public_data`, `cache_time=60` |
| `editMessageWidgetData` | `chat_id`, `message_id`, `inline_message_id`, `public_data`, `chat_data` | not used (the shared-widget path edits by `widget_id`) |

Type `Widget`: `id` (String, *optional* — "omitted if the widget isn't shared"),
`width`, `height`, `html`, `public_data`, `chat_data`. The changelog also adds
the field `widget` to `Message`, which is where `message.widget.id` comes from.

Exact spellings matter: the parameter is **`public_data`**, the method is
**`editSharedWidgetData`** (and `editMessageWidgetData`), and the period knob is
**`cache_time`**. Method names are case-insensitive per the API page; the code
sends the documented camelCase.

### 1.2 How the widget block actually works (per the documentation)

Yes — the widget block is **HTML content rendered from `public_data` state**.
The documented mechanism is:

* `sendWidget` stores the **HTML as fixed code**; the widgets page says "The
  code, its HTML, is set when the widget is sent and never changes" and "no
  method replaces it". The HTML we send is a small document whose JavaScript
  reads `TelegramWidget.publicData`, parses the JSON and redraws itself.
* `editSharedWidgetData` replaces the **shared state** (our compact JSON:
  percent, counts, rate, ETA, WER, disk, spend) for every message showing the
  shared widget. The message HTML is not resent and does not change.
* The running widget learns about the change through
  `publicDataChanged` after calling `TelegramWidget.watchPublicData()`; the docs
  say the app then polls the shared widget at its refresh period, which is the
  `cache_time` we set to 60 s. Clients that do not support widgets show the
  caption instead.
* If the real mechanism differs from this documentation, the live check in
  §1.4 is what will reveal it.

### 1.3 Fallback path, proven by tests

If `sendWidget` returns an error (e.g. unknown method on the server), the bot
creates exactly **one plain message per chat** and refreshes it with
`editMessageText` every 60 s. The record in `state.json` changes `kind` from
`widget` to `message` and keeps its `message_id`; later refreshes edit that same
message.

Existing tests (ran again for this report, all `ok`):

```text
tests.tg.test_bot.BotTests.test_widget_unavailable_falls_back_to_a_message ... ok
tests.tg.test_bot.BotTests.test_message_fallback_edits_the_same_message ... ok
tests.tg.test_client.ClientTests.test_widget_methods_use_documented_parameters ... ok
```

The first asserts the call sequence is exactly `sendWidget` → `sendMessage` and
that the stored record is `kind="message"`; the second advances the fake clock
61 s and asserts the only follow-up is `editMessageText` on the **same**
`message_id` (no second `sendMessage`).

### 1.4 Live behaviour is still unverified

Nothing here has been exercised against the real production server, because no
chat id was saved in `remote/tg/state.json`, so no `/status` was sent and
`--ping` was skipped. The code is documented-API-correct but unproven in
production. What would prove it:

1. send `/status`; the Bot API response must be a `Message` carrying
   `message.widget.id` (the `Widget.id` field is optional and present only for a
   shared widget), and the bot must persist that id in `state.json`;
2. one later `editSharedWidgetData` call for that `widget_id` must return `True`
   with HTTP 200;
3. on screen, the widget must visibly change within `cache_time` (60 s) while
   the message HTML stays the same.

If step 1 returns "unknown method" or another error, the fallback in §1.3 takes
over automatically.

**Correction to the earlier reading:** the widgets page is documentation. It is
not proof that any widget is live, and this report does not claim otherwise.


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

See §1.4 for the exact live proof required. In short:

* No real `sendWidget` / `editSharedWidgetData` round trip, and no view of the
  widget on a real client. The documentation was followed, not exercised; the
  fallback is tested.
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
