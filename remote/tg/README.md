# Telegram alerts and the live status widget

Optional. Everything here is inert unless `TG_CHAT_ID` (alerts) or a saved chat
id (bot) is present, so the pipeline works exactly as before without Telegram.

| File | What it is |
|---|---|
| `notify.sh` / `notify.py` | one-shot alert: escape, truncate, dedupe, throttle, fail soft |
| `watch-errors.sh` / `errorwatch.py` | tails logs and alerts on new errors |
| `status-bot.py` | long-polling bot: `/status` posts a live widget or an edited message |
| `hook.sh` | the optional EXIT/tee hook sourced by `generate.sh` / `bootstrap.sh` |
| `tgcore.py` | all pure logic: formatting, state, rate limit, error detection, widget |

## What the Bot API actually allows (read from the docs, 2026-11)

* **`sendWidget`** posts a message with an HTML widget. With `is_shared: true`
  the first send **creates a shared widget**: "one object on the server ... any
  number of messages show it by its identifier". The returned `Message.widget.id`
  is the identifier.
* **`editSharedWidgetData`** changes `public_data` by `widget_id`; this is the
  documented way to update a shared widget, and the only way to set its refresh
  period via `cache_time`. A shared widget takes **at most one edit per second**;
  `cache_time` is **5-3600 s** (60 s here, its default is 30 s).
* **`sendSharedWidget`** re-sends an existing shared widget by id (not needed:
  this bot keeps one message per chat).
* **`editMessageWidgetData`** changes a static widget's data per message.
* A widget is **anchored to a message** (and its shared state to the widget
  object, not the chat). Widget state is what changes; the HTML is immutable
  ("no method replaces it").
* Widget HTML: **32768 bytes**, and **4096 bytes after zlib-9 compression**.
  `public_data`: **1024 UTF-8 bytes**. Widgets are offline (no `fetch`).
* Production availability: widgets work **only in the private chat with the
  bot**; groups/inline/forwards are test-server only. That is exactly this use
  case.
* `sendMessage` text: **1-4096 characters after entities parsing**; with
  `parse_mode=HTML` only `&`, `<`, `>` need escaping. Bots may send ~30
  messages/second; a shared-widget edit counts as one sent message.
* 429 responses carry `parameters.retry_after`; `tgcore` sleeps that many
  seconds (capped at 30) and retries up to three times.

If `sendWidget` fails (older server, app without widget support), `status-bot.py`
falls back to **one plain message per chat edited with `editMessageText` every
60 s** — visually the same live dashboard.

## First-time setup on the pod

1. Get the token onto the box **outside the repo**:

   ```sh
   install -m 600 /dev/null ~/.secrets/telegram-bot-token
   printf '%s' '<token>' > ~/.secrets/telegram-bot-token   # or scp it
   ```

   `notify.py` reads `$TG_TOKEN` first, then `$TG_TOKEN_FILE`, then
   `~/.secrets/telegram-bot-token`. It never prints or logs the token.

2. Start the bot and message it:

   ```sh
   tmux new -s tgbot 'python3 remote/tg/status-bot.py'
   ```

3. Send `/start` to the bot (or `/chatid`). It replies with the chat id and
   writes it to `remote/tg/state.json`. That id is also the id used by
   `notify.sh` when `TG_CHAT_ID` is unset.

4. Export the id for the pipeline (in the same shell / tmux session):

   ```sh
   export TG_CHAT_ID="$(python3 -c 'import json;print(json.load(open("remote/tg/state.json"))["chat_id"])')"
   ```

## Running the two services

```sh
# status bot (widget refreshes every 60 s)
nohup python3 remote/tg/status-bot.py >> .cache/logs/tg-bot.log 2>&1 &

# error watcher, if you run generate.sh outside a Telegram-enabled shell
nohup bash remote/tg/watch-errors.sh >> .cache/logs/tg-watch.log 2>&1 &
```

Or in tmux (easier to read):

```sh
tmux new -s gen    'bash remote/generate.sh'      # starts the watcher itself
tmux new -s tgbot  'python3 remote/tg/status-bot.py'
```

**Do not run the watcher twice.** `generate.sh` starts it when `TG_CHAT_ID` is
set; a second copy would send the same alerts. The watcher's offsets and dedupe
state live in `.cache/`-style state under `remote/tg/state/`, so a restart does
not replay old lines.

Commands in the bot: `/status` starts/refreshes the widget, `/stop` stops
refreshing, `/start` and `/chatid` show the chat id.

## Alerts

`watch-errors.sh` tails `data/corpus-ru/run.log` and `health.log` (or the files
given on the command line) and sends one message with the hostname, the matching
line and the last 15 lines of context:

| Trigger | Meaning |
|---|---|
| `Traceback (most recent call last)` | Python crash; the pipeline stopped |
| `ERROR` | explicit error line from a dependency or our code |
| `FAIL ...` | our scripts' failure line (e.g. `FAIL generate.sh: pipeline exited 1`) |
| non-zero exit (`returncode=1`, `exited 1`) | a child process failed |
| `CUDA out of memory` / `OutOfMemoryError` | lower `--batch-size` |
| `Killed` / `OOMKilled` | host OOM killer |
| `No space left on device` | disk full |
| `ALARM:` (health.log) | slow/stalled generation, WER, stress, disk or temperature |

Repeats of the same error are coalesced for 10 minutes (`TG_ERROR_DEDUPE_SECONDS`).
`generate.sh` / `bootstrap.sh` also report their own non-zero exit through the
hook, once.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `TG_CHAT_ID` | from `state.json` | recipient chat |
| `TG_TOKEN` / `TG_TOKEN_FILE` | `~/.secrets/telegram-bot-token` | bot token |
| `TG_STATE` | `remote/tg/state.json` | bot state (chat id, offset, widget ids) |
| `TG_STATE_DIR` | `remote/tg/state` | rate-limit, offsets, `tg-errors.log` |
| `TG_MAX_PER_MIN` | `8` | hard cap for `notify.sh` |
| `TG_DEDUPE_SECONDS` | `300` | identical alerts within this window are dropped |
| `TG_ERROR_DEDUPE_SECONDS` | `600` | same for the log watcher |
| `TG_CORPUS_DIR` | `data/corpus-ru` | where `status.json` / logs live |
| `TG_POD_JSON` | `.cache/runpod-pod.json` | `cost` + `created` for spend |
| `TG_API_BASE` | `https://api.telegram.org` | override for local testing |
| `VOCA_ROOT` | repo root | base for `data/` and `.cache/` |

Per-chat state: one widget record per chat in `state.json`
(`kind`, `widget_id`/`message_id`, `active`). `tg-errors.log` records every HTTP
error (token masked). Nothing in `remote/tg/state*` is committed.

## Tests

```sh
python3 -m unittest discover -s tests/tg -v
```

No network: HTTP is replaced by a fake transport. Shell tests run
`bash -n`, check the disabled path, and confirm `generate.sh --dry-run` still
works with Telegram unset.

Optional one-off connectivity check (only if a chat id is already saved):

```sh
bash remote/tg/notify.sh --ping
```
