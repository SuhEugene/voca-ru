# Rules for agents working on voca-ru

Read `docs/plan-ru-qwen.md`, `docs/contracts.md` and `docs/repo-map.md` before starting.

## Cache: never re-download

Source the shared cache first, in every shell:

```sh
. "$(git rev-parse --show-toplevel)/scripts/cache-env.sh"
```

It points `HF_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `TORCH_HOME` and `XDG_CACHE_HOME` at `<repo>/.cache`. Model weights, wheels, uv's toolchains and torch hub downloads are reused by every agent and every later session. The network here is slow; a re-download is a defect. `.cache/` is git-ignored and must never be committed.

Large artefacts that are not caches (teacher downloads, generated corpora, checkpoints, venvs) also live under `<repo>/.cache/models`, `<repo>/.cache/venvs` or the git-ignored `data/`, never in `/tmp`.

## Worktrees and ownership

- Work only inside your assigned worktree, on your assigned branch.
- Touch only the paths your task owns. If you need a change elsewhere, put it in your report.
- Commit early with the task prefix (`R1:`, `F3:`, …), end with a clean `git status`.
- Rebase on `origin/main` before reporting done.

## Never commit

Data, audio, checkpoints, model weights, venvs, `.cache/`, secrets. Tokens come from env vars only.

## Reports

End with `reports/<task-id>.md`: what was done, how to reproduce, numbers, open issues, and what you could not verify. State uncertainty instead of guessing.
