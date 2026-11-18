# Teacher generation (Qwen3-TTS on MPS)

Everything here runs on the M5 Pro with PyTorch MPS in bfloat16. The cache is
shared by all worktrees and lives at the **main repo root**
(`<main-repo>/.cache`), not inside `worktrees/<task>/.cache`.

```sh
# 1. environment (the verified venv already exists at .cache/venvs/r4)
uv venv --python 3.12 .cache/venvs/g1
uv pip install --python .cache/venvs/g1/bin/python -r env/requirements-qwen.txt
make doctor                      # versions, MPS, model cache

# 2. fetch checkpoints into the project cache (Base is already there)
./scripts/gen/fetch-voicedesign.sh   # 3.83 GB, parallel ranges, sha256-checked

# 3. smoke: 6 batch1 sentences, transcript-free clone of clone_ref.wav
.cache/venvs/r4/bin/python scripts/gen/teacher.py smoke

# 4. four VoiceDesign candidate voices (2 F, 2 M), fixed seeds
.cache/venvs/r4/bin/python scripts/gen/teacher.py design

# 5. 8-clip batch with the best candidate reference (reference + transcript)
.cache/venvs/r4/bin/python scripts/gen/teacher.py batch f_warm
```

Outputs are git-ignored:

| Path | What |
|---|---|
| `out/g1/base/*.wav` + `base-metrics.json` | smoke clips, 24 kHz mono |
| `out/g1/voices/<name>.wav` / `.txt` | reference clip + exact transcript |
| `out/g1/voices-metrics.json` | per-voice duration, peak, RTF, seed, description |
| `out/g1/batch8/*.wav` + `batch8-metrics.json` | batch throughput check |

Notes:

- `clone_ref.wav` in the model cache has **no transcript**, so the smoke run
  uses `x_vector_only_mode=True` (the path R4 verified). The VoiceDesign
  references do carry exact transcripts and are the better cloning prompt.
- `data/scripts-ru/batch*.jsonl` contain **no digits**: numbers and dates are
  spelled out by the text contract. `number` in the smoke set means a spelled
  numeral.
- Seeds are fixed with `torch.manual_seed` right before each generation call;
  they are recorded in the metrics JSON.
- `sox` and `flash-attn` warnings are expected and harmless on macOS.
