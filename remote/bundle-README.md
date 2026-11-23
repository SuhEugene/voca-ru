# voca-ru — corpus generation bundle (run on a rented CUDA GPU)

This tarball generates the **teacher corpus** only: 8 000 Russian sentences read
by two designed voices with Qwen3-TTS-1.7B Base + the mandatory Russian stress
LoRA, then screened (Whisper WER, duration/rate, clipping, meter2 stress).
Training is out of scope.

You need a Linux GPU box with an NVIDIA Ampere-or-newer card, ≥16 GB VRAM
(24 GB recommended), ≥32 GB RAM, ≥100 GB free disk, Python 3.12 and a CUDA 12.x
driver. No models are in the tarball; bootstrap downloads ~7.5 GB and verifies
every SHA-256.

## The six commands

```sh
# 1. rent: Ubuntu 22.04/24.04 + CUDA 12.x driver, RTX 4090 24 GB (min) or L40S/A100 40 GB
#    upload this tarball somewhere reachable, then on the box:
tar -xf remote-bundle.tar.zst && cd remote-bundle

# 2. prepare: checks hardware, installs pinned deps, downloads + verifies assets, runs a 1-clip smoke
bash remote/bootstrap.sh

# 3. generate in tmux (full 16 000 clips, resumable). Do the 40-clip smoke first; listen to it.
tmux new -s gen 'bash remote/generate.sh --limit 40 && bash remote/generate.sh'

# 4. monitor in a second tmux window (writes status.json + health.log, alarms loudly)
tmux new -s watch 'bash remote/watch.sh'

# 5. when generate says DONE: verify, write the dataset card and upload (private repo by default)
HF_TOKEN=hf_... bash remote/publish.sh

# 6. ssh out: the corpus is in data/corpus-ru/; `bash remote/publish.sh --dry-run` prints stats
```

`HF_TOKEN` is needed only in step 5. `HF_REPO` selects the target dataset
(default `SuhEugene/voca-ru-corpus`).

## Read next

`remote/README.md` — hardware table, first-10-minutes checklist, troubleshooting,
and what each script does. `remote/generate.sh --help` lists the partial-run
flags.
