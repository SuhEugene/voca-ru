#!/bin/bash
# Supervisor for the two-stage GAN fine-tune: keeps the chain alive until both
# stage checkpoints exist, restarting after any crash or early exit.
#   bash supervise-ft.sh
# Each stage resumes from runs/ft/last.pt, so a restart loses at most one step.
set -u
cd /workspace/voca || exit 1
art=/workspace/artifacts

while :; do
    if [ -f "$art/stage2/DONE" ]; then
        echo "[supervisor] both stages done" >> /workspace/supervise.log
        break
    fi
    if [ -f "$art/stage1/DONE" ]; then
        stage=stage2
    else
        stage=stage1
    fi
    echo "[supervisor] $(date -u +%FT%TZ) start $stage" >> /workspace/supervise.log
    bash gan-finetune.sh "$stage" >> /workspace/train.log 2>&1
    code=$?
    echo "[supervisor] $(date -u +%FT%TZ) $stage exited $code" >> /workspace/supervise.log
    sleep 10
done
