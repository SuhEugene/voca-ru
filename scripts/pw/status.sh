#!/bin/bash
# Progress fields for the GAN fine-tune, called by pw over ssh.
#   status.sh corpus   -> percent of FLAC extracted out of 19 217
#   status.sh prepare  -> percent of .pt features written out of 19 217
#   status.sh train    -> 0-98 across both stages (104k -> 164k), 100 when stage 2 is done
# Prints "<percent> <name>"; a missing value is 0, never a cap below 100.
set -u
what=${1:-prepare}
total=19217
log=/workspace/train.log

case "$what" in
corpus)
    n=$(find /workspace/corpus -name '*.flac' 2>/dev/null | wc -l)
    [ "$n" -ge "$total" ] && { echo "100 corpus"; exit 0; }
    echo "$((n * 98 / total)) corpus"
    ;;
prepare)
    n=$(find /workspace/voca/data/ru-corpus/prepared* -name '*.pt' 2>/dev/null | sort -u | wc -l)
    if [ "$n" -ge "$total" ]; then echo "100 prepare"; exit 0; fi
    if grep -q PREPARE_DONE /workspace/prep.log 2>/dev/null; then
        # finished but the count does not add up: let a human look
        echo "99 prepare"; exit 0
    fi
    echo "$((n * 98 / total)) prepare"
    ;;
train)
    step=$(grep -ao '"step": *[0-9]*' "$log" 2>/dev/null | tail -1 | grep -o '[0-9]*$')
    step=${step:-104000}
    # 100 only when both stages are really finished; otherwise the plain
    # progress number, frozen where it stopped. A dead run is is_dead.sh's job.
    if grep -aq "STAGE_DONE stage2" "$log" 2>/dev/null; then
        echo "100 train"
        exit 0
    fi
    # one decimal place so the sidebar moves between polls
    awk -v s="$step" 'BEGIN{ p=(s-104000)*98/60000; if(p<0)p=0; if(p>98)p=98; printf "%.1f train\n", p }'
    ;;
esac
