#!/bin/bash
# Auto-runs diag_objective.py on each NEW checkpoint from the fixed pretrain run.
# Only diagnoses checkpoint-N once the live training step has PASSED N,
# guaranteeing we test the new (fixed) checkpoint, not the old broken one.
#
# Launch on the server:
#   nohup bash scripts/pretrain/diag_watcher.sh >/dev/null 2>&1 &
cd /root/futo-transformer-basque
echo "[$(date)] watcher started" >> pretrain/diag_watcher.log
while true; do
  cur=$(tail -1 pretrain_fixed_full.log | tr '\r' '\n' | grep -oE '[0-9]+/24000' | head -1 | cut -d/ -f1)
  cur=${cur:-0}
  for N in 5000 10000 15000 20000 24000; do
    ckpt=pretrain/checkpoint-$N
    marker=pretrain/diag_done_fixed_$N
    if [ "$cur" -ge "$N" ] && [ -d "$ckpt" ] && [ ! -f "$marker" ]; then
      echo "[$(date)] step=$cur >= $N : diagnosing $ckpt" >> pretrain/diag_watcher.log
      CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/pretrain/diag_objective.py "$ckpt" 5 >> pretrain/diag_fixed_$N.log 2>&1
      touch "$marker"
    fi
  done
  # stop once all done
  if [ -f pretrain/diag_done_fixed_24000 ]; then
    echo "[$(date)] all checkpoints diagnosed, watcher exiting" >> pretrain/diag_watcher.log
    break
  fi
  sleep 120
done
