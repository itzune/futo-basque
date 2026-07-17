#!/usr/bin/env bash
# Watch the 4mr recover run for loss spikes.
# Real loss = logged_loss / 8 (grad_accum). Alert if real > 3.0.
LOG="/root/futo-transformer-basque/finetune/stage_m/progress_recover.log"
THRESHOLD=3.0  # real loss threshold (was 2.47 baseline, spiked to 2.84)
echo "[watcher] monitoring $LOG for real loss > $THRESHOLD"
echo "[watcher] will check every 2 minutes"

while true; do
  if [ ! -f "$LOG" ]; then sleep 5; continue; fi
  # Get latest progress line
  LAST=$(grep "progress" "$LOG" | tail -1)
  if [ -z "$LAST" ]; then sleep 5; continue; fi
  # Extract loss and divide by 8
  RAW=$(echo "$LAST" | grep -oP 'loss=\K[0-9.]+' | head -1)
  if [ -z "$RAW" ]; then sleep 5; continue; fi
  REAL=$(python3 -c "print(f'{$RAW/8:.3f}')")
  STEP=$(echo "$LAST" | grep -oP 'step=\K[0-9]+')
  # Check if done
  if grep -q "\[done\]" "$LOG" 2>/dev/null; then
    echo "[watcher] DONE — training finished. Last real loss: $REAL (step $STEP)"
    break
  fi
  # Alert on spike
  SPIKE=$(python3 -c "print(1 if $REAL > $THRESHOLD else 0)")
  if [ "$SPIKE" = "1" ]; then
    echo "[watcher] ⚠ SPIKE at step $STEP: real=$REAL (threshold $THRESHOLD)"
  else
    echo "[watcher] step=$STEP real=$REAL ✓"
  fi
  sleep 120
done
