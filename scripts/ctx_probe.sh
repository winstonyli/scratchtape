#!/bin/bash
# Context-length probe: seq 64/128/256 at 8M windows of 64 bytes (524M training bytes) each; batch and warmup scale
# down with seq so every run has the same tokens per step and the same step count. Held-out CE scores the same
# target bytes for every seq (score_tail=32: the last 32 positions of windows that end every 32 bytes). docs/tiers_design.md.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
for s in 64 128 256; do
  b=$((32 * 64 / s))
  TRAIN_F16=1 fmc ctx_probe_s$s 1 $b 0.05 windows=$((8192000 * b / 32)) dropout=0.1 weight_decay=0.0002 warmup_windows=$((6400 * b / 32)) momentum=0.9 corpus=novels6 d_model=256 d_ff=512 seq=$s score_tail=32
done
echo "$(date +%T) done"
