#!/bin/bash
# nov22 recipe (train_32m_novels22.sh) at seq 128: same step count and bytes per step (batch 16 windows of 128 bytes),
# so 32M windows of 64 bytes. Held-out scores the same target bytes as the seq-64 run (score_tail=32).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
TRAIN_F16=1 fmc nov22_s128_k1_d${D:-0.1}_32m_f16 1 16 0.05 windows=16384000 dropout=${D:-0.1} weight_decay=0.0002 warmup_windows=3200 momentum=0.9 corpus=novels22 d_model=256 d_ff=512 seq=128 score_tail=32
echo "$(date +%T) done"
