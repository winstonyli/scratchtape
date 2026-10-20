#!/bin/bash
# nov22 recipe (train_32m_novels22.sh) at seq 128: same step count and bytes per step (batch 16 windows of 128 bytes),
# so 32M windows of 64 bytes. Held-out scores the same target bytes as the seq-64 run (score_tail=32).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
train16 nov22_s128_k1_d${D:-0.1}_32m_f16 corpus=novels22 dropout=${D:-0.1}
echo "$(date +%T) done"
