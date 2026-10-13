#!/bin/bash
# The seq-128 nov22 recipe (train_32m_novels22_seq128.sh) with twice the windows (64M windows of 64 bytes-equivalent steps, ~3 h).
# Resumable: rerun to continue from runs/nov22_s128_k1_d0.1_64m_f16.resume.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
TRAIN_F16=1 fmc nov22_s128_k1_d${D:-0.1}_64m_f16 1 16 0.05 windows=32768000 dropout=${D:-0.1} weight_decay=0.0002 warmup_windows=3200 momentum=0.9 corpus=novels22 d_model=256 d_ff=512 seq=128 score_tail=32
echo "$(date +%T) done"
