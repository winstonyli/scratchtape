#!/bin/bash
# TRIME falsifier (docs/tiers_design.md "TRIME-style memory-augmented training"): the nov22 seq-128 recipe at 4M windows, plain
# loss vs memory=1, same seed. ~20 min and ~25 min. Resumable: rerun to continue.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
R="windows=4096000 dropout=0.1 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 corpus=novels22 d_model=256 d_ff=512 seq=128 score_tail=32"
TRAIN_F16=1 fmc nov22_s128_4m_plain 1 16 0.05 $R
TRAIN_F16=1 fmc nov22_s128_4m_trime 1 16 0.05 $R memory=1
echo "$(date +%T) done"
