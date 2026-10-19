#!/bin/bash
# d=384 16M-window novels40 model at another seed (same recipe as train_novels40_d384.sh); usage: train_d384_seed.sh <seed>
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
s=$1
R="windows=16384000 dropout=0.15 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 d_model=384 heads=8 d_ff=768 seq=128 score_tail=32 corpus=novels40 seed=$s"
TRAIN_F16=1 fmc novels40_s128_16m_d384_seed$s 1 16 0.05 $R
echo "$(date +%T) done seed $s"
