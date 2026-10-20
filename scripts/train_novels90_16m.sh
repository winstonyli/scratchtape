#!/bin/bash
# novels90 at 16M windows (the 40-vs-90 question at full training), then its fresh-split stack on part 0, no dyn and dyn (one memory build).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
R="windows=16384000 dropout=0.1 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 d_model=256 d_ff=512 seq=128 score_tail=32 corpus=novels90"
TRAIN_F16=1 fmc novels90_s128_16m 1 16 0.05 $R
tev_begin
K=runs/novels90_s128_16m_m0.ckpt
tev n90_16m_nodyn $K part=0/4 $(tiers $W_OLD $W_TUNED)
tev n90_16m_dyn $K part=0/4 $DYN $(tiers $W_OLD)
tev_end
echo "$(date +%T) done"
