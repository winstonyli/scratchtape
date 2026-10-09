#!/bin/bash
# f16 matrix-core training (TRAIN_F16=1) at the 8M-window recipe: compare held-out CE with the f32 run
# nov_big_k1_d0.1_8m (1.2408, 4312 s, 11.8 ms/step contended). docs/tiers_design.md.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
TRAIN_F16=1 fmc nov_big_k1_d0.1_8m_f16 1 32 0.05 windows=8192000 dropout=0.1 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
