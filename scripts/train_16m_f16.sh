#!/bin/bash
# The 16M-window recipe in the f16 default (attention f32). Reference nov_big_k1_d0.1_16m: held-out 1.2319, stack 1.0876.
# Expected ~1 h at ~5 ms/step (the f32 run took ~2.4 h). docs/tiers_design.md.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov_big_k1_d0.1_16m_f16 1 32 0.05 windows=16384000 dropout=0.1 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
