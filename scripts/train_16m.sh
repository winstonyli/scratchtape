#!/bin/bash
# Longer training at the same size: the 8M-window recipe (nov_big_k1_d0.1_8m, held-out 1.2408, warm=32 stack 1.0956) at
# 16M windows. Held-out CE at 1M-window marks in the 8M run: 1.3349 1.2870 1.2684 1.2578 1.2518 1.2475 1.2430 1.2408.
# Expected ~2.4 h (the 8M run took 4312 s). docs/tiers_design.md.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov_big_k1_d0.1_16m 1 32 0.05 windows=16384000 dropout=0.1 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
