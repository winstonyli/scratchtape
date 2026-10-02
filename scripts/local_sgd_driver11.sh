#!/bin/bash
# Round 12 (docs/fusion_results.md): does averaging help the big model once data is plentiful?
# big K = 1 at 4M windows on novels6, the control for nov_big_k4_h100_a0.1_4m.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov_big_k1_4m 1 32 0.05 windows=4096000 dropout=0.3 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
