#!/bin/bash
# Round 14 (docs/fusion_results.md): big K = 1 at 4M windows on novels6 with dropout 0.1 instead of 0.3
# (control: nov_big_k1_4m, 1.2929).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov_big_k1_d0.1_4m 1 32 0.05 windows=4096000 dropout=0.1 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
