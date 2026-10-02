#!/bin/bash
# Round 13 (docs/gpu_step_design.md): big K = 1 at 8M windows on novels6, where does the curve flatten?
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov_big_k1_8m 1 32 0.05 windows=8192000 dropout=0.3 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
