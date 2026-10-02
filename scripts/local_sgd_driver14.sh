#!/bin/bash
# Round 15 (docs/gpu_step_design.md): dropout sweep 0.0 and 0.05, big K = 1 at 4M windows on novels6
# (controls: dropout 0.1 = 1.2578, 0.3 = 1.2929).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov_big_k1_d0_4m 1 32 0.05 windows=4096000 dropout=0 $NOV d_model=256 d_ff=512
fmc nov_big_k1_d0.05_4m 1 32 0.05 windows=4096000 dropout=0.05 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
