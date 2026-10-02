#!/bin/bash
# Round 17 (docs/fusion_results.md): big K = 1 with dropout 0.1 at 8M windows on novels6, where does it flatten?
# (controls: dropout 0.1 at 4M = 1.2578; dropout 0.3 at 8M = 1.2622).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov_big_k1_d0.1_8m 1 32 0.05 windows=8192000 dropout=0.1 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
