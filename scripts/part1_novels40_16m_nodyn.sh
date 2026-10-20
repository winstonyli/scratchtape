#!/bin/bash
# Rerun of the no-dyn half of part1_novels40_16m.sh (the first attempt hit a memory-allocation failure).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
tev_begin
tev n40_16m_p1_nodyn runs/novels40_s128_16m_m0.ckpt part=1/4 $(tiers $W_OLD $W_TUNED)
tev_end
echo "$(date +%T) done"
