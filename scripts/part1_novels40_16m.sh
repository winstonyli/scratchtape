#!/bin/bash
# Confirm the best config (novels40 16M windows) on fresh-split part 1/4: no dyn (old + tuned kNN weights) and dyn (old weights), one memory build.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
K=runs/novels40_s128_16m_m0.ckpt
tev_begin
tev n40_16m_p1_nodyn $K part=1/4 $(tiers $W_OLD $W_TUNED)
tev n40_16m_p1_dyn $K part=1/4 $DYN $(tiers $W_OLD)
tev_end
echo "$(date +%T) done"
