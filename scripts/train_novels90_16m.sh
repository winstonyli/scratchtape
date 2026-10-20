#!/bin/bash
# novels90 at 16M windows (the 40-vs-90 question at full training), then its fresh-split stack on part 0, no dyn and dyn (one memory build).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
train16 novels90_s128_16m corpus=novels90
tev_begin
K=runs/novels90_s128_16m_m0.ckpt
tev n90_16m_nodyn $K part=0/4 $(tiers $W_OLD $W_TUNED)
tev n90_16m_dyn $K part=0/4 $DYN $(tiers $W_OLD)
tev_end
echo "$(date +%T) done"
