#!/bin/bash
# Matched half-memory comparison: the d=384 memory (125M keys) does not fit on the 16 GB card, so d=384 and the d=256 16M seed-1 baseline
# are both scored with store=500000 train windows (~64M keys). Parts 0 and 1: no dyn old + tuned weights, dyn old weights. Bar: d=384 dyn beats d=256 by > 0.004.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
STORE=500000
run() { # tag ckpt extra-args
  tev_begin
  for p in 0 1; do
    tev half_${1}_p${p}_nodyn $2 $3 part=$p/4 $(tiers $W_OLD $W_TUNED)
    tev half_${1}_p${p}_dyn $2 $3 part=$p/4 $DYN $(tiers $W_OLD)
  done
  tev_end
}
run d384 runs/novels40_s128_16m_d384_m0.ckpt "d_model=384 heads=8 d_ff=768"
run d256 runs/novels40_s128_16m_m0.ckpt ""
echo "$(date +%T) done"
