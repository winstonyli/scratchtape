#!/bin/bash
# Key narrowing (`whiten_rank` now narrows the stored keys): half-memory checks on the 16M novels40 models, parts 0 and 1, no dyn old + tuned, dyn old.
#  A: d=384 with whiten_rank=256 against the unnarrowed half-memory numbers (dyn 0.9727 / 1.0126): the cost of narrowing.
#  B: d=256 with the default rank against the pre-change half-memory numbers (dyn 0.9901 / 1.0291): the rewritten whitener must be neutral.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
STORE=500000
run() { # tag ckpt extra-args
  tev_begin
  for p in 0 1; do
    tev rank_${1}_p${p}_nodyn $2 $3 part=$p/4 $(tiers $W_OLD $W_TUNED)
    tev rank_${1}_p${p}_dyn $2 $3 part=$p/4 $DYN $(tiers $W_OLD)
  done
  tev_end
}
run d384r256 runs/novels40_s128_16m_d384_m0.ckpt "d_model=384 heads=8 d_ff=768 whiten_rank=256"
run d256 runs/novels40_s128_16m_m0.ckpt ""
echo "$(date +%T) done"
