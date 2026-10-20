#!/bin/bash
# After the kNN retune (docs/tiers_design.md "kNN weights retuned per model"): (1) full stack with dyn and the tuned weights on the
# 64M-window novels22 model and the 4M novels40 model, part 0; (2) novels40 at 16M windows (~85 min), then its no-dyn stack on part 0.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
run() { # ckpt log parts-and-extra weights...
  local ck=$1 log=$2 extra=$3; shift 3
  [ -f runs/$log.done ] && return
  tev_begin
  tev $log runs/${ck}_m0.ckpt $extra $(tiers "$@")
  tev_end; echo done > runs/$log.done
}
run nov22_s128_k1_d0.1_64m_f16 dyn64 "part=0/4 $DYN" 128:0.5:30:15:100 128:0.8:30:22:100 128:0.65:30:22:100
run novels40_s128_4m dyn40 "part=0/4 $DYN" 128:0.5:30:15:100 128:0.8:30:22:100 128:0.65:30:22:100
train16 novels40_s128_16m
run novels40_s128_16m n40_16m_p0 "part=0/4" 128:0.5:30:15:100 128:0.8:30:22:100
echo "$(date +%T) done"
