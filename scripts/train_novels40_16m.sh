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
R="windows=16384000 dropout=0.1 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 d_model=256 d_ff=512 seq=128 score_tail=32 corpus=novels40"
TRAIN_F16=1 fmc novels40_s128_16m 1 16 0.05 $R
run novels40_s128_16m n40_16m_p0 "part=0/4" 128:0.5:30:15:100 128:0.8:30:22:100
echo "$(date +%T) done"
