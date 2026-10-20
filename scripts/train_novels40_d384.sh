#!/bin/bash
# NOTE: the eval half omitted d_model/heads/d_ff and failed; evals rerun by scripts/eval_d384.sh.
# Width test: novels40 at d=384 (8 heads, d_ff 768, dropout 0.15, 4 blocks) for 16M windows, same recipe otherwise; stack on fresh-split parts 0 and 1
# (no dyn old + tuned weights, dyn old weights), one memory build. Bar: dyn stack must beat the 16M two-seed mean (0.9834 / 1.0217) by > 0.004.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
train16 novels40_s128_16m_d384 dropout=0.15 d_model=384 heads=8 d_ff=768
tev_begin
K=runs/novels40_s128_16m_d384_m0.ckpt
for p in 0 1; do
  tev n40_d384_p${p}_nodyn $K part=$p/4 $(tiers $W_OLD $W_TUNED)
  tev n40_d384_p${p}_dyn $K part=$p/4 $DYN $(tiers $W_OLD)
done
tev_end
echo "$(date +%T) done"
