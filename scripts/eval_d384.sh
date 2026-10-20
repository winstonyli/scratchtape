#!/bin/bash
# Stack evals for the d=384 model (the first attempt in train_novels40_d384.sh omitted the model-size args, so tier_eval rejected the checkpoint).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
M="d_model=384 heads=8 d_ff=768"
K=runs/novels40_s128_16m_d384_m0.ckpt
tev_begin
for p in 0 1; do
  tev n40_d384_p${p}_nodyn $K $M part=$p/4 $(tiers $W_OLD $W_TUNED)
  tev n40_d384_p${p}_dyn $K $M part=$p/4 $DYN $(tiers $W_OLD)
done
tev_end
echo "$(date +%T) done"
