#!/bin/bash
# Full 125M-key memory for the d=384 16M model with narrowed keys (whiten_rank 256 = the d=256 key footprint, then 320), parts 0 and 1:
# no dyn old + tuned weights, dyn old weights. Compare with the half-memory d=384 (dyn 0.9727 / 1.0126, rank 256 0.9758 / 1.0175) and the best d=256 (64M windows, 0.9765 / 1.0131).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
M="d_model=384 heads=8 d_ff=768"
K=runs/novels40_s128_16m_d384_m0.ckpt
for r in 256 320; do
  tev_begin
  for p in 0 1; do
    tev full_r${r}_p${p}_nodyn $K $M whiten_rank=$r part=$p/4 $(tiers $W_OLD $W_TUNED)
    tev full_r${r}_p${p}_dyn $K $M whiten_rank=$r part=$p/4 $DYN $(tiers $W_OLD)
  done
  tev_end
done
echo "$(date +%T) done"
