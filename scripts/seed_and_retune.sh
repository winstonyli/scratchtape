#!/bin/bash
# (1) seed repeat of novels40 16M (seed=2; the original used the default seed 1), stack on fresh-split parts 0 and 1;
# (2) kNN weight retune for the novels40 32M model, no dyn: coordinate grid around 128:0.8:30:22:100 on part 1, same grid on part 0 to validate.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
train16 novels40_s128_16m_seed2 seed=2
tev_begin
K=runs/novels40_s128_16m_seed2_m0.ckpt
for p in 0 1; do
  tev n40_16m_s2_p${p}_nodyn $K part=$p/4 $(tiers $W_OLD $W_TUNED)
  tev n40_16m_s2_p${p}_dyn $K part=$p/4 $DYN $(tiers $W_OLD)
done
tev_end
K=runs/novels40_s128_32m_m0.ckpt
G=$(tiers 128:0.8:30:22:100 128:0.9:30:22:100 128:0.7:30:22:100 128:0.8:30:30:100 128:0.8:30:16:100 128:0.8:24:22:100 128:0.8:40:22:100 96:0.8:30:22:100 64:0.8:30:22:100)
tev_begin
for p in 1 0; do tev n40_32m_retune_p$p $K part=$p/4 $G; done
tev_end
echo "$(date +%T) done"
