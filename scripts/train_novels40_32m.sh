#!/bin/bash
# (1) novels40 at 32M windows (twice the 16M budget), stack on fresh-split parts 0 and 1 (one memory build);
# (3) the 64M-window novels22 model on part 1 (no dyn old+tuned, dyn old) for the cross-model comparison.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
train16 novels40_s128_32m windows=32768000
tev_begin
K=runs/novels40_s128_32m_m0.ckpt
for p in 0 1; do
  tev n40_32m_p${p}_nodyn $K part=$p/4 $(tiers $W_OLD $W_TUNED)
  tev n40_32m_p${p}_dyn $K part=$p/4 $DYN $(tiers $W_OLD)
done
tev_end
K=runs/nov22_s128_k1_d0.1_64m_f16_m0.ckpt
tev n22_64m_p1_nodyn $K part=1/4 $(tiers $W_OLD $W_TUNED)
tev n22_64m_p1_dyn $K part=1/4 $DYN $(tiers $W_OLD)
tev_end
echo "$(date +%T) done"
