#!/bin/bash
# novels40 at 64M windows, continued from the finished 32M checkpoint (constant lr, so the same schedule extended): 32.8M more windows
# with a new seed (the shuffle position is not carried over) and a short warmup. Then dyn-only stack evals on parts 0 and 1 (one memory build).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
train16 novels40_s128_64m windows=32768000 seed=3 init=runs/novels40_s128_32m_m0.ckpt
tev_begin
K=runs/novels40_s128_64m_m0.ckpt
for p in 0 1; do tev n40_64m_p${p}_dyn $K part=$p/4 $DYN $(tiers $W_OLD); done
tev_end
echo "$(date +%T) done"
