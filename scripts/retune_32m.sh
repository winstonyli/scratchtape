#!/bin/bash
# The retune half of seed_and_retune.sh, rerun (the first grid had k=256, but tier_eval caps k at 128). Grid is in the script below.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
K=runs/novels40_s128_32m_m0.ckpt
G=$(tiers 128:0.8:30:22:100 128:0.9:30:22:100 128:0.7:30:22:100 128:0.8:30:30:100 128:0.8:30:16:100 128:0.8:24:22:100 128:0.8:40:22:100 96:0.8:30:22:100 64:0.8:30:22:100)
tev_begin
for p in 1 0; do tev n40_32m_retune_p$p $K part=$p/4 $G; done
tev_end
echo "$(date +%T) done"
