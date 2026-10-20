#!/bin/bash
# kNN weights under dyn for the 64M novels40 model: coordinate grid around the old 128:0.5:30:15:100 on part 1, same grid on part 0 to validate (one memory build).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
K=runs/novels40_s128_64m_m0.ckpt
G=$(tiers 128:0.5:30:15:100 128:0.4:30:15:100 128:0.6:30:15:100 128:0.5:30:10:100 128:0.5:30:20:100 128:0.5:24:15:100 128:0.5:40:15:100 128:0.8:30:22:100 96:0.5:30:15:100)
tev_begin
for p in 1 0; do tev n40_64m_dynknn_p$p $K part=$p/4 $DYN $G; done
tev_end
echo "$(date +%T) done"
