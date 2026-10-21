#!/bin/bash
# kNN weights under dyn for the 6-block d=512 model (dyn 0.9409 / 0.9820 at the old 128:0.5:30:15:100): usage dyn_knn_b6.sh W...; part 1 first, then part 0 to validate
# (one memory build, whiten_rank=256; ROUND=r2 tags the logs). Round 1 (the 64M grid of scripts/dyn_knn_64m.sh): temp_doc 10 won by 0.0033 / 0.0029 (parts 1 / 0). Round 2: 128:0.4:30:7:100 wins, 0.9782 / 0.9366.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
K=runs/novels40_s128_16m_d512_b6_m0.ckpt
M="d_model=512 heads=8 d_ff=1024 blocks=6 whiten_rank=256"
G=$(tiers "${@:-128:0.5:30:15:100}")   # weights from the command line; round 1 was 128:0.5:30:15:100 128:0.4:30:15:100 128:0.6:30:15:100 128:0.5:30:10:100 128:0.5:30:20:100 128:0.5:24:15:100 128:0.5:40:15:100 128:0.8:30:22:100 96:0.5:30:15:100
tev_begin
for p in 1 0; do tev n40_d512_b6_dynknn${ROUND:+_$ROUND}_p$p $K $M part=$p/4 $DYN $G; done
tev_end
echo "$(date +%T) done"
