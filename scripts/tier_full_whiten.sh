#!/bin/sh
# Full held-out run with whitened keys (whiten=<alpha>, default 0.5), same stack and grid as tier_full_search_split.sh around
# the slice optimum. Control without whitening: 1.1138 (runs/tier_full_search_split.log). docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt; A=${1:-0.5}
B="lexicon:0.3+words:1:0.25+knn"
S="tier=knn:256:0.4:30:50:100"
for K in 256:0.4:30:50:100 256:0.4:22:36:73 256:0.4:15:25:50 256:0.4:38:63:127 256:0.4:30:40:100; do S="$S tier=$B:$K"; done
$E $C warm=32 stride=1 memory=online store=100000 whiten=$A $S > runs/tier_full_whiten_$A.log 2>&1
