#!/bin/sh
# search=split (per-source top-256) vs the merged top-256, write=first, per-source weighting; slice (default part=3/6). docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt; P=${1:-part=3/6}
S=""
for K in 256:0.4:15:25:40 256:0.4:15:25:50 256:0.4:15:40:40 256:0.5:20:40:50 256:0.4:15:25:20 256:0.4:15:25:60 64:0.4:15:25:40 32:0.4:15:25:40; do S="$S tier=knn:$K"; done
$E $C warm=32 $P store=100000 memory=online write=first $S > runs/tier_search_merged.log 2>&1
$E $C warm=32 $P store=100000 memory=online write=first search=split $S > runs/tier_search_split.log 2>&1
