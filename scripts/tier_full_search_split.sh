#!/bin/sh
# Full held-out run,, search=split (per-source top-256), finalists from the part=3/6 slice (runs/tier_search_split.log) on the CPU-tier stack.
# Control: merged search 1.1182 at 0.4:15:25:40 (runs/tier_full_doc_split.log). docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
B="lexicon:0.3+words:1:0.25+knn"
S="tier=knn:256:0.4:15:25:50"
for K in 256:0.4:15:25:40 256:0.4:15:25:50 256:0.4:15:25:60 256:0.4:15:25:70 256:0.4:15:40:50 256:0.5:20:40:50 64:0.4:15:25:50; do S="$S tier=$B:$K"; done
$E $C warm=32 stride=1 memory=online search=split store=100000 $S > runs/tier_full_search_split.log 2>&1
