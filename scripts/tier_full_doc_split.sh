#!/bin/sh
# Full held-out run,, per-source neighbour weighting knn:256:<lambda>:<temp>:<temp_doc>:<shift>, finalists from the part=3/6 grids
# (runs/tier_doc_split*.log) on the CPU-tier stack, plus the plain stack as the control (1.1274 in tier_full_wf.log). docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
B="lexicon:0.3+words:1:0.25+knn:256"
S="tier=$B:0.4:15 tier=knn:256:0.4:15:25:40"
for K in 0.4:15:25:30 0.4:15:25:40 0.4:15:25:50 0.4:15:40:40 0.5:15:25:50 0.5:20:40:50 0.5:20:25:40; do S="$S tier=$B:$K"; done
$E $C warm=32 stride=1 memory=online store=100000 $S > runs/tier_full_doc_split.log 2>&1
