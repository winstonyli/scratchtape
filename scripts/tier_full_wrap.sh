#!/bin/sh
# Full held-out run with the line-wrap tier (wrap:<lambda>) before and after the memory, on the best stack (whiten=0.5,
# knn:256:0.4:30:50:100). Control without wrap: 1.1118 (runs/tier_full_whiten.log). docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
B="lexicon:0.3+words:1:0.25"; K="knn:256:0.4:30:50:100"
S="tier=$B+$K"
for L in 0.8 0.95; do S="$S tier=$B+wrap:$L+$K tier=$B+$K+wrap:$L"; done
$E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 classes=1 $S > runs/tier_full_wrap.log 2>&1
