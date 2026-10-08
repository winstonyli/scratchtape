#!/bin/sh
# Grid under (one search shared by all specs): words weight {0.25,0.19} x knn weight {0.4,0.5,0.6} x temp {10,15,20,25}.
# Full held-out, warm=32, stride 1. docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
S=""
for W in 0.25 0.19; do for L in 0.4 0.5 0.6; do for T in 10 15 20 25; do
  S="$S tier=lexicon:0.3+words:1:$W+knn:256:$L:$T"
done; done; done
$E $C warm=32 stride=1 memory=online store=100000 $S > runs/tier_retune_wf.log 2>&1
