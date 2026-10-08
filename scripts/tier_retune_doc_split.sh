#!/bin/sh
# Retune under write=first + per-source weighting (full held-out, one shared search): knn weight {0.35,0.4,0.45} x temp {15,20} x
# (temp_doc,shift) {(25,40),(40,40),(25,50)}, words weight 0.25; plus words 0.19 at three of them. docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
S=""
for L in 0.35 0.4 0.45; do for T in 15 20; do for D in 25:40 40:40 25:50; do S="$S tier=lexicon:0.3+words:1:0.25+knn:256:$L:$T:$D"; done; done; done
for K in 0.4:15:25:40 0.4:20:25:40 0.35:20:25:40; do S="$S tier=lexicon:0.3+words:1:0.19+knn:256:$K"; done
$E $C warm=32 stride=1 memory=online write=first store=100000 $S > runs/tier_retune_doc_split.log 2>&1
