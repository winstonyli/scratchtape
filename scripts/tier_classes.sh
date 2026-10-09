#!/bin/sh
# Loss by position class (classes=1) for the model alone and for the symbolic tiers, the memory and the whole stack
# (online split memory, whiten=0.5). Spec columns in the table: 0 lexicon+words, 1 memory alone, 2 whole stack. docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt; P=${1:-}
K="knn:256:0.4:30:50:100"
$E $C warm=32 stride=1 $P memory=online store=100000 whiten=0.5 classes=1 tier=lexicon:0.3+words:1:0.25 tier=$K tier=lexicon:0.3+words:1:0.25+$K > runs/tier_classes.log 2>&1
