#!/bin/sh
# Best stack on the 32M-window checkpoint trained on novels22.
# same checkpoint with the lexicon, words and memory built from the novels22 train split (the plain script uses the default corpus=novels6 for them); survey=1 adds the CE breakdown by word kind (docs/tiers_design.md)
E=runs/tier_eval_survey.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
B="lexicon:0.3+words:1:0.25"; K="knn:256:0.4:30:50:100"
$E $C warm=32 corpus=novels22 stride=1 memory=online store=100000 whiten=0.5 classes=1 survey=1 tier=$B+$K tier=$B+$K+wrap:0.95 > runs/tier_full_32m_novels22_corpus22_recency.log 2>&1
