#!/bin/sh
# Full held-out split, 4.5M-key memory (k up to 256), word-bigram and lexicon tiers; settings tuned on offset 0 of stride 16
# and confirmed on offset 8 (runs/tier_o8_final.log). docs/tiers_design.md.
runs/tier_eval_run.exe runs/nov_big_k1_d0.1_8m_m0.ckpt expect=1.2408 store=100000 \
  tier=words:1:0.25 tier=lexicon:0.3+words:1:0.25 tier=knn:256:0.4:25 tier=words:1:0.25+knn:256:0.4:25 \
  tier=lexicon:0.3+words:1:0.25+knn:256:0.4:25 > runs/tier_full2.log 2>&1
