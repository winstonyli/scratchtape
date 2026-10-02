#!/bin/sh
# docs/tiers_design.md, "fair-context protocol": warm=32 (every scored byte has >= 32 bytes of context), every other
# held-out window scored (stride 2). Causal in-document memory (train keys + past-only held-out windows of the same
# book) vs the flat train-only memory, and the stacks with the CPU tiers. Settings tuned on offset 0 of stride 16.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
$E $C warm=32 stride=2 memory=causal store=100000 \
  tier=lexicon:0.3 tier=words:1:0.25 tier=lexicon:0.3+words:1:0.25 tier=knn:256:0.5:15 \
  tier=words:1:0.25+knn:256:0.5:15 tier=lexicon:0.3+words:1:0.25+knn:256:0.5:15 > runs/tier_full3_causal.log 2>&1
$E $C warm=32 stride=2 store=100000 \
  tier=knn:256:0.4:25 tier=lexicon:0.3+words:1:0.25+knn:256:0.4:25 > runs/tier_full3_flat.log 2>&1
