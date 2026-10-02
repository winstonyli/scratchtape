#!/bin/sh
# Experiment B on the FULL held-out split (docs/tiers_design.md): 4.5M-key memory, settings tuned on offset 0 of stride 16.
runs/tier_eval_run.exe runs/nov_big_k1_d0.1_8m_m0.ckpt expect=1.2408 store=100000 \
  tier=lexicon:0.3 tier=knn:32:0.4:25 tier=knn:64:0.4:25 tier=knn:128:0.4:25 tier=knn:128:0.3:25 \
  tier=lexicon:0.3+knn:128:0.4:25 tier=lexicon:0.3+knn:128:0.3:25 > runs/tier_full.log 2>&1
