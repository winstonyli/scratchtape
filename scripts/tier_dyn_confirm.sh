#!/bin/sh
# Confirm the re-tuned tiers on held-out half 1 (the tuning used half 0): old tuning, the round-2 winner (k 128), k 256, words 0.1.
E=runs/tier_eval_dyn6.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_tier
w() { echo "tier=$1+wrap:0.95"; }
$E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 part=1/2 dyn_every=4 dyn=0.05 \
  $(w "lexicon:0.3+words:1:0.25+knn:256:0.4:30:50:100") \
  $(w "lexicon:0.3+words:1:0.15+knn:128:0.3:30:20:100") $(w "lexicon:0.3+words:1:0.15+knn:256:0.3:30:20:100") \
  $(w "lexicon:0.3+words:1:0.1+knn:128:0.3:30:20:100") > runs/dyn_tier/confirm_p1.log 2>&1
