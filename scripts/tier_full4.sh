#!/bin/sh
# docs/tiers_design.md: full held-out run of the deployable stack: online (write-as-read) memory + lexicon + word bigram,
# fair-context protocol (warm=32), stride 1 (online needs it). Settings from the tier_full3 tuning.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
$E $C warm=32 stride=1 memory=online store=100000 \
  tier=lexicon:0.3 tier=words:1:0.25 tier=lexicon:0.3+words:1:0.25 tier=knn:256:0.5:15 \
  tier=words:1:0.25+knn:256:0.5:15 tier=lexicon:0.3+words:1:0.25+knn:256:0.5:15 > runs/tier_full4_online.log 2>&1
