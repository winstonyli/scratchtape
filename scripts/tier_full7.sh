#!/bin/sh
# NOTE: written for the write-after, merged-search online memory. Today memory=online writes first and searches train and
# in-document keys separately, so this reproduces its logged numbers only at commit 92266bc (before 1cce0ef).
# KNN_TIMING=1 (per-block stage times in the log). Full held-out run of the deployable stack (online memory + lexicon + word bigram), f16 search (default), top-k live-count shift, incremental sample upload. Fair-context protocol (warm=32), stride 1 (online needs it). Settings from the tier_full3 tuning.
export KNN_TIMING=1
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
$E $C warm=32 stride=1 memory=online store=100000 \
  tier=lexicon:0.3 tier=words:1:0.25 tier=lexicon:0.3+words:1:0.25 tier=knn:256:0.5:15 \
  tier=words:1:0.25+knn:256:0.5:15 tier=lexicon:0.3+words:1:0.25+knn:256:0.5:15 > runs/tier_full7.log 2>&1
