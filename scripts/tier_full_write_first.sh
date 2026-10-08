#!/bin/sh
# Full held-out run with the online memory written before each chunk's search (write=first). Same specs as tier_full4/tier_gate_dump
# (model alone 1.1764; write-after: memory alone 1.1352, stack at knn weight 0.5 1.1313, at 0.4 1.1301). docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
$E $C warm=32 stride=1 memory=online store=100000 \
  tier=knn:256:0.5:15 tier=lexicon:0.3+words:1:0.25+knn:256:0.5:15 tier=lexicon:0.3+words:1:0.25+knn:256:0.4:15 > runs/tier_full_wf.log 2>&1
