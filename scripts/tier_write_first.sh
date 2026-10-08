#!/bin/sh
# Online memory with the chunk's keys written before its search (write=first) vs the old write-after, warm=32, stride 1,
# on a slice (default part=3/6, the one scripts/tier_online.sh used; causal there: -0.0407 at knn:256:0.5:15). docs/tiers_design.md.
# A tiny slice (part=5/96) is not a test: online starts each slice with an empty in-document memory, causal does not.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt; P=${1:-part=3/6}
S="tier=knn:256:0.5:15 tier=knn:256:0.4:15"
$E $C warm=32 $P store=100000 memory=online $S > runs/tier_wf_after.log 2>&1
$E $C warm=32 $P store=100000 memory=online write=first $S > runs/tier_wf_first.log 2>&1
