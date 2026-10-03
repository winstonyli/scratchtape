#!/bin/sh
# Online (write-as-read) memory vs the offline causal and flat memories, warm=32, stride 1, on one sixth of the held-out text
# (part 3/6, mostly Dracula). docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
S="tier=knn:256:0.5:15 tier=knn:256:0.4:15 tier=knn:256:0.5:25"
$E $C warm=32 part=3/6 store=100000 memory=online $S > runs/tier_online_online.log 2>&1
$E $C warm=32 part=3/6 store=100000 memory=causal $S > runs/tier_online_causal.log 2>&1
$E $C warm=32 part=3/6 store=100000 $S > runs/tier_online_flat.log 2>&1
