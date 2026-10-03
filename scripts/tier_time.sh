#!/bin/sh
# Phase timing of tier_eval on a small slice (part 5/24, warm=32, stride 1): online vs flat memory. docs/tiers_design.md.
E=runs/tier_eval_time.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
S="tier=knn:256:0.5:15"
$E $C warm=32 part=5/24 store=100000 memory=online $S > runs/tier_time_online.log 2>&1
$E $C warm=32 part=5/24 store=100000 $S > runs/tier_time_flat.log 2>&1
