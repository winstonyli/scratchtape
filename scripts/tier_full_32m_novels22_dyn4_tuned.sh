#!/bin/sh
# Full-pass stack with the tiers re-tuned for the adapted model (tier_dyn_retune*.sh on half 0, tier_dyn_confirm.sh on half 1):
# words 0.15, knn k 128, lambda 0.3, temp 30, in-document temp 20, shift 100; dynamic evaluation in 4-window SGD steps at 0.05.
E=runs/tier_eval_dyn6.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
B="lexicon:0.3+words:1:0.15"; K="knn:128:0.3:30:20:100"
$E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 dyn_every=4 dyn=0.05 tier=$B+$K tier=$B+$K+wrap:0.95 > runs/tier_full_32m_novels22_dyn4_tuned.log 2>&1
