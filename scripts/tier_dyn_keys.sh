#!/bin/sh
# Stack on held-out half 0 with dynamic evaluation (4-window SGD 0.05): hidden states for the memory from the adapted model
# (as tier_full_32m_novels22_dyn4.sh; half-0 value 1.0235) against from the frozen checkpoint (dyn_keys=frozen).
E=runs/tier_eval_dyn6.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
B="lexicon:0.3+words:1:0.25"; K="knn:256:0.4:30:50:100"
mkdir -p runs/dyn_tier
$E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 part=0/2 dyn_every=4 dyn=0.05 dyn_keys=frozen tier=$B+$K tier=$B+$K+wrap:0.95 > runs/dyn_tier/c4_dyn0.05_frozenkeys.log 2>&1
