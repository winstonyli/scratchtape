#!/bin/sh
# The best stack (tiers from the default novels6 train split, as the 1.0821 run) on the 32M novels22 model with dynamic
# evaluation at the lr the model-alone sweep chose (0.2). Train memory keys from the frozen model, queries adapted.
E=runs/tier_eval_dyn2.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
B="lexicon:0.3+words:1:0.25"; K="knn:256:0.4:30:50:100"
$E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 dyn=${DYN:-0.2} tier=$B+$K tier=$B+$K+wrap:0.95 > runs/tier_full_32m_novels22_dyn${DYN:-0.2}.log 2>&1
