#!/bin/sh
# Full-pass stack (tiers from the default novels6 train split) on the 32M novels22 model with dynamic evaluation in 4-window
# steps: plain SGD at lr 0.05, then RMS-normalised steps (beta 0.99, lr 1e-4; host-side prototype, ~10 min extra). One at a time.
E=runs/tier_eval_dyn5.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
B="lexicon:0.3+words:1:0.25"; K="knn:256:0.4:30:50:100"
run() { name=$1; shift; $E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 dyn_every=4 "$@" tier=$B+$K tier=$B+$K+wrap:0.95 > runs/tier_full_32m_novels22_$name.log 2>&1; }
run dyn4_sgd0.05 dyn=0.05
run dyn4_rms1e-4 dyn=0.0001 dyn_rms=0.99
