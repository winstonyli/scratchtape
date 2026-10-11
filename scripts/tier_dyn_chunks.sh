#!/bin/sh
# Stack (default novels6 tiers, as tier_full_32m_novels22.sh) on the first held-out half (part=0/2) with the scoring/adapting
# chunk size set by dyn_every: A/B check that the tiers do not depend on the chunk size without dynamic evaluation, then dynamic
# evaluation at 8 and 4 windows per step (model-alone optimum lrs 0.1 and 0.05) against 32 windows at 0.2. One at a time.
E=runs/tier_eval_dyn4.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
B="lexicon:0.3+words:1:0.25"; K="knn:256:0.4:30:50:100"
mkdir -p runs/dyn_tier
run() { name=$1; shift; $E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 part=0/2 "$@" tier=$B+$K tier=$B+$K+wrap:0.95 > runs/dyn_tier/$name.log 2>&1; }
run c32 dyn_every=32
run c8 dyn_every=8
run c32_dyn0.2 dyn_every=32 dyn=0.2
run c8_dyn0.1 dyn_every=8 dyn=0.1
run c4_dyn0.05 dyn_every=4 dyn=0.05
