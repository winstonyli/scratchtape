#!/bin/sh
# kNN search knobs on the real eval (seq 128, half 0 / 32 = 123 chunks of 128 queries, store 50000, dynamic evaluation):
# the `timing:` line's tier-prepare seconds for KNN_PARTS / KNN_SLICE settings (default parts = 32, slice 16384).
cd "$(dirname "$0")/.."
E=runs/tier_eval_rank.exe; C=runs/nov22_s128_k1_d0.1_32m_f16_m0.ckpt
A="seq=128 warm=96 stride=1 corpus=novels22 memory=online whiten=0.5 part=0/32 store=50000 dyn_every=4 dyn=0.05 tier=lexicon:0.3+words:1:0.15+knn:128:0.3:30:20:100"
run() { n=$1; shift; env "$@" $E $C $A > runs/dyn_tier/knob_$n.log 2>&1; echo "$n $(grep -h '^timing' runs/dyn_tier/knob_$n.log | cut -c1-90) $(grep -o 'CE [0-9.]*' runs/dyn_tier/knob_$n.log | tail -1)" >> runs/dyn_tier/knobs.out; }
run base A=1
run parts64 KNN_PARTS=64
run parts128 KNN_PARTS=128
run slice32k KNN_SLICE=32768
run slice8k KNN_SLICE=8192
run parts16 KNN_PARTS=16
echo done >> runs/dyn_tier/knobs.out
