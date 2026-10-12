#!/bin/sh
# Seq-128 model, memory of 100000 windows, held-out half 0: SGD lr around 0.05 at 4-window steps (base 0.9902 at 0.05).
cd "$(dirname "$0")/.."
E=runs/tier_eval_seq.exe; C=runs/nov22_s128_k1_d0.1_32m_f16_m0.ckpt
for lr in 0.03 0.07; do
  $E $C seq=128 warm=96 stride=1 corpus=novels22 memory=online whiten=0.5 part=0/2 store=100000 dyn_every=4 dyn=$lr tier=lexicon:0.3+words:1:0.15+knn:128:0.3:30:20:100 > runs/dyn_tier/s128_store100k_lr${lr}_p0.log 2>&1
done
echo "$(date +%T) done" > runs/dyn_tier/s128_lr_done
