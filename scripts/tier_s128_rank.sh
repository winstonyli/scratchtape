#!/bin/sh
# Key compression feasibility: whiten=0.5 keeps only the top-r eigen-directions (keys stay 256 wide, so no VRAM saving yet).
# Seq-128 model, store 50000 windows, half 0, dynamic evaluation 4 x 0.05; full rank is 0.9941. Only rank 128 ran (0.9990); stopped there.
cd "$(dirname "$0")/.."
E=runs/tier_eval_rank.exe; C=runs/nov22_s128_k1_d0.1_32m_f16_m0.ckpt
for r in 128 64 32; do
  $E $C seq=128 warm=96 stride=1 corpus=novels22 memory=online whiten=0.5 whiten_rank=$r part=0/2 store=50000 dyn_every=4 dyn=0.05 tier=lexicon:0.3+words:1:0.15+knn:128:0.3:30:20:100 > runs/dyn_tier/s128_rank${r}_p0.log 2>&1
done
echo "$(date +%T) done" > runs/dyn_tier/s128_rank_done
