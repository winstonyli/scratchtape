#!/bin/sh
# KNN_TILE (keys per tile / matmul launch) on the same short real eval as knn_knobs.sh; default 16384 gave tier prepare 30.5 s.
cd "$(dirname "$0")/.."
E=runs/tier_eval_tile.exe; C=runs/nov22_s128_k1_d0.1_32m_f16_m0.ckpt
A="seq=128 warm=96 stride=1 corpus=novels22 memory=online whiten=0.5 part=0/32 store=50000 dyn_every=4 dyn=0.05 tier=lexicon:0.3+words:1:0.15+knn:128:0.3:30:20:100"
for t in 65536 262144 4096; do
  KNN_TILE=$t $E $C $A > runs/dyn_tier/tile_$t.log 2>&1
  echo "tile $t $(grep -h '^timing' runs/dyn_tier/tile_$t.log | cut -c1-90) $(grep -o 'CE [0-9.]*' runs/dyn_tier/tile_$t.log | tail -1)" >> runs/dyn_tier/tile.out
done
echo done >> runs/dyn_tier/tile.out
