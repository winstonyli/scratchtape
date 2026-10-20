#!/bin/bash
# One dyn-stack eval of a width-test checkpoint (full memory, whiten_rank=256, old kNN weights, dyn). Usage: eval_width_part.sh <name> <d> <heads> <dff> <blocks> <part>
# Reruns a part whose eval in train_eval_width.sh failed. Deletes the memcache entries it builds.
cd "$(dirname "$0")/.."
M="d_model=$2 heads=$3 d_ff=$4 blocks=$5"
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache $M whiten_rank=256"
P="lexicon:0.2+words:1:0.1+knn"
before=$(ls runs/memcache)
target/release/examples/tier_eval.exe runs/${1}_m0.ckpt $B part=$6/4 dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05 tier=$P:128:0.5:30:15:100 > runs/${1}_p$6_dyn.log 2>&1
for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
echo "$(date +%T) done"
