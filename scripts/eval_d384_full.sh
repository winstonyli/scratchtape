#!/bin/bash
# Full 125M-key memory for the d=384 16M model with narrowed keys (whiten_rank 256 = the d=256 key footprint, then 320), parts 0 and 1:
# no dyn old + tuned weights, dyn old weights. Compare with the half-memory d=384 (dyn 0.9727 / 1.0126, rank 256 0.9758 / 1.0175) and the best d=256 (64M windows, 0.9765 / 1.0131).
cd "$(dirname "$0")/.."
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache d_model=384 heads=8 d_ff=768"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
D="dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05"
K=runs/novels40_s128_16m_d384_m0.ckpt
for r in 256 320; do
  before=$(ls runs/memcache)
  for p in 0 1; do
    $E $K $B whiten_rank=$r part=$p/4 tier=$P:128:0.5:30:15:100 tier=$P:128:0.8:30:22:100 > runs/full_r${r}_p${p}_nodyn.log 2>&1
    $E $K $B whiten_rank=$r part=$p/4 $D tier=$P:128:0.5:30:15:100 > runs/full_r${r}_p${p}_dyn.log 2>&1
  done
  for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
done
echo "$(date +%T) done"
