#!/bin/bash
# Stack evals for the d=384 model (the first attempt in train_novels40_d384.sh omitted the model-size args, so tier_eval rejected the checkpoint).
cd "$(dirname "$0")/.."
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache d_model=384 heads=8 d_ff=768"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
D="dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05"
K=runs/novels40_s128_16m_d384_m0.ckpt
before=$(ls runs/memcache)
for p in 0 1; do
  $E $K $B part=$p/4 tier=$P:128:0.5:30:15:100 tier=$P:128:0.8:30:22:100 > runs/n40_d384_p${p}_nodyn.log 2>&1
  $E $K $B part=$p/4 $D tier=$P:128:0.5:30:15:100 > runs/n40_d384_p${p}_dyn.log 2>&1
done
for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
echo "$(date +%T) done"
