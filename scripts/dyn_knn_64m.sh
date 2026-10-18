#!/bin/bash
# kNN weights under dyn for the 64M novels40 model: coordinate grid around the old 128:0.5:30:15:100 on part 1, same grid on part 0 to validate (one memory build).
cd "$(dirname "$0")/.."
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
D="dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05"
K=runs/novels40_s128_64m_m0.ckpt
G=""
for w in 128:0.5:30:15:100 128:0.4:30:15:100 128:0.6:30:15:100 128:0.5:30:10:100 128:0.5:30:20:100 128:0.5:24:15:100 128:0.5:40:15:100 128:0.8:30:22:100 96:0.5:30:15:100; do G="$G tier=$P:$w"; done
before=$(ls runs/memcache)
for p in 1 0; do $E $K $B part=$p/4 $D $G > runs/n40_64m_dynknn_p${p}.log 2>&1; done
for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
echo "$(date +%T) done"
