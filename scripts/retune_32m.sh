#!/bin/bash
# The retune half of seed_and_retune.sh, rerun (the first grid had k=256, but tier_eval caps k at 128). Grid is in the script below.
cd "$(dirname "$0")/.."
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
K=runs/novels40_s128_32m_m0.ckpt
G=""
for w in 128:0.8:30:22:100 128:0.9:30:22:100 128:0.7:30:22:100 128:0.8:30:30:100 128:0.8:30:16:100 128:0.8:24:22:100 128:0.8:40:22:100 96:0.8:30:22:100 64:0.8:30:22:100; do G="$G tier=$P:$w"; done
before=$(ls runs/memcache)
for p in 1 0; do
  $E $K $B part=$p/4 $G > runs/n40_32m_retune_p${p}.log 2>&1
done
for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
echo "$(date +%T) done"
