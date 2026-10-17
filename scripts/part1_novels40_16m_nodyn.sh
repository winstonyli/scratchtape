#!/bin/bash
# Rerun of the no-dyn half of part1_novels40_16m.sh (the first attempt hit a memory-allocation failure).
cd "$(dirname "$0")/.."
C="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache part=1/4"
P="lexicon:0.2+words:1:0.1+knn"
before=$(ls runs/memcache)
target/release/examples/tier_eval.exe runs/novels40_s128_16m_m0.ckpt $C tier=$P:128:0.5:30:15:100 tier=$P:128:0.8:30:22:100 > runs/n40_16m_p1_nodyn.log 2>&1
for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
echo "$(date +%T) done"
