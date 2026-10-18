#!/bin/bash
# Matched half-memory comparison: the d=384 memory (125M keys) does not fit on the 16 GB card, so d=384 and the d=256 16M seed-1 baseline
# are both scored with store=500000 train windows (~64M keys). Parts 0 and 1: no dyn old + tuned weights, dyn old weights. Bar: d=384 dyn beats d=256 by > 0.004.
cd "$(dirname "$0")/.."
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=500000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
D="dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05"
run() { # tag ckpt extra-args
  local before=$(ls runs/memcache)
  for p in 0 1; do
    $E $2 $B $3 part=$p/4 tier=$P:128:0.5:30:15:100 tier=$P:128:0.8:30:22:100 > runs/half_${1}_p${p}_nodyn.log 2>&1
    $E $2 $B $3 part=$p/4 $D tier=$P:128:0.5:30:15:100 > runs/half_${1}_p${p}_dyn.log 2>&1
  done
  for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
}
run d384 runs/novels40_s128_16m_d384_m0.ckpt "d_model=384 heads=8 d_ff=768"
run d256 runs/novels40_s128_16m_m0.ckpt ""
echo "$(date +%T) done"
