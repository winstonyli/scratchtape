#!/bin/bash
# Key narrowing (`whiten_rank` now narrows the stored keys): half-memory checks on the 16M novels40 models, parts 0 and 1, no dyn old + tuned, dyn old.
#  A: d=384 with whiten_rank=256 against the unnarrowed half-memory numbers (dyn 0.9727 / 1.0126): the cost of narrowing.
#  B: d=256 with the default rank against the pre-change half-memory numbers (dyn 0.9901 / 1.0291): the rewritten whitener must be neutral.
cd "$(dirname "$0")/.."
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=500000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
D="dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05"
run() { # tag ckpt extra-args
  local before=$(ls runs/memcache)
  for p in 0 1; do
    $E $2 $B $3 part=$p/4 tier=$P:128:0.5:30:15:100 tier=$P:128:0.8:30:22:100 > runs/rank_${1}_p${p}_nodyn.log 2>&1
    $E $2 $B $3 part=$p/4 $D tier=$P:128:0.5:30:15:100 > runs/rank_${1}_p${p}_dyn.log 2>&1
  done
  for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
}
run d384r256 runs/novels40_s128_16m_d384_m0.ckpt "d_model=384 heads=8 d_ff=768 whiten_rank=256"
run d256 runs/novels40_s128_16m_m0.ckpt ""
echo "$(date +%T) done"
