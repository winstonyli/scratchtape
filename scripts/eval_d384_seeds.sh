#!/bin/bash
# Seed repeat of the d=384 16M model at full memory (whiten_rank 256), dyn old weights, parts 0 and 1; usage: eval_d384_seeds.sh <seed>...
# Seed 1 (runs/novels40_s128_16m_d384_m0.ckpt): dyn 0.9689 / 1.0085. Each seed builds its own ~10 GB memcache, removed after its evals.
cd "$(dirname "$0")/.."
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache d_model=384 heads=8 d_ff=768 whiten_rank=256"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
D="dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05"
for s in "$@"; do
  K=runs/novels40_s128_16m_d384_seed${s}_m0.ckpt
  for i in $(seq 1 60); do [ -f "$K" ] && [ ! -f "runs/novels40_s128_16m_d384_seed${s}.resume" ] && break; sleep 30; done
  [ -f "$K" ] || { echo "no checkpoint $K"; exit 1; }
  before=$(ls runs/memcache)
  for p in 0 1; do
    $E $K $B part=$p/4 $D tier=$P:128:0.5:30:15:100 > runs/seed${s}_d384_p${p}_dyn.log 2>&1
  done
  for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
done
echo "$(date +%T) done"
