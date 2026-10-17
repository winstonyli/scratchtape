#!/bin/bash
# (1) seed repeat of novels40 16M (seed=2; the original used the default seed 1), stack on fresh-split parts 0 and 1;
# (2) kNN weight retune for the novels40 32M model, no dyn: coordinate grid around 128:0.8:30:22:100 on part 1, same grid on part 0 to validate.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
R="windows=16384000 dropout=0.1 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 d_model=256 d_ff=512 seq=128 score_tail=32 corpus=novels40 seed=2"
TRAIN_F16=1 fmc novels40_s128_16m_seed2 1 16 0.05 $R
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
D="dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05"
clean() { for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done; }
before=$(ls runs/memcache)
K=runs/novels40_s128_16m_seed2_m0.ckpt
for p in 0 1; do
  $E $K $B part=$p/4 tier=$P:128:0.5:30:15:100 tier=$P:128:0.8:30:22:100 > runs/n40_16m_s2_p${p}_nodyn.log 2>&1
  $E $K $B part=$p/4 $D tier=$P:128:0.5:30:15:100 > runs/n40_16m_s2_p${p}_dyn.log 2>&1
done
clean
K=runs/novels40_s128_32m_m0.ckpt
G=""
for w in 128:0.8:30:22:100 128:0.9:30:22:100 128:0.7:30:22:100 128:0.8:30:30:100 128:0.8:30:16:100 128:0.8:24:22:100 128:0.8:40:22:100 96:0.8:30:22:100 64:0.8:30:22:100; do G="$G tier=$P:$w"; done
for p in 1 0; do
  $E $K $B part=$p/4 $G > runs/n40_32m_retune_p${p}.log 2>&1
done
clean
echo "$(date +%T) done"
