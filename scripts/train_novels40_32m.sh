#!/bin/bash
# (1) novels40 at 32M windows (twice the 16M budget), stack on fresh-split parts 0 and 1 (one memory build);
# (3) the 64M-window novels22 model on part 1 (no dyn old+tuned, dyn old) for the cross-model comparison.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
R="windows=32768000 dropout=0.1 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 d_model=256 d_ff=512 seq=128 score_tail=32 corpus=novels40"
TRAIN_F16=1 fmc novels40_s128_32m 1 16 0.05 $R
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
D="dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05"
clean() { for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done; }
before=$(ls runs/memcache)
K=runs/novels40_s128_32m_m0.ckpt
for p in 0 1; do
  $E $K $B part=$p/4 tier=$P:128:0.5:30:15:100 tier=$P:128:0.8:30:22:100 > runs/n40_32m_p${p}_nodyn.log 2>&1
  $E $K $B part=$p/4 $D tier=$P:128:0.5:30:15:100 > runs/n40_32m_p${p}_dyn.log 2>&1
done
clean
K=runs/nov22_s128_k1_d0.1_64m_f16_m0.ckpt
$E $K $B part=1/4 tier=$P:128:0.5:30:15:100 tier=$P:128:0.8:30:22:100 > runs/n22_64m_p1_nodyn.log 2>&1
$E $K $B part=1/4 $D tier=$P:128:0.5:30:15:100 > runs/n22_64m_p1_dyn.log 2>&1
clean
echo "$(date +%T) done"
