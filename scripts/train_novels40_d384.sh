#!/bin/bash
# NOTE: the eval half omitted d_model/heads/d_ff and failed; evals rerun by scripts/eval_d384.sh.
# Width test: novels40 at d=384 (8 heads, d_ff 768, dropout 0.15, 4 blocks) for 16M windows, same recipe otherwise; stack on fresh-split parts 0 and 1
# (no dyn old + tuned weights, dyn old weights), one memory build. Bar: dyn stack must beat the 16M two-seed mean (0.9834 / 1.0217) by > 0.004.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
R="windows=16384000 dropout=0.15 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 d_model=384 heads=8 d_ff=768 seq=128 score_tail=32 corpus=novels40"
TRAIN_F16=1 fmc novels40_s128_16m_d384 1 16 0.05 $R
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache"
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
