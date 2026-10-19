#!/bin/bash
# Width test, next step: novels40 at d=512 (8 heads, d_ff 1024, dropout 0.15, 4 blocks) for 16M windows, same recipe otherwise; full-memory stack at whiten_rank=256
# (dyn old weights, parts 0 and 1). Bar: beat the d=384 three-seed mean (dyn 0.9694 / 1.0087) by > 0.003 on both parts.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
M="d_model=512 heads=8 d_ff=1024"
R="windows=16384000 dropout=0.15 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 $M seq=128 score_tail=32 corpus=novels40"
TRAIN_F16=1 fmc novels40_s128_16m_d512 1 16 0.05 $R
B="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache $M whiten_rank=256"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
D="dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05"
K=runs/novels40_s128_16m_d512_m0.ckpt
[ -f "$K" ] || { echo "no checkpoint $K"; exit 1; }
before=$(ls runs/memcache)
for p in 0 1; do
  $E $K $B part=$p/4 $D tier=$P:128:0.5:30:15:100 > runs/n40_d512_p${p}_dyn.log 2>&1
done
for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
echo "$(date +%T) done"
