#!/bin/bash
# novels90 at 16M windows (the 40-vs-90 question at full training), then its fresh-split stack on part 0, no dyn and dyn (one memory build).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
R="windows=16384000 dropout=0.1 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 d_model=256 d_ff=512 seq=128 score_tail=32 corpus=novels90"
TRAIN_F16=1 fmc novels90_s128_16m 1 16 0.05 $R
C="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache part=0/4"
P="lexicon:0.2+words:1:0.1+knn"
E=target/release/examples/tier_eval.exe
before=$(ls runs/memcache)
$E runs/novels90_s128_16m_m0.ckpt $C tier=$P:128:0.5:30:15:100 tier=$P:128:0.8:30:22:100 > runs/n90_16m_nodyn.log 2>&1
$E runs/novels90_s128_16m_m0.ckpt $C dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05 tier=$P:128:0.5:30:15:100 > runs/n90_16m_dyn.log 2>&1
for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
echo "$(date +%T) done"
