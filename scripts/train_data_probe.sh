#!/bin/bash
# Data probe (docs/tiers_design.md "Model-training data probe"): the nov22 seq-128 4M-window plain recipe on novels40 and
# novels90 (neither holds the fresh books), then the fresh-split stack for each; memory caches are deleted after use (9.5 GB each).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
R="windows=4096000 dropout=0.1 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 d_model=256 d_ff=512 seq=128 score_tail=32"
for c in novels40 novels90; do TRAIN_F16=1 fmc ${c}_s128_4m 1 16 0.05 $R corpus=$c; done
C="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 store=1000000 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache part=0/4"
T="tier=lexicon:0.2+words:1:0.1+knn:128:0.5:30:15:100"
for c in novels40 novels90; do
  [ -f runs/probe_$c.log ] && grep -q "^tier" runs/probe_$c.log && continue
  before=$(ls runs/memcache)
  target/release/examples/tier_eval.exe runs/${c}_s128_4m_m0.ckpt $C $T > runs/probe_$c.log 2>&1
  for f in $(ls runs/memcache); do echo "$before" | grep -qx "$f" || rm -f "runs/memcache/$f"; done
done
echo "$(date +%T) done"
