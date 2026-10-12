#!/bin/sh
# A/B: the same short eval (half 0 / 4, store 50000, dynamic evaluation) at Normal vs BelowNormal priority, alternating.
cd "$(dirname "$0")/.."
E=runs/tier_eval_rank.exe; C=runs/nov22_s128_k1_d0.1_32m_f16_m0.ckpt
A="seq=128 warm=96 stride=1 corpus=novels22 memory=online whiten=0.5 part=0/4 store=50000 dyn_every=4 dyn=0.05 tier=lexicon:0.3+words:1:0.15+knn:128:0.3:30:20:100"
for r in 1 2; do
  for p in normal belownormal; do
    s=$(date +%s)
    if [ $p = normal ]; then $E $C $A > runs/dyn_tier/prio_${p}_$r.log 2>&1; else cmd //c start "" //b //belownormal //wait $E $C $A > runs/dyn_tier/prio_${p}_$r.log 2>&1; fi
    echo "$p $r $(( $(date +%s) - s ))s" >> runs/dyn_tier/prio_ab.out
  done
done
echo done >> runs/dyn_tier/prio_ab.out
