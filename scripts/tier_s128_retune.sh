#!/bin/sh
# Seq-128 novels22 model, held-out half 0: tiers one at a time around the seq-64 tuning (base 1.0392 on the full pass),
# then the adaptation step (8 windows, two lrs), then the memory size (store 35000 windows = 4.5M keys, as at seq 64).
cd "$(dirname "$0")/.."
E=runs/tier_eval_seq.exe; C=runs/nov22_s128_k1_d0.1_32m_f16_m0.ckpt
O="seq=128 warm=96 stride=1 corpus=novels22 memory=online whiten=0.5 part=0/2"
L="lexicon:0.3+words:1:0.15"; K="knn:128:0.3:30:20:100"
mkdir -p runs/dyn_tier
S="tier=$L+$K"
for v in "lexicon:0.2+words:1:0.15+$K" "lexicon:0.45+words:1:0.15+$K" "$L+knn:128:0.3:30:20:100" "lexicon:0.3+words:1:0.1+$K" "lexicon:0.3+words:1:0.2+$K" \
  "$L+knn:128:0.25:30:20:100" "$L+knn:128:0.35:30:20:100" "$L+knn:96:0.3:30:20:100" "$L+knn:192:0.3:30:20:100" \
  "$L+knn:128:0.3:25:20:100" "$L+knn:128:0.3:35:20:100" "$L+knn:128:0.3:30:15:100" "$L+knn:128:0.3:30:30:100" \
  "$L+knn:128:0.3:30:20:75" "$L+knn:128:0.3:30:20:150"; do S="$S tier=$v"; done
$E $C $O store=50000 dyn_every=4 dyn=0.05 $S > runs/dyn_tier/s128_tiers_p0.log 2>&1
for lr in 0.05 0.1; do
  $E $C $O store=50000 dyn_every=8 dyn=$lr tier=$L+$K > runs/dyn_tier/s128_e8_lr${lr}_p0.log 2>&1
done
$E $C $O store=35000 dyn_every=4 dyn=0.05 tier=$L+$K > runs/dyn_tier/s128_store35k_p0.log 2>&1
echo "$(date +%T) done" > runs/dyn_tier/s128_done
