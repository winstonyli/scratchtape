#!/bin/sh
# Tier re-tune, round 3, on the adapted model (4-window SGD 0.05), held-out half 0 (part=0/2). Base: words 0.15, knn:128:0.3:30:20:100,
# wrap 0.95 (half 0: 1.0130). Run A: wrap lambda, query temperature and shift in one pass. Runs B: whiten and tap (they change the
# memory keys, so one run each; each also scores the base). `store` is not varied: 100000 windows already exceeds the 70855 train windows.
E=runs/tier_eval_dyn6.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_tier
L="lexicon:0.3+words:1:0.15"
base="tier=$L+knn:128:0.3:30:20:100+wrap:0.95"
common="warm=32 stride=1 memory=online store=100000 part=0/2 dyn_every=4 dyn=0.05"
$E $C $common whiten=0.5 $base \
  "tier=$L+knn:128:0.3:30:20:100+wrap:0.9" "tier=$L+knn:128:0.3:30:20:100+wrap:0.98" \
  "tier=$L+knn:128:0.3:25:20:100+wrap:0.95" "tier=$L+knn:128:0.3:35:20:100+wrap:0.95" \
  "tier=$L+knn:128:0.3:30:20:75+wrap:0.95" "tier=$L+knn:128:0.3:30:20:125+wrap:0.95" \
  "tier=$L+knn:128:0.35:30:20:100+wrap:0.95" "tier=$L+knn:96:0.3:30:20:100+wrap:0.95" > runs/dyn_tier/retune3_A_p0.log 2>&1
for w in 0.25 0.75; do $E $C $common whiten=$w $base > runs/dyn_tier/retune3_whiten${w}_p0.log 2>&1; done
$E $C $common whiten=0.5 tap=2 $base > runs/dyn_tier/retune3_tap2_p0.log 2>&1
