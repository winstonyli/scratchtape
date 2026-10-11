#!/bin/sh
# Round 2 of the tier re-tune on the adapted model, half 0. Round 1 (tier_dyn_retune.sh, base 1.0235): in-document temperature
# 30 (from 50) 1.0170; knn lambda 0.3 1.0222; words 0.15 1.0218; k 128 1.0219. Start from those and move the in-document
# temperature and shift further. Every spec ends in +wrap:0.95.
E=runs/tier_eval_dyn6.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_tier
w() { echo "tier=$1+wrap:0.95"; }
L="lexicon:0.3+words:1:0.15"
$E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 part=0/2 dyn_every=4 dyn=0.05 \
  $(w "$L+knn:256:0.3:30:30:100") $(w "$L+knn:256:0.3:30:20:100") $(w "$L+knn:256:0.3:30:15:100") $(w "$L+knn:256:0.3:30:10:100") \
  $(w "$L+knn:256:0.3:30:20:50") $(w "$L+knn:256:0.3:30:20:150") $(w "$L+knn:256:0.3:30:20:200") \
  $(w "$L+knn:128:0.3:30:20:100") $(w "lexicon:0.3+words:1:0.1+knn:256:0.3:30:20:100") $(w "$L+knn:256:0.25:30:20:100") \
  > runs/dyn_tier/retune2_p0.log 2>&1
