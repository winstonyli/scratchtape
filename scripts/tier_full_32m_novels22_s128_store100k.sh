#!/bin/sh
# Seq-128 novels22 model, tuned tiers, dynamic evaluation (4 windows, SGD 0.05), memory of 100000 train windows (12.8M keys).
cd "$(dirname "$0")/.."
E=runs/tier_eval_seq.exe; C=runs/nov22_s128_k1_d0.1_32m_f16_m0.ckpt
$E $C seq=128 warm=96 stride=1 corpus=novels22 memory=online store=100000 whiten=0.5 dyn_every=4 dyn=0.05 tier=lexicon:0.3+words:1:0.15+knn:128:0.3:30:20:100 > runs/tier_full_32m_novels22_s128_store100k.log 2>&1
echo "$(date +%T) done" >> runs/tier_full_32m_novels22_s128_store100k.log
