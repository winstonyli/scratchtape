#!/bin/sh
# Full-pass stack on the seq-128 novels22 model (scored on the same bytes as the seq-64 runs: warm = 96). No wrap (redundant at
# seq 128). a: static, old and tuned tiers; b/c: dynamic evaluation, tuned tiers, SGD 0.05 in 2- and 4-window steps (256/512 bytes).
cd "$(dirname "$0")/.."
E=runs/tier_eval_seq.exe; C=runs/nov22_s128_k1_d0.1_32m_f16_m0.ckpt
O="seq=128 warm=96 stride=1 corpus=novels22 memory=online store=50000 whiten=0.5"
BO="lexicon:0.3+words:1:0.25"; KO="knn:256:0.4:30:50:100"
B="lexicon:0.3+words:1:0.15"; K="knn:128:0.3:30:20:100"
$E $C $O tier=$BO+$KO tier=$B+$K > runs/tier_full_32m_novels22_s128_static.log 2>&1
$E $C $O dyn_every=2 dyn=0.05 tier=$B+$K > runs/tier_full_32m_novels22_s128_dyn2.log 2>&1
$E $C $O dyn_every=4 dyn=0.05 tier=$B+$K > runs/tier_full_32m_novels22_s128_dyn4.log 2>&1
echo "$(date +%T) done" >> runs/tier_full_32m_novels22_s128_dyn4.log
