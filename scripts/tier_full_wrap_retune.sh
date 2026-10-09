#!/bin/sh
# Re-tune the memory with the wrap tier in place (wrap:0.95 after knn). Control knn:256:0.4:30:50:100 -> 1.0956
# (runs/tier_full_wrap.log). One coordinate at a time: lambda, temperature, in-document temperature, shift.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
B="lexicon:0.3+words:1:0.25"; S=""
for K in 0.3:30:50:100 0.5:30:50:100 0.4:25:50:100 0.4:40:50:100 0.4:30:35:100 0.4:30:70:100 0.4:30:50:50 0.4:30:50:150; do
  S="$S tier=$B+knn:256:$K+wrap:0.95"
done
$E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 $S > runs/tier_full_wrap_retune.log 2>&1
