#!/bin/bash
# Round 6 (docs/fusion_results.md): (A) sherlock at shorter runs, (B) a final lr decay on aesop,
# (C) a larger model (d 256, d_ff 512; lr 0.48 diverges, so lr 0.05 / 0.1). Sequential.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
# run name K lr windows alpha sync shared corpus decay_frac d heads d_ff blocks
run() { echo "$(date +%T) start $1"; $F $1 $2 32 $3 $4 1 0.0002 0.3 6400 0.9 $5 0 $6 $7 0 0.9 $8 $9 ${10} ${11} ${12} ${13} > runs/$1.log 2>&1; }
for w in 64000 128000; do
  run sh${w}_k1 1 0.48 $w 0 0 0 sherlock_holmes 0 128 8 256 4
  run sh${w}_k4_indep 4 0.48 $w 0 0 0 sherlock_holmes 0 128 8 256 4
  run sh${w}_k4_h100 4 0.48 $w 0 100 1 sherlock_holmes 0 128 8 256 4
  run sh${w}_k4_h100_a0.1 4 0.48 $w 0.1 100 1 sherlock_holmes 0 128 8 256 4
done
for f in 0.2 0.5; do
  run local_k4_h100_a0.1_decay$f 4 0.48 1024000 0.1 100 1 aesops_fables $f 128 8 256 4
done
run big_k1_lr0.05 1 0.05 1024000 0 0 0 aesops_fables 0 256 8 512 4
run big_k1_lr0.1 1 0.1 1024000 0 0 0 aesops_fables 0 256 8 512 4
run big_k4_h100_a0.1_lr0.05 4 0.05 1024000 0.1 100 1 aesops_fables 0 256 8 512 4
run big_k4_h100_a0.1_lr0.1 4 0.1 1024000 0.1 100 1 aesops_fables 0 256 8 512 4
echo "$(date +%T) done"
