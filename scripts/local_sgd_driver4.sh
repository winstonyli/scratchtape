#!/bin/bash
# Round 5 (docs/gpu_step_design.md): does averaging + alpha 0.1 carry to another corpus (sherlock_holmes,
# 256000 windows: ~same passes over its 54 KB as 1M over aesop's 214 KB), and shorter sync periods on aesop.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
# run name K windows alpha sync shared corpus
run() { echo "$(date +%T) start $1"; $F $1 $2 32 0.48 $3 1 0.0002 0.3 6400 0.9 $4 0 $5 $6 0 0.9 $7 > runs/$1.log 2>&1; }
run sh_k1 1 256000 0 0 0 sherlock_holmes
run sh_k4_indep 4 256000 0 0 0 sherlock_holmes
run sh_k4_h100 4 256000 0 100 1 sherlock_holmes
run sh_k4_h100_a0.1 4 256000 0.1 100 1 sherlock_holmes
run local_k4_h10_a0.1 4 1024000 0.1 10 1 aesops_fables
run local_k4_h30_a0.1 4 1024000 0.1 30 1 aesops_fables
echo "$(date +%T) done"
