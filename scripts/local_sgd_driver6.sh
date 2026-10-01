#!/bin/bash
# Round 7 (docs/gpu_step_design.md): (1) the d=256 model with a shorter run / more dropout, averaged + alpha 0.1,
# lr 0.05; (2) K = 8 as two groups of 4, averaged within each group, ensembled across groups. Sequential.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
# run name K lr windows dropout alpha groups d heads d_ff blocks  (aesop, batch 32, wd 2e-4, warmup 6400, momentum 0.9, H = 100, shared init)
run() { echo "$(date +%T) start $1"; $F $1 $2 32 $3 $4 1 0.0002 $5 6400 0.9 $6 0 100 1 0 0.9 aesops_fables 0 $8 $9 ${10} ${11} $7 > runs/$1.log 2>&1; }
run big_k4_w256k_d0.3 4 0.05 256000 0.3 0.1 1 256 8 512 4
run big_k4_w512k_d0.3 4 0.05 512000 0.3 0.1 1 256 8 512 4
run big_k4_w512k_d0.5 4 0.05 512000 0.5 0.1 1 256 8 512 4
run groups_k8x2_a0 8 0.48 1024000 0.3 0 2 128 8 256 4
run groups_k8x2_a0.1 8 0.48 1024000 0.3 0.1 2 128 8 256 4
echo "$(date +%T) done"
