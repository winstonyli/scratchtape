#!/bin/bash
# Round 4 (docs/gpu_step_design.md): seed repeats of averaging + alpha 0.1, a small DiLoCo outer-step
# sweep, alpha 0.05 / 0.2 with averaging, and K = 2 / 8 with averaging + alpha 0.1. Sequential.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
R="0.48 1024000"
# run name K seed alpha sync outer_lr outer_mu
run() { echo "$(date +%T) start $1"; $F $1 $2 32 $R $3 0.0002 0.3 6400 0.9 $4 0 $5 1 $6 $7 > runs/$1.log 2>&1; }
run local_k4_h100_a0.1_s11 4 11 0.1 100 0 0.9
run local_k4_h100_a0.1_s21 4 21 0.1 100 0 0.9
run diloco_k4_h100_lr1_mu0 4 1 0 100 1.0 0.0
run diloco_k4_h100_lr1_mu0.9 4 1 0 100 1.0 0.9
run diloco_k4_h100_lr0.5_mu0.5 4 1 0 100 0.5 0.5
run diloco_k4_h100_lr0.3_mu0.9 4 1 0 100 0.3 0.9
run diloco_k4_h100_lr1_mu0.5 4 1 0 100 1.0 0.5
run local_k4_h100_a0.05 4 1 0.05 100 0 0.9
run local_k4_h100_a0.2 4 1 0.2 100 0 0.9
run local_k2_h100_a0.1 2 1 0.1 100 0 0.9
run local_k8_h100_a0.1 8 1 0.1 100 0 0.9
echo "$(date +%T) done"
