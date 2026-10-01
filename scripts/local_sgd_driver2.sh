#!/bin/bash
# Follow-ups to local_sgd_driver.sh (docs/gpu_step_design.md): seed repeats of the H=100 averaged arm and
# of the single batch-128 baseline, DiLoCo's outer optimizer, and averaging plus alpha 0.1. Sequential.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
T=target/release/examples/training_recipe_check.exe
R="0.48 1024000"
run() { n=$1; shift; echo "$(date +%T) start $n"; $F $n 4 32 $R "$@" > runs/$n.log 2>&1; }
# args after $R: seed wd dropout warmup momentum alpha ramp sync shared outer_lr outer_mu
run local_k4_h100_s11 11 0.0002 0.3 6400 0.9 0 0 100 1
run local_k4_h100_s21 21 0.0002 0.3 6400 0.9 0 0 100 1
for s in 2 3; do
  echo "$(date +%T) start b128 s$s"
  $T gpu_b128_drop0.3_wd2e-4_s$s 0 128 0.48 4096000 600 gpu $s 0.0002 all 0.3 25600 0.9 > runs/gpu_b128_drop0.3_wd2e-4_s$s.log 2>&1
done
run diloco_k4_h100 1 0.0002 0.3 6400 0.9 0 0 100 1 0.7 0.9
run diloco_k4_h1000 1 0.0002 0.3 6400 0.9 0 0 1000 1 0.7 0.9
run local_k4_h100_a0.1 1 0.0002 0.3 6400 0.9 0.1 0 100 1
echo "$(date +%T) done"
