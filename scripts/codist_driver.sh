#!/bin/bash
# Codistillation experiment (docs/fusion_results.md, "Next" step 2). Arms run one after another.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
T=target/release/examples/training_recipe_check.exe
R="0.48 1024000"            # lr, windows; then seed, wd, dropout, warmup, momentum
for a in 0 0.5 1; do
  echo "$(date +%T) start codist_k4_a$a"
  $F codist_k4_a$a 4 32 $R 1 0.0002 0.3 6400 0.9 $a > runs/codist_k4_a$a.log 2>&1
done
echo "$(date +%T) start single model, 4x windows"
$T gpu_long4m_drop0.3_wd2e-4_s1 0 32 0.48 4096000 600 gpu 1 0.0002 all 0.3 6400 0.9 > runs/gpu_long4m_drop0.3_wd2e-4_s1.log 2>&1
echo "$(date +%T) done"
