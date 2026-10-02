#!/bin/bash
# Codistillation follow-up (docs/fusion_results.md): train-probe for the alpha=0 baseline,
# a weaker constant alpha, and alpha ramped up from 0. Arms run one after another.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
R="0.48 1024000"            # lr, windows; then seed, wd, dropout, warmup, momentum, alpha, ramp windows
echo "$(date +%T) start codist2_k4_a0"
$F codist2_k4_a0 4 32 $R 1 0.0002 0.3 6400 0.9 0 > runs/codist2_k4_a0.log 2>&1
echo "$(date +%T) start codist2_k4_a0.1"
$F codist2_k4_a0.1 4 32 $R 1 0.0002 0.3 6400 0.9 0.1 > runs/codist2_k4_a0.1.log 2>&1
echo "$(date +%T) start codist2_k4_a0.5_ramp"
$F codist2_k4_a0.5_ramp 4 32 $R 1 0.0002 0.3 6400 0.9 0.5 512000 > runs/codist2_k4_a0.5_ramp.log 2>&1
echo "$(date +%T) done"
