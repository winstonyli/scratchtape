#!/bin/bash
# Round 8 (docs/gpu_step_design.md): the recipe on all four bundled books (373 KB train; held-out is the
# last 10% of each book). Resumable: relaunch the script and each run continues from runs/<name>.resume
# (a finished run's .ckpt is not rechecked, so delete it to rerun). Sequential.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
# run name K lr windows alpha sync shared d heads d_ff blocks  (batch 32, wd 2e-4, dropout 0.3, warmup 6400, momentum 0.9)
run() {
  [ -f runs/$1_m0.ckpt ] && return
  echo "$(date +%T) start $1"
  $F $1 $2 32 $3 $4 1 0.0002 0.3 6400 0.9 $5 0 $6 $7 0 0.9 all_four 0 $8 $9 ${10} ${11} 1 600 >> runs/$1.log 2>&1
}
run all4_k1 1 0.48 1024000 0 0 0 128 8 256 4
run all4_k4_h100_a0.1 4 0.48 1024000 0.1 100 1 128 8 256 4
run all4_k4_h100_a0.1_2m 4 0.48 2048000 0.1 100 1 128 8 256 4
run all4_big_k1 1 0.05 1024000 0 0 0 256 8 512 4
run all4_big_k4_h100_a0.1 4 0.05 1024000 0.1 100 1 256 8 512 4
echo "$(date +%T) done"
