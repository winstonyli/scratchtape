#!/bin/bash
# Round 9 (docs/gpu_step_design.md), all_four corpus: seeds 11 and 21 for K = 1 and K = 4 averaged + alpha 0.1
# (1M windows), K = 4 at 4M windows, and the d = 256 model K = 4 at 2M windows. Resumable like driver7.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
# run name K lr windows seed alpha sync shared d heads d_ff blocks  (batch 32, wd 2e-4, dropout 0.3, warmup 6400, momentum 0.9)
run() {
  [ -f runs/$1_m0.ckpt ] && return
  echo "$(date +%T) start $1"
  $F $1 $2 32 $3 $4 $5 0.0002 0.3 6400 0.9 $6 0 $7 $8 0 0.9 all_four 0 $9 ${10} ${11} ${12} 1 600 >> runs/$1.log 2>&1
}
for s in 11 21; do
  run all4_k1_s$s 1 0.48 1024000 $s 0 0 0 128 8 256 4
  run all4_k4_h100_a0.1_s$s 4 0.48 1024000 $s 0.1 100 1 128 8 256 4
done
run all4_k4_h100_a0.1_4m 4 0.48 4096000 1 0.1 100 1 128 8 256 4
run all4_big_k4_h100_a0.1_2m 4 0.05 2048000 1 0.1 100 1 256 8 512 4
echo "$(date +%T) done"
