#!/bin/bash
# Round 11 (docs/gpu_step_design.md): the recipe on `novels6` (6 Gutenberg novels, 4.5 MB train; run
# scripts/fetch_gutenberg.sh first). Waits for the pid in $1, rebuilds, then runs resumably, in priority order.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 20; done
cargo build --release -q --example fused_models_check -j 4 || exit 1
F=target/release/examples/fused_models_check.exe
# run name K lr windows alpha sync shared d heads d_ff blocks  (batch 32, wd 2e-4, dropout 0.3, warmup 6400, momentum 0.9)
run() {
  [ -f runs/$1_m0.ckpt ] && return
  echo "$(date +%T) start $1"
  $F $1 $2 32 $3 $4 1 0.0002 0.3 6400 0.9 $5 0 $6 $7 0 0.9 novels6 0 $8 $9 ${10} ${11} 1 600 >> runs/$1.log 2>&1
}
run nov_k1 1 0.48 1024000 0 0 0 128 8 256 4
run nov_k4_h100_a0.1 4 0.48 1024000 0.1 100 1 128 8 256 4
run nov_big_k1 1 0.05 1024000 0 0 0 256 8 512 4
run nov_big_k4_h100_a0.1 4 0.05 1024000 0.1 100 1 256 8 512 4
run nov_k4_h100_a0.1_4m 4 0.48 4096000 0.1 100 1 128 8 256 4
run nov_big_k4_h100_a0.1_4m 4 0.05 4096000 0.1 100 1 256 8 512 4
echo "$(date +%T) done"
