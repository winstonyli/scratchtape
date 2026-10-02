#!/bin/bash
# Round 15 (docs/gpu_step_design.md): dropout sweep 0.0 and 0.05, big K = 1 at 4M windows on novels6
# (controls: dropout 0.1 = 1.2578, 0.3 = 1.2929). Resumable.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
run() {
  [ -f runs/$1_m0.ckpt ] && return
  echo "$(date +%T) start $1"
  $F $1 $2 32 $3 $4 1 0.0002 ${12} 6400 0.9 $5 0 $6 $7 0 0.9 novels6 0 $8 $9 ${10} ${11} 1 600 >> runs/$1.log 2>&1
}
run nov_big_k1_d0_4m 1 0.05 4096000 0 0 0 256 8 512 4 0.0
run nov_big_k1_d0.05_4m 1 0.05 4096000 0 0 0 256 8 512 4 0.05
echo "$(date +%T) done"
