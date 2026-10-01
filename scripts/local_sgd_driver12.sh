#!/bin/bash
# Round 13 (docs/gpu_step_design.md): big K = 1 at 8M windows on novels6, where does the curve flatten? Resumable.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
run() {
  [ -f runs/$1_m0.ckpt ] && return
  echo "$(date +%T) start $1"
  $F $1 $2 32 $3 $4 1 0.0002 0.3 6400 0.9 $5 0 $6 $7 0 0.9 novels6 0 $8 $9 ${10} ${11} 1 600 >> runs/$1.log 2>&1
}
run nov_big_k1_8m 1 0.05 8192000 0 0 0 256 8 512 4
echo "$(date +%T) done"
