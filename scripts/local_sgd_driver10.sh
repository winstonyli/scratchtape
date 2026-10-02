#!/bin/bash
# Round 11 (docs/gpu_step_design.md): the recipe on `novels6` (6 Gutenberg novels, 4.5 MB train; run
# scripts/fetch_gutenberg.sh first). Waits for the pid in $1, rebuilds, then runs resumably, in priority order.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
while kill -0 "$1" 2>/dev/null; do sleep 20; done
cargo build --release -q --example fused_models_check -j 4 || exit 1
# small model (d 128, the default) then big (d 256), at dropout 0.3
fmc nov_k1 1 32 0.48 windows=1024000 dropout=0.3 $NOV
fmc nov_k4_h100_a0.1 4 32 0.48 windows=1024000 dropout=0.3 $NOV $AVG
fmc nov_big_k1 1 32 0.05 windows=1024000 dropout=0.3 $NOV d_model=256 d_ff=512
fmc nov_big_k4_h100_a0.1 4 32 0.05 windows=1024000 dropout=0.3 $NOV $AVG d_model=256 d_ff=512
fmc nov_k4_h100_a0.1_4m 4 32 0.48 windows=4096000 dropout=0.3 $NOV $AVG
fmc nov_big_k4_h100_a0.1_4m 4 32 0.05 windows=4096000 dropout=0.3 $NOV $AVG d_model=256 d_ff=512
echo "$(date +%T) done"
