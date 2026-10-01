#!/bin/bash
# Local-SGD arms (docs/gpu_step_design.md, "Later: weight averaging"): K = 4 from one shared init,
# averaged every H steps. Waits for the pid in $1 (the codistillation follow-up driver) to exit,
# then rebuilds the example, so the running binary isn't replaced under it.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 20; done
cargo build --release -q --example fused_models_check -j 4 || exit 1
F=target/release/examples/fused_models_check.exe
R="0.48 1024000"            # lr, windows; then seed, wd, dropout, warmup, momentum, alpha, ramp windows, sync steps, shared init
for h in 0 100 1000; do
  echo "$(date +%T) start local_k4_h$h"
  $F local_k4_h$h 4 32 $R 1 0.0002 0.3 6400 0.9 0 0 $h 1 > runs/local_k4_h$h.log 2>&1
done
echo "$(date +%T) done"
