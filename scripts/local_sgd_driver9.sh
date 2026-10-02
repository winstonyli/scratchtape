#!/bin/bash
# Round 10 (docs/fusion_results.md): how the averaged model's held-out CE depends on the sync period H, at
# alpha = 0, on aesop, shared init, K = 4 (H = 100 and 1000 exist: 1.651; one model at batch 128: 1.755).
# H = 1 averages every step (close to one model at batch 128), so the curve shows where the gain appears.
cd "$(dirname "$0")/.."
F=target/release/examples/fused_models_check.exe
for h in 1 3 10 30; do
  [ -f runs/hcurve_h${h}_m0.ckpt ] && continue
  echo "$(date +%T) start hcurve_h$h"
  $F hcurve_h$h 4 32 0.48 1024000 1 0.0002 0.3 6400 0.9 0 0 $h 1 0 0.9 aesops_fables 0 128 8 256 4 1 600 >> runs/hcurve_h$h.log 2>&1
done
echo "$(date +%T) done"
