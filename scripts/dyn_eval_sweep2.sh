#!/bin/sh
# Second dynamic-evaluation sweep: higher SGD learning rates (the first, up to 0.02, was still improving), plus controls.
# Same setup as dyn_eval_sweep.sh.
E=runs/tier_eval_dyn.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_sweep
run() { name=$1; shift; $E $C warm=32 corpus=novels22 stride=1 "$@" > runs/dyn_sweep/$name.log 2>&1; }
for p in 0 1; do
  for lr in 0.05 0.1 0.2 0.4; do run dyn${lr}_p$p part=$p/2 dyn=$lr; done
  for lr in 0.05 0.2; do run ctl${lr}_p$p part=$p/2 dyn=$lr dyn_control=1; done
done
