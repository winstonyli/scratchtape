#!/bin/sh
# Dynamic evaluation, model alone, 8-window steps, first held-out half (part=0/2): RMS-normalised steps (host-side prototype,
# ~2 min per pass). SGD at this chunk size: 1.0595 (lr 0.1). Output in runs/dyn_sweep/.
E=runs/tier_eval_dyn5.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_sweep
run() { name=$1; shift; $E $C warm=32 corpus=novels22 stride=1 "$@" > runs/dyn_sweep/$name.log 2>&1; }
for b in 0.9 0.99; do for lr in 0.0001 0.0003 0.001; do run rms${b}_lr${lr}_e8_p0 part=0/2 dyn=$lr dyn_every=8 dyn_rms=$b; done; done
