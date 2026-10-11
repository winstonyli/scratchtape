#!/bin/sh
# Dynamic evaluation, model alone, first held-out half (part=0/2), RMS steps (sweep5: beta 0.99 at lr 1e-4 and 8-window steps gave
# 1.0568, SGD best 1.0573): lower lrs, a pull back to the checkpoint, and 4-window steps. Output in runs/dyn_sweep/.
E=runs/tier_eval_dyn5.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_sweep
run() { name=$1; shift; $E $C warm=32 corpus=novels22 stride=1 part=0/2 "$@" > runs/dyn_sweep/$name.log 2>&1; }
for lr in 0.00003 0.00005; do run rms0.99_lr${lr}_e8_p0 dyn=$lr dyn_every=8 dyn_rms=0.99; done
for pull in 0.001 0.01; do run rms0.99_lr0.0001_pull${pull}_e8_p0 dyn=0.0001 dyn_every=8 dyn_rms=0.99 dyn_pull=$pull; done
for lr in 0.00005 0.0001; do run rms0.99_lr${lr}_e4_p0 dyn=$lr dyn_every=4 dyn_rms=0.99; done
