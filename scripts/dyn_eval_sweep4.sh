#!/bin/sh
# Dynamic evaluation, model alone: even finer steps at lower learning rates (sweep3: 4-window chunks were best at lr 0.05 and
# collapsed at 0.2, so the best lr falls with the chunk size). Both held-out halves; output in runs/dyn_sweep/.
E=runs/tier_eval_dyn3.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_sweep
run() { name=$1; shift; $E $C warm=32 corpus=novels22 stride=1 "$@" > runs/dyn_sweep/$name.log 2>&1; }
for p in 0 1; do
  for lr in 0.02 0.03; do run e4_lr${lr}_p$p part=$p/2 dyn=$lr dyn_every=4; done
  for lr in 0.01 0.02 0.03; do run e2_lr${lr}_p$p part=$p/2 dyn=$lr dyn_every=2; done
  for lr in 0.005 0.01 0.02; do run e1_lr${lr}_p$p part=$p/2 dyn=$lr dyn_every=1; done
done
