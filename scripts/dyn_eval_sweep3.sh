#!/bin/sh
# Dynamic evaluation, model alone: finer steps (dyn_every windows per chunk) and several steps per chunk, at a few learning
# rates, on both held-out halves (see dyn_eval_sweep.sh). Output in runs/dyn_sweep/.
E=runs/tier_eval_dyn3.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_sweep
run() { name=$1; shift; $E $C warm=32 corpus=novels22 stride=1 "$@" > runs/dyn_sweep/$name.log 2>&1; }
for p in 0 1; do
  for every in 16 8 4; do for lr in 0.05 0.1 0.2; do run e${every}_lr${lr}_p$p part=$p/2 dyn=$lr dyn_every=$every; done; done
  for k in 2 4; do for lr in 0.05 0.1 0.2; do run s${k}_lr${lr}_p$p part=$p/2 dyn=$lr dyn_steps=$k; done; done
done
