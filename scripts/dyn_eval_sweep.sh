#!/bin/sh
# Dynamic evaluation probe, model alone (no tiers), on the 32M novels22 checkpoint: baseline, an SGD learning-rate sweep and
# a control (steps taken from a chunk half the held-out text away), each on the two halves of the held-out text
# (part=0/2 picks the learning rate, part=1/2 confirms it). One process at a time; output in runs/dyn_sweep/.
E=runs/tier_eval_dyn.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_sweep
run() { name=$1; shift; $E $C warm=32 corpus=novels22 stride=1 "$@" > runs/dyn_sweep/$name.log 2>&1; }
for p in 0 1; do
  run base_p$p part=$p/2
  for lr in 0.002 0.005 0.01 0.02; do run dyn${lr}_p$p part=$p/2 dyn=$lr; done
  run ctl0.01_p$p part=$p/2 dyn=0.01 dyn_control=1
done
