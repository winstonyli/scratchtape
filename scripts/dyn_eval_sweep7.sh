#!/bin/sh
# Dynamic evaluation, model alone, 8-window plain SGD steps, held-out half 0: adapt only some parameters (dyn_layers; host-side step).
# All parameters at lr 0.1 gives 1.0595 (device SGD). Output in runs/dyn_sweep/.
E=runs/tier_eval_dyn7.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_sweep
run() { name=$1; shift; $E $C warm=32 corpus=novels22 stride=1 part=0/2 dyn_every=8 "$@" > runs/dyn_sweep/$name.log 2>&1; }
run layers_all_check dyn=0.1 dyn_layers=emb,b0,b1,b2,b3,head
for lr in 0.1 0.3; do run layers_head_lr$lr dyn=$lr dyn_layers=head; done
run layers_b3head_lr0.1 dyn=0.1 dyn_layers=b3,head
run layers_b23head_lr0.1 dyn=0.1 dyn_layers=b2,b3,head
run layers_blocks_lr0.1 dyn=0.1 dyn_layers=b0,b1,b2,b3
run layers_emb_lr0.1 dyn=0.1 dyn_layers=emb
