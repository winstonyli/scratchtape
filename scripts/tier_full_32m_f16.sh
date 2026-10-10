#!/bin/sh
# Best stack (whiten=0.5, knn:256:0.4:30:50:100, wrap:0.95 after knn) on the 16M-window checkpoint
# f16-trained 32M checkpoint (f16 16M: model 1.1684, stack 1.0892; f32 16M: 1.1651 / 1.0876).
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_32m_f16_m0.ckpt
B="lexicon:0.3+words:1:0.25"; K="knn:256:0.4:30:50:100"
$E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 classes=1 tier=$B+$K tier=$B+$K+wrap:0.95 > runs/tier_full_32m_f16.log 2>&1
