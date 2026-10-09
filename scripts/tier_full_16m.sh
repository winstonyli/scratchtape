#!/bin/sh
# Best stack (whiten=0.5, knn:256:0.4:30:50:100, wrap:0.95 after knn) on the 16M-window checkpoint
# (held-out 1.2319, 8M run 1.2408). 8M-run references: model 1.1764, stack 1.1118, stack + wrap 1.0956.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_16m_m0.ckpt
B="lexicon:0.3+words:1:0.25"; K="knn:256:0.4:30:50:100"
$E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 classes=1 tier=$B+$K tier=$B+$K+wrap:0.95 > runs/tier_full_16m.log 2>&1
