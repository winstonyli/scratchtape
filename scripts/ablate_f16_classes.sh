#!/bin/bash
# f16 convergence ablation at 2M windows: all-f32 and all-f16 references, then f16 with one matmul class kept on f32
# (F16_F32 bits: 1 linear forward, 2 linear dX, 4 linear dW, 8 attention). docs/tiers_design.md.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
A="1 32 0.05 windows=2048000 dropout=0.1 $NOV d_model=256 d_ff=512"
fmc abl2m_f32 $A
TRAIN_F16=1 fmc abl2m_f16 $A
for b in 1 2 4 8; do TRAIN_F16=1 F16_F32=$b fmc abl2m_f16_keep$b $A; done
echo "$(date +%T) done"
