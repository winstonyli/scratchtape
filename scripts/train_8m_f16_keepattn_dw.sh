#!/bin/bash
# 8M-window f16 run with attention AND the linear weight-gradient matmuls (dW + bias fold) in f32. runs/fmc_dwf32.exe is
# the trainer built with tape.rs's dW matmul switched to matmul_f32 (runs/f16_attn_dw_f32.patch, git-ignored).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
F=runs/fmc_dwf32.exe
TRAIN_F16=1 fmc nov_big_k1_d0.1_8m_f16attn_dw 1 32 0.05 windows=8192000 dropout=0.1 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
