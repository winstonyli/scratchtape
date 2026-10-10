#!/bin/bash
# 8M-window f16 run with the attention matmuls kept on f32 (F16_F32=8). Needs the throwaway class flag: build from
# runs/f16_class_ablation.patch (git-ignored; see docs/tiers_design.md). Compare nov_big_k1_d0.1_8m (f32 1.2408) and _8m_f16 (1.2475).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
F=runs/fmc_ablate.exe
TRAIN_F16=1 F16_F32=8 fmc nov_big_k1_d0.1_8m_f16attn 1 32 0.05 windows=8192000 dropout=0.1 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
