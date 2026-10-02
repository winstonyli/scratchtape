#!/bin/bash
# Round 18 (docs/fusion_results.md): d = 384 with dropout 0.15 (control: dropout 0.1 = 1.2457), K = 1,
# 4M windows on novels6. The wider model overfits more (gap 0.150 vs 0.105), so it may want more dropout.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov_d384_k1_d0.15_4m 1 32 0.05 windows=4096000 dropout=0.15 $NOV d_model=384 d_ff=768
echo "$(date +%T) done"
