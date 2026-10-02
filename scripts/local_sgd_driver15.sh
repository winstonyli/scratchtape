#!/bin/bash
# Round 16 (docs/fusion_results.md): d = 384 (d_ff 768, 8 heads), K = 1, dropout 0.1, 4M windows on novels6
# (control: d = 256 at 1.2578).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov_d384_k1_4m 1 32 0.05 windows=4096000 dropout=0.1 $NOV d_model=384 d_ff=768
echo "$(date +%T) done"
