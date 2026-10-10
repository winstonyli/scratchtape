#!/bin/bash
# The 32M-window recipe in the f16 default (attention f32): twice the 16M run (held-out 1.2333, stack 1.0892; f32 16M 1.2319 / 1.0876).
# Expected ~1.6 h of training at ~5.5 ms/step. docs/tiers_design.md.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov_big_k1_d0.1_32m_f16 1 32 0.05 windows=32768000 dropout=0.1 $NOV d_model=256 d_ff=512
echo "$(date +%T) done"
