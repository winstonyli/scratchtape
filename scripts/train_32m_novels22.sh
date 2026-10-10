#!/bin/bash
# 32M-window f16 run on novels22 (novels6 + sixteen training-only novels; same held-out text). Compare
# nov_big_k1_d0.1_32m_f16 on novels6: held-out 1.2307, stack 1.0844. Expected ~2.7 h. Fetch first: scripts/fetch_gutenberg_extra.sh.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
fmc nov22_k1_d0.1_32m_f16 1 32 0.05 windows=32768000 dropout=0.1 weight_decay=0.0002 warmup_windows=6400 momentum=0.9 corpus=novels22 d_model=256 d_ff=512
echo "$(date +%T) done"
