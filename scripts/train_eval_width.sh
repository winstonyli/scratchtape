#!/bin/bash
# Width test: novels40 at d=<d> (<heads> heads, d_ff <dff>, dropout 0.15, 4 blocks) for 16M windows, same recipe otherwise; full-memory stack at whiten_rank=256
# (dyn old weights, parts 0 and 1). Usage: [PARTS="0"] train_eval_width.sh <d> <heads> <dff> [seed] [blocks]; rerunning skips finished training (fmc), so PARTS=<p> redoes one failed eval. Ran d=512 (8, 1024): dyn 0.9568 / 0.9928; d=384 3-seed mean 0.9694 / 1.0087.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
d=$1; seed=${4:-1}; nb=${5:-4}
M="d_model=$d heads=$2 d_ff=$3 blocks=$nb"
N=novels40_s128_16m_d$d; [ "$seed" != 1 ] && N=${N}_seed$seed; [ "$nb" != 4 ] && N=${N}_b$nb
R="windows=16384000 dropout=0.15 weight_decay=0.0002 warmup_windows=3200 momentum=0.9 $M seq=128 score_tail=32 corpus=novels40 seed=$seed"
TRAIN_F16=1 fmc $N 1 16 0.05 $R
K=runs/${N}_m0.ckpt
[ -f "$K" ] || { echo "no checkpoint $K"; exit 1; }
tev_begin
for p in ${PARTS:-0 1}; do tev ${N}_p${p}_dyn $K $M whiten_rank=256 part=$p/4 $DYN $(tiers $W_OLD); done
tev_end
echo "$(date +%T) done"
