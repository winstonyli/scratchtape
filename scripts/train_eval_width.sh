#!/bin/bash
# Width test: novels40 at d=<d> (<heads> heads, d_ff <dff>, dropout 0.15, 4 blocks) for 16M windows, same recipe otherwise; full-memory stack at whiten_rank=256
# (dyn old weights, parts 0 and 1). Usage: [PARTS="0"] [W=<knn weights, default the old ones>] train_eval_width.sh <d> <heads> <dff> [seed] [blocks]; rerunning skips finished training (fmc), so PARTS=<p> redoes one failed eval. Ran d=512 (8, 1024): dyn 0.9568 / 0.9928; d=384 3-seed mean 0.9694 / 1.0087.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
d=$1; seed=${4:-1}; nb=${5:-4}
M="d_model=$d heads=$2 d_ff=$3 blocks=$nb"
N=novels40_s128_16m_d$d; [ "$seed" != 1 ] && N=${N}_seed$seed; [ "$nb" != 4 ] && N=${N}_b$nb
train16 $N dropout=0.15 $M seed=$seed
K=runs/${N}_m0.ckpt
[ -f "$K" ] || { echo "no checkpoint $K"; exit 1; }
tev_begin
for p in ${PARTS:-0 1}; do tev ${N}_p${p}_dyn $K $M whiten_rank=256 part=$p/4 $DYN $(tiers ${W:-$W_OLD}); done
tev_end
echo "$(date +%T) done"
