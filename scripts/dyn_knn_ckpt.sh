#!/bin/bash
# kNN weights under dyn for one checkpoint; usage: dyn_knn_ckpt.sh <name> <ckpt> "<model args>" W...; part 1 first, then part 0 (one memory build, whiten_rank=256 in the model args).
# ROUND=r2 tags the logs. History: the 6-block d=512 model (old 128:0.5:30:15:100 dyn 0.9820 / 0.9409) won with 128:0.4:30:7:100, 0.9782 / 0.9366 (docs/tiers_design.md "kNN weights retuned for the 6-block model").
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
name=$1; K=$2; M=$3; shift 3
G=$(tiers "${@:-$W_OLD}")
tev_begin
for p in 1 0; do tev ${name}_dynknn${ROUND:+_$ROUND}_p$p $K $M part=$p/4 $DYN $G; done
tev_end
echo "$(date +%T) done"
