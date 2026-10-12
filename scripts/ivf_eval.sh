#!/bin/sh
# Short eval (seq 128, half 0 / 32 = 123 chunks, store 50000, dynamic evaluation) with extra options: ivf_eval.sh <name> [key=value ...]
# Prints the timing line and the stack CE; logs to runs/dyn_tier/ivf_<name>.log.
cd "$(dirname "$0")/.."
E=${EXE:-runs/tier_eval_ivf.exe}; C=runs/nov22_s128_k1_d0.1_32m_f16_m0.ckpt
n=$1; shift
A="seq=128 warm=96 stride=1 corpus=novels22 memory=online whiten=0.5 part=0/32 store=50000 dyn_every=4 dyn=0.05 tier=lexicon:0.3+words:1:0.15+knn:128:0.3:30:20:100"
$E $C $A "$@" > runs/dyn_tier/ivf_$n.log 2>&1
echo "$n $(grep -h '^timing' runs/dyn_tier/ivf_$n.log | cut -c1-110) $(grep -o 'CE [0-9.]*' runs/dyn_tier/ivf_$n.log | tail -1)"
