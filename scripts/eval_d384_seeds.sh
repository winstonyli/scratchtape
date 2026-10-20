#!/bin/bash
# Seed repeat of the d=384 16M model at full memory (whiten_rank 256), dyn old weights, parts 0 and 1; usage: eval_d384_seeds.sh <seed>...
# Seed 1 (runs/novels40_s128_16m_d384_m0.ckpt): dyn 0.9689 / 1.0085. Each seed builds its own ~10 GB memcache, removed after its evals.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
M="d_model=384 heads=8 d_ff=768 whiten_rank=256"
for s in "$@"; do
  K=runs/novels40_s128_16m_d384_seed${s}_m0.ckpt
  for i in $(seq 1 60); do [ -f "$K" ] && [ ! -f "runs/novels40_s128_16m_d384_seed${s}.resume" ] && break; sleep 30; done
  [ -f "$K" ] || { echo "no checkpoint $K"; exit 1; }
  tev_begin
  for p in 0 1; do tev seed${s}_d384_p${p}_dyn $K $M part=$p/4 $DYN $(tiers $W_OLD); done
  tev_end
done
echo "$(date +%T) done"
