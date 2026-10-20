#!/bin/bash
# Data probe (docs/tiers_design.md "Model-training data probe"): the nov22 seq-128 4M-window plain recipe on novels40 and
# novels90 (neither holds the fresh books), then the fresh-split stack for each; memory caches are deleted after use (9.5 GB each).
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
for c in novels40 novels90; do train16 ${c}_s128_4m windows=4096000 corpus=$c; done
for c in novels40 novels90; do
  [ -f runs/probe_$c.log ] && grep -q "^tier" runs/probe_$c.log && continue
  tev_begin
  tev probe_$c runs/${c}_s128_4m_m0.ckpt part=0/4 $(tiers $W_OLD)
  tev_end
done
echo "$(date +%T) done"
