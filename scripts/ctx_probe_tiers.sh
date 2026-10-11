#!/bin/sh
# Does the context-length gain survive under the tiers? Stack (whiten=0.5, knn:256:0.4:30:50:100, wrap:0.95) on the
# ctx_probe seq 64/128/256 models (novels6), held-out half 0, same scored bytes (warm = seq - 32). Store size in bytes is kept
# fixed (store = 100000 windows of 64 bytes). Logs in runs/ctx_probe_tiers_s*.log.
cd "$(dirname "$0")/.."
E=runs/tier_eval_seq.exe
B="lexicon:0.3+words:1:0.25"; K="knn:256:0.4:30:50:100"
for s in 64 128 256; do
  $E runs/ctx_probe_s${s}_m0.ckpt corpus=novels6 seq=$s warm=$((s-32)) stride=1 part=0/2 memory=online store=$((100000 * 64 / s)) whiten=0.5 tier=$B+$K tier=$B+$K+wrap:0.95 > runs/ctx_probe_tiers_s$s.log 2>&1
done
echo "$(date +%T) done" >> runs/ctx_probe_tiers_s256.log
