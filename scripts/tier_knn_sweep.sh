#!/bin/sh
# Experiment B (docs/tiers_design.md): kNN memory sweep on two disjoint held-out subsamples (tune: offset 0, test: offset 8).
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
for o in 0 8; do $E $C stride=16 offset=$o store=30000 threads=4 tier=lexicon:0.3 tier=knn:32:0.1:20 tier=knn:32:0.2:20 tier=knn:32:0.3:20 tier=knn:32:0.1:30 tier=knn:32:0.2:30 tier=knn:32:0.3:30 tier=knn:32:0.1:50 tier=knn:32:0.2:50 tier=knn:32:0.3:50 tier=lexicon:0.3+knn:32:0.2:30 tier=knn:64:0.1:20 tier=knn:64:0.2:20 tier=knn:64:0.3:20 tier=knn:64:0.1:30 tier=knn:64:0.2:30 tier=knn:64:0.3:30 tier=knn:64:0.1:50 tier=knn:64:0.2:50 tier=knn:64:0.3:50 tier=lexicon:0.3+knn:64:0.2:30 > runs/tier_knn_2m_o$o.log 2>&1; done
