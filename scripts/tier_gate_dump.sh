#!/bin/sh
# NOTE: written for the write-after, merged-search online memory. Today memory=online writes first and searches train and
# in-document keys separately, so this reproduces its logged numbers only at commit 92266bc (before 1cce0ef).
# Full held-out run of the deployable stack with a per-position dump for gate_fit (docs/superpowers/specs/2026-10-07-uncertainty-gating-design.md).
# Only the full-stack spec is scored (the knn search is shared, so this costs about the same as tier_full9). Needs a quiet machine: run the LONG_RUNS checks first.
# No KNN_TIMING: it syncs after every tile and slows the run; the "speed:" lines give s/chunk.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
$E $C warm=32 stride=1 memory=online store=100000 \
  tier=lexicon:0.3+words:1:0.25+knn:256:0.5:15 dump=runs/tier_gate_dump.bin > runs/tier_gate_dump.log 2>&1
