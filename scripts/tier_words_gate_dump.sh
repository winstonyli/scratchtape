#!/bin/sh
# Full held-out run of the deployable stack (kNN weight 0.4) with the kNN and words per-position dumps for words_gate_fit
# (docs/superpowers/specs/2026-10-08-words-gating-design.md). One spec; no KNN_TIMING (it syncs after every tile).
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
$E $C warm=32 stride=1 memory=online store=100000 \
  tier=lexicon:0.3+words:1:0.25+knn:256:0.4:15 dump=runs/wg_dump_k.bin wdump=runs/wg_dump_w.bin > runs/wg_dump.log 2>&1
