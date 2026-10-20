#!/bin/bash
# The seq-128 nov22 recipe (train_32m_novels22_seq128.sh) with twice the windows (64M windows of 64 bytes-equivalent steps, ~3 h).
# Resumable: rerun to continue from runs/nov22_s128_k1_d0.1_64m_f16.resume.
cd "$(dirname "$0")/.."
. scripts/lib_runs.sh
train16 nov22_s128_k1_d${D:-0.1}_64m_f16 corpus=novels22 dropout=${D:-0.1} windows=32768000
echo "$(date +%T) done"
