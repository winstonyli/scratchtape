#!/bin/bash
# Recorded runs (~/.claude/RUNS.md): dyn kNN grid for the 64M model, then the d=384 width test.
cd "$(dirname "$0")/.."
W="$USERPROFILE/.claude/runs/run.ps1"
powershell -NoProfile -File "$W" -Kind sweep -Name dynknn-64m -Priority Normal -Command "bash scripts/dyn_knn_64m.sh"
powershell -NoProfile -File "$W" -Kind train -Name novels40-d384-16m -Priority Normal -Command "bash scripts/train_novels40_d384.sh"
echo "$(date +%T) chain done"
