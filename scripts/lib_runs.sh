# Shared helpers for the driver scripts; source after `cd`-ing to the repo root.
F=target/release/examples/fused_models_check.exe
mkdir -p runs

# fmc NAME K BATCH LR [key=value ...]: one fused_models_check run (see its header for the keys), appended
# to runs/NAME.log. Resumable: skipped once its final checkpoint exists, continues from runs/NAME.resume.
fmc() {
  [ -f runs/$1_m0.ckpt ] && return
  echo "$(date +%T) start $1"
  $F "$@" >> runs/$1.log 2>&1
}

# The novels6 recipe (docs/gpu_step_design.md, round 11); set dropout, windows and the model size per run.
NOV="weight_decay=0.0002 warmup_windows=6400 momentum=0.9 corpus=novels6"
AVG="alpha=0.1 sync_every_steps=100 shared_init=1"   # K = 4 averaged + alpha 0.1
