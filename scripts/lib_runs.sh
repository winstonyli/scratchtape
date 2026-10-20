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

# The novels6 recipe (docs/fusion_results.md, round 11); set dropout, windows and the model size per run.
NOV="weight_decay=0.0002 warmup_windows=6400 momentum=0.9 corpus=novels6"
AVG="alpha=0.1 sync_every_steps=100 shared_init=1"   # K = 4 averaged + alpha 0.1

# tier_eval on the fresh split (corpus=fresh22 memory_corpus=fresh150). Wrap the evals that share one memory build in
# tev_begin / tev_end (the cache is ~10 GB; tev_end deletes the entries built since tev_begin), and call
#   tev LOG CKPT [key=value ...] [tier=... ...]     -> runs/LOG.log   (CKPT is a path; STORE=500000 halves the memory)
# with `tiers W...` expanding kNN weights (k:lambda:temp:temp_doc:shift) into lexicon+words+knn tier= arguments.
E=target/release/examples/tier_eval.exe
EVB="seq=128 warm=96 stride=1 corpus=fresh22 memory_corpus=fresh150 memory=online whiten=0.5 batch=128 ivf=12288:8 keybits=2 rotate=1 memcache=runs/memcache"
DYN="dyn=0.04 dyn_every=2 dyn_nn=16 dyn_nn_lr=0.05"
W_OLD=128:0.5:30:15:100      # the original kNN weights
W_TUNED=128:0.8:30:22:100    # retuned for no dyn
tiers() { local w; for w in "$@"; do printf 'tier=lexicon:0.2+words:1:0.1+knn:%s ' "$w"; done; }
tev_begin() { TEV_BEFORE=$(ls runs/memcache); }
tev_end() { local f; for f in $(ls runs/memcache); do echo "$TEV_BEFORE" | grep -qx "$f" || rm -f "runs/memcache/$f"; done; }
tev() { local log=$1 ck=$2; shift 2; $E $ck $EVB store=${STORE:-1000000} "$@" > runs/$log.log 2>&1; }

# train16 NAME [key=value ...]: the seq-128 fresh-split recipe (novels40, d_model 256, d_ff 512, 16.38M windows, batch 16, lr 0.05, f16, lr
# constant after a 3200-window warmup); key=value arguments override the defaults and may add keys (heads=, blocks=, seed=, init=, ...).
# fused_models_check rejects a key given twice, hence the merge here.
train16() {
  local name=$1; shift
  local -A kv=([windows]=16384000 [dropout]=0.1 [weight_decay]=0.0002 [warmup_windows]=3200 [momentum]=0.9 [seq]=128 [score_tail]=32 [d_model]=256 [d_ff]=512 [corpus]=novels40)
  local a k args=()
  for a in "$@"; do kv[${a%%=*}]=${a#*=}; done
  for k in "${!kv[@]}"; do args+=("$k=${kv[$k]}"); done
  TRAIN_F16=1 fmc $name 1 16 0.05 "${args[@]}"
}
