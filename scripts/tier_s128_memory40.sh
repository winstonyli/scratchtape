#!/bin/sh
# Memory-only extra text: the memory is filled from novels40 (novels22 + 18 novels, 23 MB more) while the model, lexicon, words and
# whitener stay on novels22. Same key budget as the full novels22 memory (116733 windows = 14.9M keys), so this tests text
# diversity at a fixed budget (windows are sampled evenly over the larger text). Half 0, base 0.9893 with the whole of novels22.
cd "$(dirname "$0")/.."
E=runs/tier_eval_mem.exe; C=runs/nov22_s128_k1_d0.1_32m_f16_m0.ckpt
$E $C seq=128 warm=96 stride=1 corpus=novels22 memory_corpus=novels40 memory=online whiten=0.5 part=0/2 store=116733 dyn_every=4 dyn=0.05 tier=lexicon:0.3+words:1:0.15+knn:128:0.3:30:20:100 > runs/dyn_tier/s128_mem40_p0.log 2>&1
echo "$(date +%T) done" > runs/dyn_tier/s128_mem40_done
