#!/bin/sh
# One-at-a-time tier re-tune on the adapted model (dynamic evaluation, 4-window SGD 0.05), held-out half 0 (part=0/2); the base
# stack there is 1.0235. Every spec ends in +wrap:0.95. Confirm the winners on half 1 afterwards.
E=runs/tier_eval_dyn6.exe; C=runs/nov22_k1_d0.1_32m_f16_m0.ckpt
mkdir -p runs/dyn_tier
w() { echo "tier=$1+wrap:0.95"; }
LEX="lexicon:0.3"; WRD="words:1:0.25"; KNN="knn:256:0.4:30:50:100"
$E $C warm=32 stride=1 memory=online store=100000 whiten=0.5 part=0/2 dyn_every=4 dyn=0.05 \
  $(w "$LEX+$WRD+$KNN") \
  $(w "$LEX+$WRD+knn:256:0.3:30:50:100") $(w "$LEX+$WRD+knn:256:0.5:30:50:100") \
  $(w "$LEX+$WRD+knn:256:0.4:20:50:100") $(w "$LEX+$WRD+knn:256:0.4:45:50:100") \
  $(w "$LEX+$WRD+knn:256:0.4:30:30:100") $(w "$LEX+$WRD+knn:256:0.4:30:80:100") \
  $(w "$LEX+words:1:0.15+$KNN") $(w "$LEX+words:1:0.4+$KNN") \
  $(w "lexicon:0.2+$WRD+$KNN") $(w "lexicon:0.5+$WRD+$KNN") \
  $(w "$LEX+$WRD+knn:128:0.4:30:50:100") > runs/dyn_tier/retune_p0.log 2>&1
