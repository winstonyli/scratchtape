#!/bin/sh
# Per-source neighbour weighting under write=first: knn:256:<lambda>:<temp>:<temp_doc>:<shift> (in-document neighbours use temp_doc and
# get `shift` taken off their squared distance). Slice (default part=3/6) or pass a different part=; the first two specs are the gate
# (plain == explicit temp_doc=temp, shift=0). docs/tiers_design.md.
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt; P=${1:-part=3/6}
S="tier=knn:256:0.4:15 tier=knn:256:0.4:15:15:0"
for L in 0.4; do for TD in 15 25 40; do for SH in 0 20 40 80; do S="$S tier=knn:256:$L:15:$TD:$SH"; done; done; done
$E $C warm=32 $P store=100000 memory=online write=first $S > runs/tier_doc_split.log 2>&1
