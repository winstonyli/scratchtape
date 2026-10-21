#!/bin/bash
# Do the 8-block d=512 and d=640 models want the 6-block model's kNN weights? Old weights: b8 dyn 0.9413 / 0.9821, d640 0.9459 / 0.9850 (parts 0 / 1).
cd "$(dirname "$0")"
W=128:0.4:30:7:100
bash dyn_knn_ckpt.sh n40_d512_b8 runs/novels40_s128_16m_d512_b8_m0.ckpt "d_model=512 heads=8 d_ff=1024 blocks=8 whiten_rank=256" $W
bash dyn_knn_ckpt.sh n40_d640 runs/novels40_s128_16m_d640_m0.ckpt "d_model=640 heads=10 d_ff=1280 whiten_rank=256" $W
