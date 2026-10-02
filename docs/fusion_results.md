# Horizontal fusion and codistillation: design and run records

Moved out of [gpu_step_design.md](gpu_step_design.md) (2026-10-01), which keeps the device-step design.
The numbered rounds, run records (command, pid, log path, results) and the argument-by-argument history of
`examples/tiny_lm/fused_models_check.rs` live here. Drivers: `scripts/local_sgd_driver*.sh`, `scripts/codist_driver*.sh`
(rounds 11 and later use `scripts/lib_runs.sh`).

## Horizontal fusion and codistillation (in progress, 2026-09-25)

Why: the step is dispatch-bound (~145 launches whatever the batch) and
Windows time-slices separate processes, so K models belong in the same
launches (HFTA, MLSys 2021, arXiv 2102.02344). The aim is the overfitting
gap that decay and dropout barely moved.

Built (commits 9dea605, 3a78555):

- `DeviceParams::upload_models(flat, k)`. Model m's params and grads sit
  at m·stride, and activations hold K equal row slices.
- Linears run as a K-batch matmul, and split-k works per matrix of a
  batch. Reductions run per model on grid y, and cross-entropy returns
  one mean per model.
- `DeviceTape::with_distill(alpha)` adds deep mutual learning (Zhang et
  al. 2018, arXiv 1706.00384; codistillation, Anil et al. 2018, arXiv
  1804.03235): dlogits += α(p − q)/rows, where q is the peers' mean
  prediction on the same row, held constant. Reported losses stay plain
  CE.
- Driver: `examples/tiny_lm/fused_models_check.rs`. It reports each
  model's held-out CE, their mean and the ensemble's, and re-scores each
  model alone at the end.

Checks:

- K = 1 is byte-identical to `training_recipe_check` (32000 windows,
  full recipe).
- `fused_models_match_separate` checks the fused step against separate
  models: step-0 losses bit-equal, params within 1e-5 after 1 step at
  real size.
- A second real-size step drifted by ~1e-4. The cause is one ReLU tie
  (block 1, FFN1 unit 130), not a bug.

Throughput is training ms/step in wall time, measured with CPU `_Total`
at or below ~20%, the eGPU free and Defender real-time protection off:

| batch | K | ms/step | per model |
|---|---|---|---|
| 32 | 1 | 4.66–4.69 | 4.66–4.69 |
| 32 | 2 | 7.38–7.63 | 3.69–3.81 |
| 32 | 4 | 13.12 | 3.28 |
| 32 | 8 | 28.99 | 3.62 |
| 8 | 16 | 12.59–12.89 | 0.79 |
| 8 | 1 | 1.97 (2.13 before batched split-k) | 1.97 |
| 8 | 4 | 3.95 (4.72 before batched split-k) | 0.99 |

Before batched split-k, K = 2 at batch 32 gained nothing (6.4 vs 6.1
ms per model), because split-k was off for batch > 1.

Next, in order:

1. **Clean timings (done 2026-09-30).** Logs `runs/ftime2_*.log`, 48000
   windows, lr 0.3, Defender off, no eGPU users at the pre-check, CPU
   `_Total` 4–21% at each launch (other sessions were compiling, so
   not always under the ~20% bar). Batch 8: K = 1 1.97 ms/step, K = 4
   3.95 (0.99 per model), both after batched split-k. Batch 32, second
   round: K = 1 4.67 (matches the first round's 4.66–4.69), K = 4 14.14
   against 13.12 before; that one launched at 21% CPU and the stretch
   median was 12.5, so read it as contended. The first stretch includes
   JIT, so the median stretch is a little lower than each overall figure.
2. **Codistillation on the recipe.** Recipe: batch 32, lr 0.48, momentum
   0.9, warmup 6400, weight decay 2e-4, dropout 0.3, 1M windows. Arms,
   covering both equal-compute baselines:
   - K = 4 at α = 0: seeds 1–4 trained independently, reporting
     per-model and ensemble CE.
   - K = 4 at α = 0.5 and α = 1.
   - One model with 4× the steps (`training_recipe_check`, 4M windows).
   **Done 2026-09-30** (started 16:16, finished 16:49). Launched by `scripts/codist_driver.sh` (copy in
   `runs/`; bash driver pid 35048, arms run in sequence, each under
   its own process at Normal priority with a shared GPU lease). Logs:
   `runs/codist_k4_a{0,0.5,1}.log` (1024000 windows, seeds 1–4, lr 0.48,
   wd 2e-4, dropout 0.3, warmup 6400, momentum 0.9), then
   `runs/gpu_long4m_drop0.3_wd2e-4_s1.log` (resumable: relaunch the same
   command from the driver). Expected ~35 min total: K = 4 arms ~8 min
   each, the 4M-window single model ~10 min. Comparison point:
   `gpu_long1m_drop0.3_wd2e-4_s{1,2,3}` (seed 1: held-out 1.6834).

   **Result: mutual distillation hurt, at both α.** Held-out CE on the
   371-window deterministic set, final stretch (each fused model
   re-scored alone matched its fused score):

   | arm | per-model | mean | ensemble | ms/step |
   |---|---|---|---|---|
   | K = 4, α = 0 | 1.681 1.673 1.697 1.694 | 1.686 | **1.548** | 13.61 (3.40/model) |
   | K = 4, α = 0.5 | 1.784 1.777 1.785 1.785 | 1.783 | 1.727 | 13.35 |
   | K = 4, α = 1 | 1.943 1.943 1.938 1.936 | 1.940 | 1.896 | 13.35 |
   | 1 model, 4M windows | 1.7175 (train probe 0.943) | | | 4.63 |
   | 1 model, 1M windows, seeds 1–3 | 1.683 1.680 1.696 | 1.686 | | |

   - α = 0 fused equals the separate single-model runs (mean 1.686 vs
     1.686), as it should: fusion changes the launch count, not the math.
   - The peers' mean prediction is a worse target than the labels here:
     per-model CE worsens monotonically with α, and the models collapse
     toward each other, so the ensemble gain shrinks too (0.14 → 0.06 →
     0.04 nats). The constraint acts as extra regularization on a model
     that was already regularized by dropout 0.3 and decay.
   - What did help: averaging four independently trained models, 1.686 →
     1.548 (−0.14), for 4 × 3.40 ms of compute against the 4M-window
     single model's 1.7175 (more steps overfit more).
   - Not measured: train-probe CE for the fused arms (the log prints
     held-out only), so "does α close the train/held-out gap" is answered
     only indirectly, by held-out getting worse. Not tried: α < 0.5
     (e.g. 0.1), or a warm-up of α from 0.
3. **Results into the README (done 2026-09-30).** Throughput,
   codistillation results, and related work: HFTA, Zhang 2018, Anil
   2018, DiLoCo.
   **Profilers checked (2026-09-30):** `gpu_train_check 0 200 profile`
   and `host` run and print per-site tables (GPU 2.32 ms/step, host
   4.77 ms/step; the host run was under ~60–80% CPU from other sessions).

   **Follow-ups (done 2026-09-30, 17:07–17:31)** (`scripts/codist_driver2.sh`,
   bash pid 1121; logs `runs/codist2_k4_*.log`, driver log
   `runs/codist_driver2.log`; ~25 min, CPU was contended by other
   sessions so ms/step is not clean): `fused_models_check` now prints a
   train-probe CE at the end and takes an α ramp (linear from 0 over N
   windows). Arms: α = 0 (for the train-probe baseline), α = 0.1, α = 0.5
   ramped over 512000 windows. Same recipe as above; train-probe is the
   first held-out-sized stretch of the training text, final step:

   | arm | held-out mean | ensemble | train-probe mean | ensemble |
   |---|---|---|---|---|
   | α = 0 | 1.686 | 1.548 | 1.035 | 0.962 |
   | α = 0.1 | **1.650** | 1.558 | 1.100 | 1.042 |
   | α = 0.5 ramped | 1.783 | 1.725 | 1.340 | 1.295 |

   α = 0.1 is the one setting that helps: each model improves by 0.037
   nats and the train/held-out gap narrows (0.65 → 0.55), but the models
   agree more, so the ensemble is slightly worse (1.558 vs 1.548). A
   ramp changes nothing at α = 0.5 (1.7825 vs 1.7827 constant). Ensemble
   of independent seeds (1.548) is still the best held-out number; α = 0.1
   gives the best single model from 4 fused models.
4. **Weight averaging (local SGD, done 2026-09-30).**
   `DeviceParams::average_models` (one launch, test
   `average_models_means_every_slot`) and `fused_models_check` args
   `sync_every_steps shared_init`. Arms from one shared init, K = 4, same
   recipe: H = 0 (control: shared init, never averaged), 100, 1000 steps;
   `scripts/local_sgd_driver.sh` (waiter pid 1421, starts after the
   follow-ups finish, ran 17:32–17:56, logs `runs/local_k4_h*.log`). Simplest form:
   plain parameter mean, no DiLoCo outer optimizer, momentum buffers left
   per model.

   | arm (shared init, K = 4) | held-out | train-probe |
   |---|---|---|
   | H = 0, never averaged: mean / ensemble | 1.693 / 1.553 | 1.035 / 0.963 |
   | H = 100, averaged model | **1.6512** | 0.912 |
   | H = 1000, averaged model | 1.6513 | 0.956 |

   Shared init alone keeps the ensemble gain (1.553 vs 1.548), so the
   models still diverge enough. Averaging gives one model at 1.651,
   better than every independent single model (1.680–1.696) and equal
   to α = 0.1's per-model 1.650, with a *lower* train loss rather than a
   smaller gap (gap 0.74 at H = 100, 0.70 at H = 1000 vs 0.65
   independent). H = 100 and 1000 reach the same held-out CE.
   Caveats: this spends 4 models' compute (4 × batch 32 of data per step)
   on one model, so the fair comparison is a single model at batch 128
   with the same steps (32000, lr 0.48, warmup 25600 windows = 200
   steps, `runs/gpu_b128_drop0.3_wd2e-4_s1.log`, 484 s, 15.1 ms/step).
   Result: held-out **1.7531**, train-probe 0.78. So averaging is not a
   large-batch effect: the batch-128 model fits train far better and
   generalizes worse than the averaged one (1.651, 0.91), which sits
   near the batch-32 single models (1.68–1.70) on fit but beats them
   on held-out. One seed each.

   **Round 3 (done 2026-09-30, 18:16–19:11)** (`scripts/local_sgd_driver2.sh`,
   bash pid 13349, driver log `runs/local_sgd_driver2.log`, ~55 min, CPU
   contended): `DeviceParams::outer_step` (DiLoCo outer Nesterov, test
   `outer_step_matches_host`) and `fused_models_check` args `outer_lr
   outer_mu`. Runs in order: H = 100 averaged at seeds 11 and 21
   (`local_k4_h100_s{11,21}`; seeds chosen so data streams don't overlap
   seed 1's), the batch-128 single model at seeds 2 and 3
   (`gpu_b128_..._s{2,3}`), DiLoCo outer lr 0.7 / mu 0.9 at H = 100 and
   1000 (`diloco_k4_h*`), and H = 100 with α = 0.1
   (`local_k4_h100_a0.1`). Held-out / train-probe of the final (averaged)
   model:

   | arm | held-out | train-probe |
   |---|---|---|
   | H = 100 averaged, seeds 1 / 11 / 21 | 1.6512 / 1.6560 / 1.6464 (mean 1.651) | 0.912 / 0.909 / 0.909 |
   | one model, batch 128, seeds 1 / 2 / 3 | 1.7531 / 1.7515 / 1.7598 (mean 1.755) | 0.784 / 0.791 / 0.787 |
   | DiLoCo outer 0.7 / 0.9, H = 100 | 1.6721 | 0.892 |
   | DiLoCo outer 0.7 / 0.9, H = 1000 | 1.6617 | 0.918 |
   | H = 100 averaged + α = 0.1 | **1.6149** | 0.989 |

   - Averaging beats the batch-128 single model by 0.10 nats at every
     seed (spread within each arm ≤ 0.01), so the effect is real, not a
     seed fluke.
   - DiLoCo's outer optimizer was *worse* than the plain mean at both H
     (1.672, 1.662 vs 1.651), at one seed and untuned (outer lr 0.7 and
     mu 0.9 are the paper's, tuned for large models and long rounds).
   - α = 0.1 and averaging stack: 1.6149 with a smaller gap (0.63) than
     either alone (single seed). Best held-out CE of any run so far.

   **Round 4 (done 2026-09-30, 19:29–21:02)** (`scripts/local_sgd_driver3.sh`,
   bash pid 2033, log `runs/local_sgd_driver3.log`, 11 runs, ~100 min,
   CPU contended): H = 100 with α = 0.1 at seeds 11 and 21; DiLoCo outer
   sweep at H = 100 (lr/mu 1/0 as a check against the plain mean, 1/0.9,
   0.5/0.5, 0.3/0.9, 1/0.5); α = 0.05 and 0.2 with averaging; K = 2 and
   K = 8 with averaging and α = 0.1. Logs `runs/local_k*`, `runs/diloco_k4_h100_lr*`. Held-out / train-probe
   of the averaged model, one seed unless noted:

   | arm (H = 100, shared init) | held-out | train-probe |
   |---|---|---|
   | K = 4, α = 0.1, seeds 1 / 11 / 21 | 1.6149 / 1.6105 / 1.6054 (mean **1.610**) | 0.989 / 0.985 / 0.991 |
   | K = 4, α = 0.05 | 1.6058 | 0.943 |
   | K = 4, α = 0.2 | 1.6372 | 1.070 |
   | K = 2, α = 0.1 | 1.6250 | 1.026 |
   | K = 8, α = 0.1 | 1.6126 | 0.964 |
   | DiLoCo outer lr 1 / mu 0 (should equal plain mean 1.6512) | 1.6561 | 0.917 |
   | DiLoCo 1 / 0.9 | 1.6949 | 0.964 |
   | DiLoCo 0.5 / 0.5 | 1.6404 | 0.906 |
   | DiLoCo 0.3 / 0.9 | 1.6608 | 0.875 |
   | DiLoCo 1 / 0.5 | 1.6613 | 0.920 |

   - Averaging + α = 0.1 holds over three seeds: 1.610 (1.605–1.615),
     against 1.651 averaging alone and 1.755 for one model at batch 128.
   - α is flat between 0.05 and 0.1 and worse by 0.2 (1.637).
   - K saturates at 4: K = 2 1.625, K = 4 1.610, K = 8 1.613.
     K = 8 costs 32.6 ms/step (4.1 per model), so K = 4 is the sweet spot.
   - DiLoCo's outer step doesn't clearly beat the plain mean. The check
     arm (lr 1, mu 0), which is the plain mean up to float rounding,
     landed 0.005 away from it (1.6561 vs 1.6512): rounding differences
     alone move a 32000-step run by that much, so differences under
     ~0.01 here are noise. Best outer arm, lr 0.5 / mu 0.5, is 1.6404
     (−0.011, one seed); higher momentum hurts (1/0.9 is 1.695).

   **Round 5 (done 2026-09-30, 21:06–21:28)** (`scripts/local_sgd_driver4.sh`,
   bash pid 8514, log `runs/local_sgd_driver4.log`, ~30 min):
   `fused_models_check` takes a corpus name (arg 17) and `ngram_baseline`
   takes one as its first argument. Sherlock Holmes (54 KB train, 5955
   bytes = 93 windows held-out, so noisy; 7-gram-family baseline 1.695 at
   order 9, picked on held-out): K = 1, K = 4 independent, K = 4 averaged,
   K = 4 averaged + α = 0.1, all 256000 windows. Aesop: α = 0.1 averaging
   at H = 10 and 30. Not done: a larger model, whose sizes are constants
  in the example and in kernel size tables.

   | arm | held-out final | train-probe | notes |
   |---|---|---|---|
   | sherlock, K = 1 | 2.038 | 0.836 | best mid-run 1.848 |
   | sherlock, K = 4 independent | mean 2.019, ensemble **1.727** | 0.842 | |
   | sherlock, K = 4 averaged H = 100 | 2.065 | 0.614 | best mid-run 1.805 |
   | sherlock, K = 4 averaged + α = 0.1 | **1.883** | 0.677 | best mid-run 1.767 |
   | aesop, α = 0.1, H = 10 | 1.634 | 0.911 | |
   | aesop, α = 0.1, H = 30 | 1.621 | 0.950 | |
   | (H = 100, seeds 1/11/21 from round 4) | 1.610 | 0.988 | |

   - Sherlock: every arm overfits at 256k windows (mid-run bests of
     1.77–1.85 vs finals of 1.88–2.07; "best" is picked on held-out, so
     optimistic). Only the 4-seed ensemble (1.727) gets near the
     7-gram-family 1.695. α helps (−0.18 vs averaging alone, −0.15 vs
     one model); plain averaging does not at this length. The run length
     and dropout were not tuned for a corpus 4× smaller, and 93 windows
     is a noisy test set, so read this as "does not transfer untuned",
     not as a verdict.
   - Aesop: longer H is slightly better (10 → 100: 1.634 → 1.610), so
     H = 100 stays. Per-model data is already different in every arm
     (each model draws its own batches).

   **Round 6 (done 2026-09-30, 21:36–22:49)** (`scripts/local_sgd_driver5.sh`,
   bash pid 10199, log `runs/local_sgd_driver5.log`, ~80 min):
   `fused_models_check` gained `lr_decay_frac` (linear to 0 over the last
   fraction of steps) and model-size args (`d_model heads d_ff blocks`,
   args 19–22; `Config` was already parametric). (A) Sherlock at 64k and
   128k windows, K = 1 / 4 independent / 4 averaged / 4 averaged + α =
   0.1. (B) aesop averaged + α = 0.1 with lr decay over the last 20% and
   50%. (C) A 4× larger model (d 256, d_ff 512, 4 blocks, 8 heads, ~3.4M
   parameters per model): lr 0.48 diverges to NaN by step 250 at lr 0.3
   too, while 0.05 and 0.15 train (smoke, 400 steps: 2.41 / 2.53), so the
   runs use lr 0.05 and 0.1; K = 1 and K = 4 averaged + α = 0.1 at each
   (`runs/big_*`). Final held-out (train-probe); "best" is the lowest
   mid-run value of the first model, picked on held-out, so optimistic:

   | arm | final | best | train-probe |
   |---|---|---|---|
   | sherlock 64k, K = 1 | 1.877 | 1.877 | 1.544 |
   | sherlock 64k, K = 4 independent (mean / ensemble) | 1.860 / 1.743 | | 1.555 |
   | sherlock 64k, averaged | 1.805 | 1.805 | 1.430 |
   | sherlock 64k, averaged + α 0.1 | **1.798** | 1.798 | 1.482 |
   | sherlock 128k, K = 1 | 1.854 | 1.848 | 1.181 |
   | sherlock 128k, K = 4 independent (mean / ensemble) | 1.864 / **1.683** | 1.822 | 1.185 |
   | sherlock 128k, averaged | 1.858 | 1.805 | 0.994 |
   | sherlock 128k, averaged + α 0.1 | 1.785 | 1.767 | 1.069 |
   | aesop, averaged + α 0.1, lr decay last 20% | 1.6226 | 1.611 | 0.859 |
   | aesop, averaged + α 0.1, lr decay last 50% | 1.6320 | 1.611 | 0.846 |
   | big (d 256), K = 1, lr 0.05 | 1.8115 | 1.731 | 0.755 |
   | big, K = 1, lr 0.1 | 1.9897 | 1.731 | 0.604 |
   | big, K = 4 averaged + α 0.1, lr 0.05 | **1.6753** | 1.648 | 0.792 |
   | big, K = 4 averaged + α 0.1, lr 0.1 | 1.7594 | 1.648 | 0.550 |

   - Sherlock at 64k windows: averaging transfers (1.805 vs 1.877 for one
     model, −0.07; α adds −0.007). At 128k it overfits and only α keeps
     it ahead (1.785 vs 1.854 / 1.858). The 7-gram-family 1.695 is beaten
     only by the independent ensemble at 128k (1.683), not by any
     single averaged model. 93 held-out windows; one seed.
   - Lr decay does not help: 1.623 / 1.632 against 1.610 without. The
     decayed runs fit train better (0.85–0.86 vs 0.99) and generalize
     slightly worse, so the final-vs-best gap here is overfitting, not lr
     noise. (The README's "lr decay is the next lever" is answered: no.)
   - Larger model: worse than the small one at this budget. One big
     model overfits (1.81, train 0.76) and averaging recovers 0.14 nats
     (1.675), still above the small model's 1.610. Lr 0.1 overfits more
     than 0.05 (final 1.76 vs 1.68 averaged) though both reach 1.648 mid-run.
     So averaging + α generalizes across model size in direction, but
     1M windows and dropout 0.3 are too much for 4× the parameters on
     214 KB of text; a shorter run or stronger regularization is
     untested. Costs: big K = 4 is 34–35 ms/step (K = 1: 10.5).

   **Round 7 (done 2026-09-30, 22:54–23:54)** (`scripts/local_sgd_driver6.sh`,
   bash pid 13561, log `runs/local_sgd_driver6.log`, ~60 min):
   `DeviceParams::average_groups` (test `average_groups_keeps_groups_apart`)
   and `fused_models_check` arg `groups`. (1) The d = 256 model, K = 4
   averaged + α = 0.1, lr 0.05: 256k windows, 512k windows, 512k windows
   with dropout 0.5 (`runs/big_k4_w*`). (2) K = 8 as 2 groups of 4 (each
   group has its own init, averages only within itself; the evaluation's
   "ensemble" is then the groups' ensemble), α = 0 and 0.1
   (`runs/groups_k8x2_*`). Note α distills across all 8 peers, groups
   included. Final held-out (train-probe), one seed:

   | arm | held-out | train-probe |
   |---|---|---|
   | big, K = 4 averaged + α 0.1, lr 0.05, 256k windows, dropout 0.3 | 1.7622 | 1.356 |
   | same, 512k windows | **1.6641** | 1.108 |
   | same, 512k windows, dropout 0.5 | 1.7943 | 1.411 |
   | K = 8 as 2 groups of 4, α 0: groups / ensemble | 1.644, 1.648 / 1.5734 | 0.909 / 0.880 |
   | K = 8 as 2 groups of 4, α 0.1: groups / ensemble | 1.608, 1.623 / **1.5678** | 0.996, 0.990 / 0.967 |

   - Larger model: 512k windows (1.664) is a little better than 1M
     (1.675), still above the small model's 1.610. 256k is undertrained
     (1.762) and dropout 0.5 underfits (1.794). So no setting tried here
     lets 4× the parameters beat the small model on 214 KB of text.
     Big K = 4 costs 33–34 ms/step.
   - Groups: averaged groups' ensemble (1.573 at α 0, 1.568 at α 0.1)
     beats any single averaged model (1.608–1.648) but not the plain
     4-independent-model ensemble (1.548), at twice the inference cost of
     one averaged model and 8 models of training. The gains do not stack:
     averaging removes the variance that ensembling exploits. α = 0.1
     helps the groups individually (1.646 → 1.616 mean) and the ensemble
     only slightly (1.573 → 1.568), even with distillation across groups.
     The best single deployable model stays K = 4 averaged + α 0.1
     (1.610 over three seeds, one set of weights).

   **Resume + larger corpus (2026-10-01).** `fused_models_check` is now
   resumable (arg 24 `checkpoint_secs`, default 600; `runs/<name>.resume`
   holds step, per-model RNG states, parameters, momentum and outer-step
   state as text, ~77 MB for K = 4). Checked: a K = 4 run with the outer
   step, killed after ~20 s and resumed, wrote byte-identical
   checkpoints to an uninterrupted run, and the resume file was removed.
   Corpus `all_four` = the four bundled books with the last 10% of
   *each* held out (373 KB train, 41 KB = 647 held-out windows, 7-gram
   1.6281 at order 7; aesop alone is unchanged at 1.6546).
   **Round 8 (done 2026-10-01, 00:05–00:56)** (`scripts/local_sgd_driver7.sh`, log
   `runs/local_sgd_driver7.log`, relaunch the script to resume; ~55 min):
   small model K = 1, K = 4 averaged + α 0.1 at 1M and 2M windows, and
   the d = 256 model K = 1 and K = 4 averaged + α 0.1 (lr 0.05), 1M
   windows. Final held-out on `all_four` (7-gram 1.6281), one seed;
   "best" is the lowest mid-run value, picked on held-out:

   | arm | final | best | train-probe | ms/step |
   |---|---|---|---|---|
   | small, K = 1 | 1.6372 | 1.6241 | 1.281 | 5.7 |
   | small, K = 4 averaged + α 0.1, 1M windows | **1.5793** | 1.5786 | 1.229 | 13.6 |
   | small, K = 4 averaged + α 0.1, 2M windows | **1.5675** | 1.5672 | 1.186 | 14.5 |
   | big (d 256), K = 1, lr 0.05 | 1.6417 | 1.6302 | 1.097 | 9.5 |
   | big, K = 4 averaged + α 0.1, lr 0.05 | 1.5969 | 1.5969 | 1.124 | 32.0 |

   - The recipe carries to a 1.75× larger, mixed-style corpus: averaging
     + α turns a model that loses to the 7-gram (1.637) into one that beats
     it by 0.049 (1.579), and 2× the windows adds 0.012 more. Final equals
     best here (no overfitting at 1M–2M windows on this corpus), unlike
     aesop alone.
   - The larger model still doesn't beat the small one (K = 4: 1.597 vs
     1.579; K = 1: 1.642 vs 1.637), though the gap is smaller than on
     aesop (1.675 vs 1.610) and it fits train better (1.12 vs 1.23); at
     373 KB a 4× model is still more capacity than the data supports at
     this lr/dropout. Not tried: the big model with 2M windows or lower
     dropout.
   - No run was interrupted, so resume wasn't exercised in the driver
     (only in the kill test above).

   **Round 9 (done 2026-10-01, 09:17–10:47)** (`scripts/local_sgd_driver8.sh`, log
   `runs/local_sgd_driver8.log`, relaunch to resume; ~95 min): on
   `all_four`, K = 1 and K = 4 averaged + α 0.1 at seeds 11 and 21 (1M
   windows); K = 4 at 4M windows; the d = 256 model K = 4 at 2M windows.
   Final held-out on `all_four` (7-gram 1.6281):

   | arm | seed 1 | seed 11 | seed 21 | mean |
   |---|---|---|---|---|
   | small K = 1, 1M windows | 1.6372 | 1.6252 | 1.6226 | **1.6283** |
   | small K = 4 averaged + α 0.1, 1M | 1.5793 | 1.5849 | 1.5833 | **1.5825** |

   | arm (seed 1) | final | train-probe |
   |---|---|---|
   | small K = 4 averaged + α 0.1, 2M windows | 1.5675 | 1.186 |
   | same, 4M windows | 1.5638 | 1.168 |
   | big (d 256) K = 4, 2M windows | 1.6288 | 0.823 |

   - A single small model ties the 7-gram (mean 1.6283 vs 1.6281);
     averaging + α beats it by 0.046 at every seed (1.579–1.585, spread
     0.006).
   - More windows: 1M 1.579 → 2M 1.5675 → 4M 1.5638. The gain roughly
     halves each time while train-probe keeps falling (1.23 → 1.19 →
     1.17), so the plateau is near 1.56.
   - The big model at 2M windows overfits (final 1.629, train-probe
     0.82) and is worse than at 1M (1.597). It does not catch the small
     model on this corpus under any budget tried.
   - Caveat on every "best" column above for averaged arms: mid-run
     evaluations fall at arbitrary points of the 100-step averaging
     period, so most score the *unaveraged* models (the 4M log's mid-run
     rows read ~1.60 for individual models against 1.564 for the
     averaged final). Only the final row, or a row at a step that is a
     multiple of 100, is the averaged model. Read "final".

   **Round 10 (done 2026-10-01, 10:51–11:25)** (`scripts/local_sgd_driver9.sh`,
   log `runs/local_sgd_driver9.log`, ~32 min): `fused_models_check` now
   evaluates averaged arms only right after a sync (so "best" is
   meaningful). Control for "is averaging just regularization/noise?":
   the sync-period curve at α = 0 on aesop, H = 1, 3, 10, 30 (`runs/hcurve_h*`),
   to set against H = 100/1000 (1.651) and one model at batch 128 (1.755).
   H = 1 is near batch-128 SGD; if the curve runs smoothly from 1.755
   down to 1.651, the gain is the local-drift noise, not the model count.
   Result (K = 4, α = 0, shared init, 1M windows, seed 1; held-out final
   / train-probe):

   | sync period H | held-out | train-probe |
   |---|---|---|
   | 1 | 1.7680 | 0.793 |
   | 3 | 1.7632 | 0.794 |
   | 10 | 1.7018 | 0.828 |
   | 30 | 1.6807 | 0.866 |
   | 100 (seeds 1/11/21) | 1.651 (1.646–1.656) | 0.91 |
   | 1000 | 1.6513 | 0.956 |
   | (one model, batch 128) | 1.755 (1.752–1.760) | 0.79 |

   The curve is smooth and monotone: H = 1 reproduces the batch-128
   model (1.768 vs 1.755; train-probe 0.79 both), and each longer period
   fits train less and generalizes better, up to a plateau by H = 100.
   So the gain comes from letting the models drift apart between syncs
   (local-SGD noise acting as a regularizer), not from having four
   models' data per step. Checked 2026-10-02 (abstract re-fetched): Lin,
   Stich, Patel and Jaggi, "Don't Use Large Mini-Batches, Use Local SGD"
   (ICLR 2020, arXiv 1808.07217), report that post-local SGD (local SGD
   started after a warm phase of ordinary large-batch SGD) generalizes
   much better than large-batch training. That matches the direction of
   our batch-128 vs averaged gap; we ran local SGD from the start and
   did not test their warm-start variant.

   **Round 11 (2026-10-01): a corpus past 373 KB.** `novels6` = six
   Gutenberg novels (Frankenstein, Pride and Prejudice, A Tale of Two
   Cities, Dracula, Great Expectations, Moby-Dick), fetched on request by
   `scripts/fetch_gutenberg.sh` into the gitignored `data/gutenberg/`
   (licence header/footer stripped; 5.04 MB; the tails of Dracula and
   Pride and Prejudice carry publisher ads, which land in held-out). Last
   10% of each book held out: 4.53 MB train, 504 KB = 7873 windows
   held-out; 7-gram (order 7) **1.3567**. 1M windows is ~14 passes, so
   less overfitting than before. `scripts/local_sgd_driver10.sh` (waits for
   the round-10 driver, rebuilds, runs resumably; log
   `runs/local_sgd_driver10.log`): small K = 1, small K = 4 averaged + α 0.1,
   big K = 1, big K = 4 (1M windows), then both K = 4 at 4M windows.
   Ran 11:26–14:12 on the RX 9060 XT (Vulkan), no resumes. Final held-out
   CE (nats/byte; K = 4 rows are the averaged model; 7-gram **1.3567**):

   | run | windows | CE |
   |---|---|---|
   | small K = 1 | 1M | 1.4609 |
   | small K = 4, H = 100, α 0.1 | 1M | 1.4443 |
   | big K = 1 | 1M | 1.4246 |
   | big K = 4, H = 100, α 0.1 | 1M | 1.4435 |
   | small K = 4 | 4M | 1.4132 |
   | big K = 4 | 4M | **1.3084** |

   Readings: (1) at 1M windows averaging helps the small model (−0.017)
   but not the big one (big K = 1 beats big K = 4 by 0.019), the reverse
   of aesop and all_four. (2) Only big K = 4 at 4M crosses the 7-gram
   (−0.048) and was still falling slowly (1.3088 → 1.3084 over the last
   500 steps); the small model plateaus ~1.41. (3) The 4M win mixes three
   changes (windows, K, averaging); round 12 isolates averaging.

   **Round 12 (2026-10-01): big K = 1 at 4M windows**, the control for
   the big K = 4 4M run. `scripts/local_sgd_driver11.sh`, run
   `nov_big_k1_4m`, pid 32852 (Normal priority, GPU-bound), started 14:54,
   log `runs/nov_big_k1_4m.log` and `runs/local_sgd_driver11.log`,
   ~9.6 ms/step × 128000 steps ≈ 20 min if the machine stays quiet
   (round 11's K = 1 big ran 11.5 ms/step). Resumable (`runs/nov_big_k1_4m.resume`).
   **Result (finished 15:27, 33 min): big K = 1 reaches 1.2929, beating
   big K = 4 averaged + α 0.1 (1.3084) at every checkpoint:**

   | windows | big K = 1 | big K = 4 avg + α 0.1 |
   |---|---|---|
   | 1M | 1.4246 | 1.4435 |
   | 2M | 1.3404 | 1.3608 |
   | 3M | 1.3085 | 1.3259 |
   | 4M | 1.2929 | 1.3084 |

   So on this corpus the 4M win over the 7-gram (1.3567) comes from
   data and training, not averaging; K = 1 is ~3× cheaper (~10 vs ~34
   ms/step) and was still falling (−0.016 over the last 1M). Averaging
   helped only the small model (1.4609 → 1.4443 at 1M), which plateaus
   ~1.41. Reading, untested directly: averaging acts as a regularizer
   that pays only when the model overfits (aesop, all_four), not on a
   corpus the model cannot yet exhaust. The "averaged recipe" results of
   rounds 1–10 should be read as "small data, overfitting regime".

   **Round 13 (2026-10-01): big K = 1 at 8M windows** (where does it
   flatten?). `scripts/local_sgd_driver12.sh`, run `nov_big_k1_8m`, pid
   37188 (Normal), started 15:42, log `runs/nov_big_k1_8m.log` and
   `runs/local_sgd_driver12.log`, ~10 ms/step × 256000 steps ≈ 45 min.
   Resumable.
   **Result (finished 16:52, 70 min): 1.2622 at 8M windows**, 0.095 below
   the 7-gram; train-probe 1.1730 (gap 0.089). Per extra 1M windows the
   gain shrinks but does not vanish: 4M 1.2929, 5M 1.2808, 6M 1.2740,
   7M 1.2670, 8M 1.2622 (−0.012, −0.007, −0.007, −0.005). Reading
   (untested): still data/training-limited rather than capacity-limited.

   **Round 14 (2026-10-01): dropout 0.1 vs 0.3**, big K = 1 at 4M windows
   (control `nov_big_k1_4m`, 1.2929; tests whether regularization is now
   too strong). `scripts/local_sgd_driver13.sh`, run
   `nov_big_k1_d0.1_4m`, pid 26248 (Normal), started 17:12, log
   `runs/nov_big_k1_d0.1_4m.log` and `runs/local_sgd_driver13.log`,
   ~35 min. Resumable.
   **Result (finished 17:44): 1.2578 at 4M windows**, 0.035 better than
   dropout 0.3 (1.2929) and better than dropout 0.3 at 8M (1.2622):

   | windows | dropout 0.1 | dropout 0.3 |
   |---|---|---|
   | 1M | 1.3349 | 1.4246 |
   | 2M | 1.2870 | 1.3404 |
   | 3M | 1.2684 | 1.3085 |
   | 4M | 1.2578 | 1.2929 |

   Train-probe 1.1526 (gap 0.105 vs 0.083 for dropout 0.3); last-1M gain
   −0.011, so overfitting is starting. Dropout 0.3, carried over from the
   small-data rounds, was too strong for novels6; the optimum may be
   lower still.

   **Round 15 (2026-10-01): dropout 0.0 and 0.05**, same setup.
   `scripts/local_sgd_driver14.sh`, runs `nov_big_k1_d0_4m` (pid 22772,
   started 18:14) then `nov_big_k1_d0.05_4m`, Normal priority, logs
   `runs/<name>.log` and `runs/local_sgd_driver14.log`, ~35 min each
   (~70 min total). Resumable.
   **Result (finished 19:14): dropout 0.1 is the optimum at 4M windows;
   0.05 is within noise of it, 0.0 overfits.** Held-out CE (K = 1, big):

   | dropout | 1M | 2M | 3M | 4M | train-probe | gap |
   |---|---|---|---|---|---|---|
   | 0.0 | 1.3146 | 1.2988 | 1.2978 | 1.2961 | 1.1112 | 0.185 |
   | 0.05 | 1.3184 | 1.2791 | 1.2668 | 1.2588 | 1.1323 | 0.127 |
   | 0.1 | 1.3349 | 1.2870 | 1.2684 | 1.2578 | 1.1526 | 0.105 |
   | 0.3 | 1.4246 | 1.3404 | 1.3085 | 1.2929 | 1.2096 | 0.083 |

   0.05 vs 0.1 differ by 0.001 (one seed, not separable); 0.0 stalls
   after 2M windows. Dropout 0.1 is the novels6 default from here. The
   best value may shift with window count (not run).

   **Round 16 (2026-10-01): d = 384** (8 heads, d_ff 768, 4 blocks), K = 1,
   dropout 0.1, lr 0.05 (not retuned), 4M windows; control is d = 256 at
   1.2578. `scripts/local_sgd_driver15.sh`, run `nov_d384_k1_4m`, pid
   26428 (Normal), started 19:49, log `runs/nov_d384_k1_4m.log` and
   `runs/local_sgd_driver15.log`, ~19 ms/step × 128000 ≈ 41 min.
   Resumable.
   **Result (finished 20:44, 55 min): 1.2457**, 0.012 better than d = 256
   (1.2578) and 0.111 under the 7-gram, at ~1.9× the time per step:

   | windows | d = 384 | d = 256 | lead |
   |---|---|---|---|
   | 1M | 1.3092 | 1.3349 | 0.026 |
   | 2M | 1.2677 | 1.2870 | 0.019 |
   | 3M | 1.2557 | 1.2684 | 0.013 |
   | 4M | 1.2457 | 1.2578 | 0.012 |

   Train-probe 1.0961 (gap 0.150 vs 0.105): the wider model overfits more,
   so capacity pays a little. lr 0.05 looked stable (not retuned). Not yet
   run: d = 384 at 8M windows; dropout 0.15 on d = 384.

   **Round 17 (2026-10-01): dropout 0.1 at 8M windows**, big K = 1 on
   novels6 (controls: dropout 0.1 at 4M 1.2578; dropout 0.3 at 8M
   1.2622). `scripts/local_sgd_driver16.sh`, run `nov_big_k1_d0.1_8m`,
   pid 36024 (Normal, RX 9060 XT Vulkan), started 23:51, log
   `runs/nov_big_k1_d0.1_8m.log` and `runs/local_sgd_driver16.log`,
   ~10 ms/step × 256000 steps ≈ 45–70 min. CPU load 54% at launch from
   other sessions (affects ms/step only); Defender real-time off.
   Resumable.
   **Result (finished 01:03, 72 min): 1.2408 at 8M windows**, 0.116 under
   the 7-gram and 0.021 better than dropout 0.3 at 8M (1.2622); train-probe
   1.1172 (gap 0.124, up from 0.105 at 4M). Held-out CE by 1M windows:
   4M 1.2578, 5M 1.2518, 6M 1.2475, 7M 1.2430, 8M 1.2408 (gains −0.006,
   −0.004, −0.0045, −0.002): nearly flat, with the gap widening. d = 384
   at 4M (1.2457) is within 0.005 of this at half the windows; d = 384 at
   8M is not run.
   **Later: weight averaging.** Periodically average replicas that share
   an init (local SGD; DiLoCo, arXiv 2311.08105).
