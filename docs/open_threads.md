# Open threads and parked items

Moved verbatim from the README (2026-10-14). The README keeps a short index; live research lines are in `docs/tiers_design.md`.
Deferred work moved to `PARKED.md` (2026-10-17); entries below that were pure deferrals are removed, mixed ones keep their closed-or-measured text.


- **Averaged-model recipe: where it stops (2026-10-01).** On the four
  bundled books, K = 4 averaged + α = 0.1 reaches 1.5825 (three seeds)
  against the 7-gram's 1.6281 and a single model's 1.6283, and plateaus
  near 1.56 by 4M windows. The d = 256 model never beat the small one
  (best 1.597). Open: a much larger corpus (well past 373 KB) where a
  bigger model might pay off, and why averaging beats one model with 4×
  the batch (1.651 vs 1.755 on aesop). The train-probe suggests a
  regularization effect, untested directly (e.g. against 4 batch-32
  models with the same gradient noise but no averaging).
  On six Gutenberg novels (4.5 MB train, 7-gram 1.3567) the big model
  with K = 4 averaged + α = 0.1 at 4M windows reaches 1.3084; the small
  model plateaus ~1.41, and at 1M windows averaging does not help the big
  model. The control, big K = 1 at 4M windows, reaches 1.2929 and beats
  big K = 4 averaged at every checkpoint (1.3084 at 4M), so the novels6
  win over the 7-gram comes from data and training, not averaging; the
  averaged recipe above is a small-data, overfitting-regime result.
  Big K = 1 keeps improving to 1.2622 at 8M windows (0.095 under the
  7-gram), with gains shrinking to ~0.005 per extra 1M.
  Lowering dropout from 0.3 to 0.1 (carried over from the small-data
  rounds) gives 1.2578 at 4M windows, better than 0.3 at 8M.
  A dropout sweep at 4M windows puts the optimum at 0.1 (0.05 is within
  noise, 0.0 overfits).
  Widening to d = 384 gives 1.2457 at 4M windows (0.012 better, ~1.9×
  the step cost); with dropout 0.15 it reaches 1.2413 (round 18).
  At dropout 0.1 the big model reaches 1.2408 at 8M windows (0.116 under
  the 7-gram), nearly flat past 6M.
  See docs/fusion_results.md, rounds 11–18.

- **GPU training backend: done** (2026-09-23 to 09-25). Built as `src/gpu_step/` (milestones 1-6 in
  `docs/gpu_step_design.md`, which also holds the background decisions): a device-resident step at ~1.5 ms on DX12 / ~2.0 ms on
  Vulkan, ~150x the single-threaded CPU tape, to the same CE as the CPU (chaotic-twin noise). Decided along the way: raw wgpu
  was dropped for cubecl on Vulkan SPIR-V (tiled f32 matmul by default, matrix cores optional); the CPU tape stays a separate
  reference because cubecl's CPU runtime loses to `NdArray` at step granularity; the NPU is inference-only and was ruled out.
  The old text of this thread (adapter choice, dispatch-overhead and cubecl-spike measurements, launch census, per-milestone
  numbers, the 2026-09-24 contention note) is in git history before the 2026-10-14 prune.
- **Parked: report the tracel-llvm space-in-path bug upstream.** The
  bundler's `get_libs` splits `llvm-config --libs` output on whitespace,
  which breaks any Windows install path containing a space. The fix is to
  use `--libnames` (`spikes/cubecl_spike/PATCHES.md`). Every version
  through 23.1.0-3 has the bug. It's parked, not dropped: reporting it
  posts publicly, so it waits on the owner's go-ahead.
  **GPU priority doesn't fix eGPU contention (2026-09-24, `gpu_priority_check.rs`).**
  Windows has a per-process GPU priority, separate from CPU priority and
  settable from outside (`D3DKMTSetProcessSchedulingPriorityClass`).
  With another session's job on the eGPU, dropping a heavy load to Idle
  GPU priority left a small probe's p95 at ~0.5–0.64 s, the same as at
  Normal. The stalls look like the third job's unpreempted packets, so
  turn-taking is the fix: `gpu_lease.rs`, which both GPU examples now
  hold, and the matching rule in LONG_RUNS.md.

  **GPU-feeding processes need Normal CPU priority (2026-09-24,
  `gpu_cpu_priority_check.rs`).** `gpu.rs` blocks in `device.poll` rather
  than sleep-polling, but that didn't protect it. With 16 BelowNormal
  spinner threads, a BelowNormal `gpu_matmul` round trip's median rose from
  5–8 ms to 138–171 ms (~25×). At Normal it was unaffected. So GPU training
  runs at Normal, the exception LONG_RUNS.md already makes for burn.
- **Closed/parked: byte-class analysis of the tiers (2026-10-15).** Tools: `tier_eval classes=1` (saving by word class, `WordClasses`), `prompts=<n> generate=<len>` (cold-window pooled version, no memory), `generate=<len>` (sampling), `byte_gap=1`, `survey=1`. Findings, all in `docs/tiers_design.md`: names and rare words carry the lexicon+words saving; kNN's increment is structure, common words and the straight-quote bytes; an in-document word cache has a 0.66e-3 ceiling (closed); the word tiers add 0.0059 nats inside the full stack (0.0108 without dyn) and are kept; normalising the quote convention is worth at most 0.0076 and is not worth a retrain. Done: lexicon and words share one `WordLists` (exact); folding them into one combine rule loses 0.003 (mask kept). Parked, not dropped: inspecting which of the common-lowercase-word bytes (56% of bytes, 37% of the saving) stay expensive.
- **Closed: tier-side levers (2026-10-15).** After the byte-class analysis, no tier-side or context-length lever is left above the 0.003 bar: the remaining CE is next-word choice on common words (first letters, 45% of CE; `docs/tiers_design.md` "Full-stack check after the `WordLists` change"), and a 256-byte window gained only -0.0021 on the stack ("Context length under the tiers"). What is left is a model-side change judged on the fresh split (a retrain with lr decay built in, 6 h; the anneal evidence says at most a tie in the stack), parked above.
- **Closed: TRIME-style memory-augmented training (2026-10-15).** Built and measured on the fresh split at a 4M-window pair: the stack is 0.0070 nats/byte worse than the plain arm (1.0599 vs 1.0529), the model alone 0.021 worse. Code removed (git 284f10c). Untested variants, parked not dropped: BM25-neighbour external batches, retuned kNN weights for the ffn tap; details and caveats in `docs/tiers_design.md` "TRIME-style memory-augmented training: negative result".
- **Live: model-training data and kNN weights (2026-10-16).** novels40/90 beat novels22 on the fresh-split stack (4M windows); the overlap explanation failed; retuning the kNN weights (`128:0.8:30:22:100`) is worth ~0.01 for every 4M model and ties 40 with 90. Best so far: novels40 4M, tuned weights, 1.0253 on part 0 (the 64M novels22 model with old weights: 1.0299). Open: retune the 64M model and the full-stack settings, finer sweep of lambda/temp_doc/k, train novels40/90 longer (8M+), seed repeat. See `docs/tiers_design.md` "kNN weights retuned per model".
- **Update (2026-10-16): book count settled.** novels90 at 16M windows ties novels40 (no dyn tuned 1.0120 vs 1.0094; dyn 0.9842 vs 0.9829), so the best stack stays novels40 16M (dyn 0.9829, no dyn 1.0094). Still open: longer than 16M windows (32M), part-1 confirmation and a seed repeat of the best config. See `docs/tiers_design.md` "novels90 at 16M windows".
- **Update (2026-10-16): part-1 check passed.** novels40 16M on part 1/4: no dyn 1.0506 old / 1.0406 tuned, dyn 1.0208; ordering and tuned-weight gain match part 0. Open: 32M windows, seed repeat.
- **Update (2026-10-17): 32M windows.** novels40 32M: dyn improves (0.9787 part 0, 1.0160 part 1; -0.004/-0.005 vs 16M), frozen no-dyn stack gets slightly worse (+0.003/+0.004), model alone better. Novels40 16M still beats the 64M novels22 model on part 1 (no dyn 1.0406 vs 1.0706, dyn 1.0208 vs 1.0319). Open: seed repeat; whether the dyn gain keeps going at 64M windows (~6 h).
- **Update (2026-10-17): seed repeat and 32M retune.** Seed noise for a trained model is ~0.002. The 32M dyn gain stands (-0.005 vs the two-seed 16M mean); the 32M no-dyn loss is within ~1.5x noise and not a kNN calibration miss (retune ties). Open: 64M-window novels40 (~6 h) to see whether the dyn gain continues; nothing else on this thread is above the noise.
- **Update (2026-10-17): 64M windows (continued from 32M).** novels40 dyn stack 0.9765 (part 0) / 1.0131 (part 1); the gain over 32M halves again (0.0022 / 0.0029), near the seed noise. Training budget at this model size is nearly exhausted; the remaining lever is model or corpus size (`ROADMAP.md` v0.3).
- **Update (2026-10-18): dyn kNN weights for 64M.** Old `0.5:30:15` stays optimal under dyn (grid of 9 on both parts, best alternative within 0.0007). Closed.
- **Update (2026-10-18): width wins.** d=384 (novels40, 16M windows) beats d=256 by 0.014-0.018 in the stack at matched half memory (dyn 0.9727 / 1.0126 vs 0.9901 / 1.0291), 4x the bar. Capacity is the binding lever, not training budget. Full memory does not fit on the card at d=384. See `docs/tiers_design.md` "Width test"; next: fit the memory, d=512 / depth.
- **Update (2026-10-18): full-memory d=384.** `whiten_rank` narrows keys; d=384 at rank 256 with the full 125M-key memory gives the new best stack: dyn 0.9689 / 1.0085, no dyn tuned 0.9989 / 1.0321 (parts 0 / 1). Rank 320 does not fit on the card. Next: d=512 or more blocks (key width fixed at 256-288), a longer d=384 run, seed repeat of d=384.
- **Update (2026-10-18): d=384 seed repeat.** Three seeds, dyn stack mean 0.9694 / 1.0087 (spread <0.0015); width win confirmed. Open: d=512 / more blocks, longer d=384.
- **Update (2026-10-19): d=512.** One seed, dyn stack 0.9568 / 0.9928 (-0.013 / -0.016 vs the d=384 mean); width still pays. Open: d=640+ / more blocks, seed repeat of d=512.
- **Update (2026-10-19): d=640.** One seed, dyn stack 0.9459 / 0.9850 (-0.011 / -0.008 vs d=512); gains are shrinking but still above the bar. Open: d=768 (dispatch limit fixed 2026-10-20), more blocks, seeds.
- **Update (2026-10-20): depth.** d=512 with 6 blocks: dyn stack 0.9409 / 0.9820, best so far (-0.005 / -0.003 vs d=640 at the same step cost). Open: 8 blocks or d=640 with 6, 32M continuation, key width, seeds.
- **Update (2026-10-21): 8 blocks.** d=512, 8 blocks (17M parameters): dyn stack 0.9413 / 0.9821, a tie with 6 blocks, though the model alone improves (0.9582 / 1.0179). The stack has saturated in model size; the kNN saving shrinks as the model improves. Open: key width above 256 dims, retuned kNN weights, seeds.
- **Update (2026-10-21): kNN weights for the 6-block model.** Retuned under dyn (two rounds, 16 settings): `128:0.4:30:7:100` gives dyn 0.9782 / 0.9366, -0.0038 / -0.0043 vs the old weights; the surface is flat around it. Open: do the 8-block and d=640 models want the same; key width; seeds.
- **Update (2026-10-21): weights across models.** `128:0.4:30:7:100`: 8 blocks dyn 0.9317 / 0.9715 (-0.0096 / -0.0106, best stack), d=640 no change (-0.0014 / -0.0004). The retune is per-model. Baseline rerun confirmed (0.9413 / 0.9821 exactly). Open: depth beyond 8 with the new weights, key width.
- **Update (2026-10-21): 8-block grid.** 7 settings around `128:0.4:30:7:100` all within 0.002; best 0.9713 / 0.9315 (a tie). Weights settled. Baseline rerun reproduced 0.9413 / 0.9821, so the gain is real. Open: depth beyond 8, key width.
