# Open threads and parked items

Moved verbatim from the README (2026-10-14). The README keeps a short index; live research lines are in `docs/tiers_design.md`.

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

- **Replay + Fisher consolidation together** — humble-cortex found them
  synergistic. Here each was only tried alone, and Fisher's new-task cost
  might be offset by replay.
- **CLS-ER-style two-speed memory** — fast and slow EMA copies of the
  model with a consistency loss (Arani et al. 2022).
- **Parked: fold the copy-and-tweak example families (2026-09-30).**
  `memory_tier_multigen*`, `memory_tier_reconsolidation*` and
  `tiny_lm_scaled*` are the same experiment with one knob changed (corpus,
  model size, budget, replay on/off). A parameterized runner would shrink
  them, but every README number is a record of one of these runs, so each
  family must be re-run and compared byte for byte before and after. The
  cheap path is extracting only what's already identical (half-life
  curation, corpus loading, phase splitting), one family at a time,
  `multigen*` first; `common::curate` is the pattern. The full rewrite
  isn't worth it unless more variants are planned.
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
- **Parked: model-side changes (2026-10-15).** Closed or measured: dynamic-evaluation variants, `dyn_nn`, meta-trained adaptability (no gain over a plain fine-tune), checkpoint anneal (-0.02 model alone on unseen books, ties the stack there). Parked with their reasons, not dropped: a longer anneal (6.4M windows), annealing with dyn_every 4, a retrain with lr decay built in (6 h); the stack result makes them unlikely to pay. Decision rule from here: a change is adopted only if it holds on the fresh split (`corpus=fresh22 memory_corpus=fresh150`, `docs/tiers_design.md`), because the original halves are tails of books the model trained on.
- **Closed/parked: byte-class analysis of the tiers (2026-10-15).** Tools: `tier_eval classes=1` (saving by word class, `WordClasses`), `prompts=<n> generate=<len>` (cold-window pooled version, no memory), `generate=<len>` (sampling), `byte_gap=1`, `survey=1`. Findings, all in `docs/tiers_design.md`: names and rare words carry the lexicon+words saving; kNN's increment is structure, common words and the straight-quote bytes; an in-document word cache has a 0.66e-3 ceiling (closed); the word tiers add 0.0059 nats inside the full stack (0.0108 without dyn) and are kept; normalising the quote convention is worth at most 0.0076 and is not worth a retrain. Done: lexicon and words share one `WordLists` (exact); folding them into one combine rule loses 0.003 (mask kept). Parked, not dropped: inspecting which of the common-lowercase-word bytes (56% of bytes, 37% of the saving) stay expensive.
- **Closed: tier-side levers (2026-10-15).** After the byte-class analysis, no tier-side or context-length lever is left above the 0.003 bar: the remaining CE is next-word choice on common words (first letters, 45% of CE; `docs/tiers_design.md` "Full-stack check after the `WordLists` change"), and a 256-byte window gained only -0.0021 on the stack ("Context length under the tiers"). What is left is a model-side change judged on the fresh split (a retrain with lr decay built in, 6 h; the anneal evidence says at most a tie in the stack), parked above.
- **Closed: TRIME-style memory-augmented training (2026-10-15).** Built and measured on the fresh split at a 4M-window pair: the stack is 0.0070 nats/byte worse than the plain arm (1.0599 vs 1.0529), the model alone 0.021 worse. Code removed (git 284f10c). Untested variants, parked not dropped: BM25-neighbour external batches, retuned kNN weights for the ffn tap; details and caveats in `docs/tiers_design.md` "TRIME-style memory-augmented training: negative result".
- **Parked: model-side work, after TRIME (2026-10-15).** Every lever surveyed above the 0.003 bar has now been measured or closed; the stack looks near its floor for a ~2M-parameter model. What remains is the 6 h retrain with lr decay built in (anneal evidence: at most a tie in the stack) and scaling the model or corpus, which changes the project's size rather than a lever. Parked until one of those is chosen on purpose.
- **Live: model-training data (2026-10-16).** novels40 beats novels22 by 0.0186 on the fresh-split stack at 4M windows; novels90 is 0.005 worse than novels40 in the stack despite a better model alone (the memory overlap). Open: a seed repeat of 40 vs 90, 8M windows for each, and whether training on a disjoint subset of the memory books (so model and memory text do not coincide) beats both. Details: `docs/tiers_design.md` "Model-training data probe".
