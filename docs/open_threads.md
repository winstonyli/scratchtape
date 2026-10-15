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
- **A persistent, on-device GPU training backend** — the current GPU
  kernel is correct but pays a full host round-trip per call, which is
  why it doesn't win at this model's scale. Scoped as its own
  multi-day project, not a small addition. The full-block GPU
  investigation lives unmerged on `claude/vigilant-shtern-780c88` (every
  op as a gradient-checked kernel; GPU lost 1.5–4× on the real block,
  bottlenecked by per-dispatch overhead). `main`'s `gpu.rs` meanwhile kept
  `RequestAdapterOptions::default()`, which picked the integrated Radeon
  780M. Fixed 2026-09-23 to `HighPerformance` (RX 9060 XT), and the chosen
  adapter is now logged. New lead: wgpu on this machine defaults to
  Vulkan, but DX12 has lower per-call overhead on small matmuls (128×128
  round trip 1.1 ms vs 7.8 ms, ~7×; 512: 6.6 vs 30 ms, ~4.5×; about even at
  1024; an idle-GPU rerun the next day gave 1.6–5.8× across four shapes,
  `examples/gpu/gpu_dispatch_overhead.rs`), so `gpu.rs` now defaults to DX12 on Windows
  (`WGPU_BACKEND` overrides). That holds for this project's short,
  overhead-bound dispatches only: humble-cortex found Vulkan 1.4–1.6×
  faster for long compute-bound kernels, and a size sweep here was
  inconclusive at 2048+ (GPU shared with other jobs at the time). Re-measure the branch's block
  benchmark on DX12 before trusting "dispatch overhead" as a hard floor.
  Discrete-GPU matmul now: 128 → 1.1 ms (CPU 0.4 ms); 1024 → ~30 ms
  (CPU ~125 ms). The RX 9060 XT is an **eGPU over USB4** (host: Ryzen 7
  7840U laptop), so every host↔device copy crosses a ~4-lane PCIe
  tunnel: part of the per-call floor is the link, not wgpu. A GPU path
  that pays off here must keep weights and activations resident on the
  device and read back only the loss, not round-trip per op as
  `gpu.rs::matmul` and the unmerged branch do. The laptop's XDNA NPU was
  ruled out: inference-only toolchain (Ryzen AI / ONNX, quantized), no
  route for a from-scratch autograd tape, and ~10 TOPS vs the eGPU's
  ~25 TFLOPS FP32. Cheaper first lever for CPU runs: `matmul` is
  single-threaded on an 8-core/16-thread CPU.
  **Measured 2026-09-24:** `gpu_dispatch_overhead.rs` records 500 chained
  dispatches into one submit on resident buffers: **~10 µs per queued
  dispatch** (DX12 and Vulkan alike), 40–100× less than today's per-op round
  trip, and the naive kernel hits 750–900 GFLOP/s at training shapes (DX12
  ≥ Vulkan throughout). `step_profile.rs` splits a batch-8 training step
  (758 ms, CPU contended): **matmul is only 52%**, so multithreading matmul
  caps at ~1.8×; the other half is elementwise/softmax/layernorm/transpose
  work. Estimate for a device-resident step: ~2400 dispatches × 10 µs ≈
  25–30 ms plus ~2 ms of compute, vs ~130 ms for a CPU with both halves
  well parallelized, so the GPU should win ~5× at batch 8 and more with
  fusion. Not adopting cubecl for now: the hard part (a GPU-resident tape,
  buffer lifetimes, one encoder per step) is ours either way; humble-cortex
  hand-wrote every kernel and got its gains from fusion, not cubecl
  features; it would add a second wgpu version and a pre-release
  dependency against this project's build-it-yourself stance. Revisit if
  kernel boilerplate becomes the bottleneck or native HIP/CUDA matters.
  **Reopened:** cubecl also has a CPU runtime (LLVM JIT via `cubecl-llvm`,
  SIMD, alpha), so one `#[cube]` kernel source could run on the eGPU *and*
  a multithreaded CPU, covering the GPU port and the CPU fallback/eval at
  once. Kernel-language-plus-runtime (not `cubecl-matmul`) keeps the
  write-it-yourself stance. Unverified here: Windows build (bundled LLVM),
  CPU-runtime speed and threading vs `NdArray::matmul`, and launch cost vs
  raw wgpu's 10 µs. Decide with a timeboxed spike: port
  `gpu_dispatch_overhead.rs` to cubecl on wgpu-DX12/Vulkan and the CPU
  runtime. **Spike result** (`spikes/cubecl_spike/`, 0.11.0-pre.3): GPU
  parity with raw wgpu (~10 µs per queued launch, 620–900 GFLOP/s), but
  the CPU runtime has a ~3.4 ms floor per launch, even for a trivial
  elementwise kernel, and its matmul is slower than single-threaded
  `NdArray::matmul`. A ~2400-launch step would take ~8 s, so the
  one-source-for-both premise fails at this granularity for now. It also
  needed a lockfile pin (pliron version skew) and two local patches for
  Windows build bugs (`PATCHES.md`). **Tuning:** no runtime speed options.
  The CPU threadpool runs one task per unit of a cube, so units-per-cube is
  the parallelism and launch shape is the only lever. At 1 unit a launch
  costs ~1 µs but runs single-threaded. With several units, the handoff
  between threads costs 1–5 ms because idle workers park after 200 µs.
  Patching that constant to 20 ms cut it to 0.05–0.4 ms, and a 64-unit
  matmul then beat `NdArray::matmul` (0.81 vs 1.3–1.6 ms). The cost is a
  forked crate, 16 threads spinning between launches, and a per-op launch
  policy, which gains little over threading `NdArray`. Decision: raw wgpu for the GPU step,
  reusing the unmerged branch's WGSL kernels. The CPU path stays
  `NdArray` + `std::thread`. **Matrix cores, the one open question:** the
  RX 9060 XT exposes `VK_KHR_cooperative_matrix`, and cubecl built with
  `--features vulkan` (its own SPIR-V compiler) reports 12 cmma configs,
  f16×16×16→f32 among them. Its default WGSL path, on DX12 or Vulkan,
  reports none. Raw wgpu 30 only has 8×8 f32 cooperative matrices. So
  cubecl is the only route to matrix cores here. **Measured 2026-09-24**
  (cubecl 0.11.0-pre.4, Vulkan SPIR-V, idle eGPU, best of two rounds,
  ms per queued launch):

  | shape | naive f32 | comptime-k f32 | cmma f16→f32 |
  |---|---|---|---|
  | (64,64)@(64,64) | 0.015 | 0.009 | 0.007 |
  | (512,128)@(128,128) | 0.048 | 0.014 | 0.007 (2.5 TFLOP/s) |
  | (512,256)@(256,256) | 0.117 | 0.057 | 0.012 (5.6) |
  | (2048,128)@(128,128) | 0.109 | 0.060 | 0.012 (5.8) |
  | (2048,512)@(512,512) | 1.24 | 0.51 | 0.09–0.28 (up to 12) |

  Matrix cores are 5–14× faster than the naive kernel. At step-sized
  shapes they reach the ~7–12 µs launch floor, so the matmul share of a
  GPU step all but disappears. The f16 error is at most 1.2e-4 against
  the f32 CPU product. Making k a compile-time constant alone gives the
  naive kernel 2–3×. A fresh output buffer per launch costs nothing
  measurable, because cubecl pools its memory.
  **A fairer f32 baseline closes most of the gap.** The tiled f32 kernel
  (`k_matmul_tiled`: a 64×64 tile per group, 4×4 per thread, staged
  through shared memory) runs at 0.97–4.7 TFLOP/s. That's 1.1–5.5×
  faster than the naive kernel. Matrix cores keep a **2–3× edge** over it
  (for example 0.017 vs 0.008 ms at (512,128)@(128,128), and 0.23 vs
  0.10 ms at (2048,512)@(512,512)). These are best of two rounds with
  another GPU job sharing the eGPU. The tiled kernel is simple, with no
  vector loads and no double buffering, so 2–3× is an upper bound on
  what matrix cores add.
  Caveats:
  - Inputs were converted to f16 ahead of time. A real step also needs
    cast kernels, or f16 copies kept alongside the f32 weights.
  - f16 matmul inputs in training need a precision check: gradient
    check tolerances, and perhaps loss scaling.
  - The largest shape varied 3× between rounds.
  **This reopens the decision.** cubecl matched raw wgpu on launch cost,
  and it's the only route to matrix cores. Matrix cores' real margin is
  2–3× over a decent f32 kernel, not 5–14×. At step shapes both kernels
  sit near the ~10 µs launch floor, so the margin matters less than
  dispatch count. Still, cubecl gives compile-time specialization and a
  matrix-core option that raw WGSL can't, so it's the better base for
  the GPU step. **Decided 2026-09-24: cubecl on Vulkan SPIR-V**, tiled
  f32 matmul by default, matrix cores kept as an option.
  **Launch count is the lever.** The `step_profile 8 1 census` example
  counts what a device-resident step would launch:

  | layout | unfused | elementwise-fused | ms at ~10 µs/launch |
  |---|---|---|---|
  | 8 heads as separate ops (today) | 2372 | 1281 | 23.7 / 12.8 |
  | heads batched into one op | 776 | 385 | 7.8 / 3.9 |

  The current tape runs each attention head as its own ops: 173 matmuls,
  against 33 with the heads batched, and the softmax ops repeat per head
  too. So the GPU step should batch heads from the start, a 3× cut, and
  fuse elementwise chains for another 2×. The census model counts one
  launch per gradient sent and per broadcast reduce, and treats an
  elementwise chain as a single kernel. Kernel time comes on top.
  **Design:** [`docs/gpu_step_design.md`](docs/gpu_step_design.md). It
  uses a small device tape of coarse, layer-level ops (fused QKV,
  attention batched over batch×heads, matmuls with bias, ReLU or residual
  folded in, fused softmax+cross-entropy), about 145 launches per step.
  Parameters live in one flat buffer, so the update is a single launch.
  The design has six milestones with parity checks against the CPU tape,
  and a kill criterion: stop if the step isn't clearly faster than a
  well-threaded CPU step.
  **Milestone 1 done:** `src/gpu_step/` sits in the main crate, with
  cubecl as a plain dependency rather than a feature. It selects the
  discrete GPU on Vulkan and logs it, with no fallback. It keeps the
  parameters and gradients in flat device buffers, packed by
  `TransformerBlock::to_flat`, and runs SGD and gradient zeroing as one
  launch each. Its GPU check tests SGD against `optim::Sgd`
  (`cargo test --lib gpu_step -- --ignored`).
  **The CPU model now uses batched heads (2026-09-24).** Each block
  stores one fused QKV `Linear`. New `split_heads`/`merge_heads` tape
  ops turn attention into one `batched_matmul` over batch × heads, and
  the per-head extras (QK-norm, sinks) are gathered from `[H, ·]`
  tables. `batched_heads_match_per_head_reference` checks it against
  outputs captured from the old per-head code. The forward is
  bit-identical, and gradients agree to 1e-4 relative (f32
  reassociation). A 200-step training curve tracks the old one. The step
  has 280 tape nodes instead of 888 and trains ~1.35× faster; the
  survey had bounded the gain at ≤1.8×. Old checkpoints load but are
  wrong, since the length is the same and the order isn't; the ones in
  `runs/` were converted once. Details are in the design note.
  **Milestone 2 done (2026-09-24):** every op in the design's table has
  forward and backward kernels in `src/gpu_step/`:
  - one generalized tiled matmul (`matmul`): transposes, batching,
    offsets into the flat buffers, and a bias/ReLU/mask/residual/accumulate
    epilogue;
  - LayerNorm and Softmax (`rows`);
  - Embed and CrossEntropy (`tokens`);
  - split/merge heads (`heads`; later replaced by head views in the
    matmul, see below).
  Each is checked against the CPU tape (`cargo test --lib gpu_step --
  --ignored`). The backward kernels are also checked by finite
  differences through the GPU forward, and deliberate mutations fail the
  tests. One intended difference: CrossEntropy uses the exact gradient,
  while the tape's −log(p + 1e-9) is off by 1e-9/p_t (1.2e-4 in the test;
  ~3e-7 at the real step's scale).
  **Milestones 3 and 4 done (2026-09-24):** `gpu_step::tape::DeviceTape`
  records coarse ops and backpropagates through them on the device. One
  full tiny_lm step (batch 8, 4 blocks, plain softmax and softmax1)
  matches the CPU tape to 1e-4: every block's output, the logits, the
  loss, and every parameter gradient. The worst gradient error measured
  was 1.7e-5 (`device_step_matches_cpu_tape`).
  **Milestone 5 done (2026-09-24): the device step trains like the CPU
  and is ~27× faster.**
  - **Training parity.** `gpu_train_check` runs 200 steps of
    training_recipe_check's recipe on both tapes in lockstep. Final
    train-probe / held-out CE:
    - plain: GPU 2.6892 / 2.8332 vs CPU 2.6914 / 2.8318;
    - softmax1: GPU 2.6428 / 2.7863 vs CPU 2.6551 / 2.7888.
    Both gaps are the size a 1e-6 nudge to one CPU weight produces.
  - **Why the runs drift apart.** The two runs agree at step 0, then
    drift: at real size, about once a step one of ~524k FFN
    pre-activations falls within rounding of 0, and the two ReLUs
    disagree on it. From then on the runs are chaotic twins, not a bug.
  - **Timing (idle eGPU).** A GPU step takes 9.4–9.6 ms best, 175
    launches, vs 259–292 ms for the single-threaded CPU tape. That is
    ~14× under the 130 ms kill criterion.
  - **Contention.** With other jobs on the eGPU, steps took 0.2–25 s.
  - **Where the time goes.** Device-timestamp profiling
    (`gpu_train_check 0 41 profile`) puts 9.1 of the 9.5 ms in kernels,
    not launch overhead. Matmul takes 3.0 ms. The row-wise kernels
    (LayerNorm, the bias column sums, softmax) take 5.7 ms: they run one
    serial unit per row, so 512 rows fill 2 cubes and every load is
    uncoalesced.
  - **Row reductions parallelized (2026-09-24).** One cube per row (or
    per 16 columns) with a fixed-order shared-memory tree: the step went
    from 9.4 to **4.6 ms best**, and the row kernels from 5.7 to 0.43 ms.
    Results stay deterministic.
  - **Matmul per shape.** The weight-gradient matmuls (Xᵀ·dY, k = 512)
    now take ~44% of the step: their small outputs make only 4–12 cubes
    of 64×64 on 32 compute units.
  - **Split-k (2026-09-24).** Such matmuls now split k into slices that
    run in parallel, then sum the partials in a fixed order (still
    deterministic). Step: **~3.3 ms best** (was 4.6); kernel time
    2.36 ms, so launch overhead (209 launches) is now ~1 ms of the step.
  - **Head views (2026-09-24).** The matmul now reads Q, K, V straight
    out of the fused QKV buffer and writes attention heads straight into
    the merged layout, removing the split/merge-heads kernels: 177
    launches, kernel time 2.16 ms. **The wall step stayed ~3.35 ms**, and
    pipelining steps (no per-step readback) doesn't help either, so the
    remaining ~1.2 ms gap is host- or driver-side serialization, not
    per-launch GPU cost. Finding it is the next step. (Corrected below:
    that run's host was slowed by other jobs' CPU load.)
  - **Host side and backends (2026-09-25).** With a free CPU the step is
    bounded by host queueing and the per-step loss readback (~1.3 ms over
    USB4). Pipelined (loss read every N steps), a step takes **~1.5 ms on
    DX12 and ~2.0 ms on Vulkan**, ~150× the CPU tape. `gpu_step` now
    submits every 128 launches (cubecl's default of 32 cost ~60 µs of
    host time per submit), and `WGPU_BACKEND=dx12` selects DX12 but
    requires `dxcompiler.dll` on PATH: without DXC, wgpu silently uses
    FXC and the step runs ~170× slower. Vulkan stays the default.
  - **A full training run on the GPU (2026-09-25).**
    `training_recipe_check <name> <softmax1> 8 0.3 64000 600 gpu` trains
    all 8000 steps and evaluates on the device, and reads the parameters
    back only for checkpoints. Uncontended, a step takes ~1.9–2.0 ms (the
    first 1000 include kernel compilation) and an evaluation ~0.3 s, so
    a run is ~20 s. Slower stretches (2.6–18 ms) were other sessions'
    eGPU jobs starting mid-run. Final CE matches the CPU runs to the
    chaotic-twin noise (plain held-out 1.857 vs 1.858). Reruns and
    kill/resume reproduce the checkpoint byte for byte.
  Details: `docs/gpu_step_design.md`, milestones 5 and 6.
  The CPU tape stays a separate reference rather than
  running the GPU's cubecl kernels on the CPU. A single cubecl source for
  both was tested and ruled out (2026-09-24, `cubecl_spike cpu-step`):
  - The GPU's tiled matmul never finishes on cubecl's CPU runtime, which
    runs one spinning OS thread per unit at each barrier.
  - The barrier-free kernel runs a step-shaped 144-launch chain in
    157–165 ms on 12 cores, against 59–67 ms for single-threaded
    `NdArray`.
  - Launch cost isn't the limit: the `IDLE_POLL` patch doesn't change
    the chain time. Upstream #1658 makes it 4.6× slower on Windows. JIT
    is ≤0.2 s per kernel.

  A survey of cubecl's CPU launch overhead also found wgpu graph capture
  in pre.4, a candidate for replaying a whole GPU step.
  **One wgpu (2026-09-24):** `gpu.rs` and `gpu_dispatch_overhead.rs`
  moved from wgpu 23 to 30, so the crate builds one wgpu stack instead
  of two. The lockfile shrank by ~530 lines. wgpu 30's deeper types
  tripped a future-incompat recursion-limit warning on `gpu.rs`'s
  `OnceLock`, so `lib.rs` sets `recursion_limit = "256"`. All 25 lib
  tests pass, including both GPU ones.
  **The eGPU "hangs" are contention, and the utilization counter can't
  show it.** Re-timing `gpu_dispatch_overhead` stalled inside
  `device.poll` after 19–287 round trips, on DX12 and Vulkan alike. The
  wgpu 23 build at `122243f` stalled the same way (after 31), so the port
  isn't the cause. Five other processes had work on the eGPU: three
  humble-cortex `conv_pc2_isolated_*` checks, sojourn and manifold.
  Windows' `GPU Engine … Utilization Percentage` counter reports millions
  of percent for this eGPU, so an idle check filtered to 0–100% reads
  "idle" while it's saturated. A usable check is to list which pids have
  3D-engine instances on the eGPU's LUID (`0x16290` this boot) and look
  them up, ignoring the values. The spike's earlier "apparent hang" was
  probably this too. Re-timing the example is deferred until the eGPU is
  actually free.
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
