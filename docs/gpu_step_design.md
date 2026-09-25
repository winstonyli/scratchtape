# GPU training step: design (2026-09-24)

Goal: run one tiny_lm training step (batch 8, `d_model` 128, 8 heads,
`seq_len` 64, 4 blocks, `d_ff` 256, vocab 256, SGD, cross-entropy)
entirely on the RX 9060 XT eGPU. It must match the CPU tape's losses and
gradients, and be faster than the CPU step. Status: **milestone 1 done
(2026-09-24)**. A single cubecl source for CPU and GPU was tested and
ruled out the same day (experiment below). The CPU model moved to
batched heads the same day (section below). **Milestone 2 (kernels)
done the same day, and so were milestones 3 and 4 (forward and gradient
parity with the CPU tape).** **Milestone 5 done the same day: the
device step trains like the CPU tape and takes ~9.5 ms on an idle eGPU,
~14× under the kill criterion.** Parallel row reductions then halved it
to ~4.6 ms (milestone 5, "Row reductions"), and split-k matmuls to
~3.3 ms (milestone 5, "Split-k"). Head views then cut 32 launches and
0.3 ms of kernel time but left the wall step at ~3.35 ms: the remaining
floor is host- or driver-side (milestone 5, "Head views"). Next: find
that floor, then milestone 6 (optional).

## What the evidence says

- **Launch cost sets the floor.** A queued cubecl launch on resident
  buffers costs ~10 µs (`spikes/cubecl_spike`). Round trips cost
  0.4–2.4 ms, so the step does one sync, not one per op.
- **Launch count is the lever** (`step_profile 8 1 census`):

  | layout | launches |
  |---|---|
  | today's tape, heads as separate ops | ~2370 |
  | heads batched | ~780 |
  | heads batched + generic elementwise fusion | ~385 |

- **Kernel speed matters less.** A tiled f32 matmul runs 0.97–4.7 TFLOP/s.
  Matrix cores (cmma) add 2–3×. At step shapes both are near the launch
  floor.
- **The earlier attempt** (`origin/claude/vigilant-shtern-780c88`, raw
  wgpu, one kernel per tape op, heads as separate ops) lost 1.5–4× to the
  CPU. The cost was building bind groups and encoding each op on the CPU
  side. It ran before the adapter fix, so likely on the 780M iGPU. It
  named batched heads and cubecl as the untried levers, and this design
  uses both.
- **CPU reference:** 94 ms per step at batch 1, 758 ms at batch 8 under
  contention, and an estimated ~130 ms for a well-threaded batch-8 step.

## Decision: coarse ops on a small device tape

Three shapes were considered:

| option | launches per step | autograd |
|---|---|---|
| A. The current `Tape` with device storage | ~2370 (~385 with fusion) | reused |
| B. A fixed model step with hand-written backward (llm.c style) | ~140 | none, duplicated by hand |
| **C. A device tape of coarse, layer-level ops** | **~140** | **kept** |

C keeps the project's idea, where the tape records ops and backward walks
it in reverse. Each op is a fused layer-level primitive with a
hand-written, gradient-checked backward. That gives B's launch count
without giving up autograd.

Ops, each with forward and backward kernels:

| op | forward | backward |
|---|---|---|
| `Embed` (token + position, summed) | 1 | 2 (one deterministic per-row reduction per table, no atomics) |
| `LayerNorm` | 1 | 2 (dx, then a dgamma/dbeta column reduction) |
| `Linear` with epilogue (bias, optional ReLU, optional residual add) | 1 | 3 (dX, dW, db; the ReLU mask applied inside dX) |
| `QKV` (one fused `[D, 3D]` projection, written straight into `[B, H, T, d_k]`) | 1 | 3 |
| `AttnScores` (batched over B·H: QKᵀ, scale, causal mask) | 1 | 2 (dQ, dK) |
| `Softmax` (row-wise; plain or softmax1 by comptime flag) | 1 | 1 |
| `AttnOut` (batched weights@V, stored merged as `[B·T, D]`) | 1 | 2 (dWeights, dV) |
| `CrossEntropy` (fused softmax + NLL) | 1 | 1 |

That's about 9 forward and 22 backward launches per block. Across 4
blocks plus embedding, the final LayerNorm, the output projection and the
loss, a step is about 140 launches. With gradient zeroing, one SGD launch
and one loss readback, it's about 145. That's ~1.5 ms of launch floor,
against ~13 ms for option A with fusion.

## Data layout

- **Parameters** live in one flat f32 device buffer, and the gradients in
  another with the same offsets. SGD is one launch over the whole buffer,
  and zeroing the gradients is one more. Q, K and V are stored fused as
  `[D, 3D]`. Pack and unpack functions map to and from the CPU model's
  per-head `Linear`s, for parity tests and checkpoints.
- **Saved activations** come from cubecl's memory pool, allocated fresh
  each step. The spike measured fresh allocation as free.
- **Per step**, the host uploads token ids and targets (2 × 512 u32) and
  reads back one f32 loss. Nothing else crosses USB4.

## Kernels and numerics

- Kernels are written in cubecl's Rust DSL. They're hand-written; nothing
  is auto-fused, which keeps the project's "write the primitive yourself"
  stance.
- Matmul: the tiled f32 kernel from the spike, generalized to NN, NT and
  TN, with comptime shapes and an epilogue. Shapes that aren't multiples
  of the tile get guarded loads.
- f32 throughout for v1. The cmma f16 matmul comes later behind a flag,
  after a precision pass (gradient-check tolerances, loss-curve match).
- Keep per-thread register arrays small (the tiled kernel uses 80 B)
  because of cubecl #1336: private arrays race above ~544 B on Vulkan
  SPIR-V.

## Backend and code placement

- **Backend:** cubecl `=0.11.0-pre.4` with the `vulkan` (SPIR-V) feature,
  the only route to matrix cores. The adapter is picked by name (discrete
  RX 9060 XT) and logged at startup. The full step is timed once on DX12
  as well, since raw wgpu had DX12 ahead on launch overhead.
- **Placement (decided 2026-09-24):** `src/gpu_step/` in the main crate,
  with cubecl as a plain dependency, not behind a feature. The cost: every
  build compiles cubecl; a cold `cargo check --all-targets` took ~4 min.
  `gpu.rs` was moved from wgpu 23 to 30 so there's only one wgpu. The CPU runtime
  (`cubecl/cpu`) is not used, so no LLVM download.
- **Device:** `WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0))` with
  `init_setup::<Vulkan>`. cubecl panics if there's no discrete adapter,
  and its `CUBECL_WGPU_DEFAULT_DEVICE` override doesn't apply to this
  kind, so there's no silent fallback. The adapter is logged once:
  `AMD Radeon RX 9060 XT (DiscreteGpu, Vulkan, driver 26.8.1 (LLPC))`.
- **Lockfile gotcha:** adding cubecl to an existing lock resolved
  `gpu-allocator 0.28` against the already-locked `windows 0.58`, and
  `wgpu-hal 30` failed to compile (D3D12 trait mismatches). Fix:
  `cargo update -p gpu-allocator@0.28.0`, which moves it to `windows 0.62`.

## Milestones and checks

Each milestone leaves a runnable check behind.

1. **Scaffolding. Done.** Device selection and logging, the flat
   parameter and gradient buffers (`DeviceParams`), and one-launch SGD
   and gradient zeroing.
   - `gpu_step::pack` flattens a whole model with `TransformerBlock::to_flat`,
     which is already the fused-QKV layout since batched heads. A
     whole-model unpack waits for milestone 5, the first thing that
     needs one.
   - Checks: `batched_heads_match_per_head_reference` (CPU, runs by
     default; it replaced the per-head-to-fused layout test) and
     `device_sgd_matches_optim_sgd` (GPU,
     `cargo test --lib gpu_step -- --ignored`, since it needs the
     discrete GPU).
2. **Kernels one by one.** Each forward is compared with the CPU tape's
   composed ops on the same inputs. Each backward is checked two ways:
   against CPU gradients, and by finite differences through the GPU
   kernels. That's the branch's discipline.
   - **Matmul: done (2026-09-24).** `gpu_step::matmul` is the one matmul
     every coarse op uses: the spike's 64×64-tile f32 kernel, generalized
     to NN/NT/TN by comptime flag, any m/n/k (guarded loads), batched
     over the grid's z with per-operand strides and element offsets (the
     parameters live in one flat buffer), and an epilogue of + bias,
     ReLU, + residual and accumulate-into-out. Check:
     `matmul_matches_reference` (ignored; needs the GPU) covers every
     transpose pair, ragged shapes, batch strides, offsets, each epilogue
     and the step's own shapes, to 1e-5 of the output scale. A
     deliberately broken NT index fails it. It's linear, so a
     finite-difference check would add nothing over the reference.
   - Caveats for milestone 5: the tile is 64×64 even when n is 16
     (AttnOut), wasting 3/4 of that launch, and QKV writes row-major
     `[B·T, 3D]`, so a permute into `[B·H·T, d_k]` is a separate launch
     for now.
   - **Row-wise: done (2026-09-24).** `gpu_step::rows`: LayerNorm
     forward (saves mean and 1/std) and backward (dx written or
     accumulated; dgamma/dbeta accumulated), Softmax with the score scale
     and causal mask folded in (plain or softmax1 by comptime flag) and
     its backward, and the column sum for bias gradients. One unit per
     row or column, fixed summation order, so deterministic. Checks
     (ignored; GPU): `layer_norm_matches_cpu_tape` and
     `softmax_matches_cpu_tape` compare with the CPU tape's composed ops
     (forward 1e-5, gradients 1e-4 relative) and check the backward by
     central differences through the GPU forward. softmax1 is also
     checked at 40× logits, where its max(row max, 0) shift matters.
     Mutations (dropping softmax1's phantom term, or LayerNorm dx's
     xhat term) fail them. `col_sum_accumulates_bias_gradient` covers
     the bias reduction.
   - One unit per row is the simplest correct shape, not the fastest:
     512 rows is 2 cubes of 256. Milestone 5 decides whether it matters.
   - **Embed, CrossEntropy, heads, ReLU mask: done (2026-09-24).**
     `gpu_step::tokens` has Embed (token + position; backward is one
     ordered reduction per table, no atomics) and CrossEntropy (per-row
     logsumexp, then one unit sums the mean; backward is the closed form
     (softmax − onehot)/rows). `gpu_step::heads` had split/merge heads,
     each the other's backward; merge writes or accumulates at a column
     offset, so dQ/dK/dV merge straight into dQKV. (Removed later the same
     day: the matmul addresses heads in place; milestone 5, "Head
     views".) The matmul epilogue
     gained the ReLU backward mask. Checks (ignored; GPU):
     `embed_matches_cpu_tape` (forward exact, repeated and unused ids),
     `cross_entropy_matches_cpu_tape` (plus finite differences),
     `split_and_merge_match_tape` (exact), and new mask cases in
     `matmul_matches_reference`.
   - **CrossEntropy differs from the tape on purpose.** The tape computes
     −log(p + 1e-9), which scales its gradient by p_t/(p_t + 1e-9). The
     GPU uses the exact form. In the test (p_t down to 8.3e-6) the tape
     sits 1.2e-4 from an f64 closed form and the GPU 2.8e-7; the gap is
     1e-9/p_t exactly. At the step's scale (p ≈ 1/256 at init) it's ~3e-7,
     so it won't show in milestones 3–5 unless a model gets very
     confident and very wrong.
   - **Milestone 2 is complete:** every op in the ops table has its
     forward and backward kernels, each checked against the CPU tape.
   - Timing note: with 5 other processes on the eGPU (2026-09-24), the
     8 GPU tests took 275 s instead of ~10 s; the finite-difference loops'
     readbacks queue behind the other jobs.
3. **Forward parity. Done (2026-09-24).** A block's output and the loss match
   `forward_full` (plain and softmax1) to 1e-4 relative.
4. **Gradient parity. Done (2026-09-24).** Every parameter gradient after one step matches
   the CPU tape to 1e-4 relative.
   - `gpu_step::tape::DeviceTape` is option C as designed: it records
     coarse ops (Embed, LayerNorm, Linear with bias/ReLU/residual,
     split/merge heads (since replaced by head views, milestone 5),
     batched matmul, causal softmax, CrossEntropy)
     with their device values, and `backward` walks them in reverse,
     calling the milestone-2 kernels. Parameter gradients accumulate
     into the flat buffer. Activation gradients are allocated on first
     contribution. A residual aliases its gradient buffer instead of
     copying it, which is safe because in reverse order every other
     reader of that buffer has already run. `Config` gives each
     parameter tensor's offset in `pack`'s layout (checked against
     `pack`'s length). `block_forward` and `model_forward` build a plain
     tiny_lm; the extras stay out of scope.
   - Check (ignored; GPU): `device_step_matches_cpu_tape` runs one step's
     forward and backward on both tapes from the same init and batch:
     small ragged shapes and the real step (batch 8, d 128, 8 heads, T 64,
     4 blocks), each with plain softmax and softmax1. Every block output,
     the logits and the loss match to 1e-4. So does every parameter
     tensor's gradient, relative to its own scale. Worst measured
     gradient error: 1.3e-5 (plain) and 1.7e-5 (softmax1), both on the
     token table at real size; ~1.5e-6 at the small size. Dropping the
     transpose in AttnScores' dK path fails it (0.77).
   - The forward records 56 device ops at real size. The ReLU backward
     is its own launch for now; the matmul's mask epilogue could fold it
     into ffn2's dX in milestone 5.
5. **Training parity and timing.**
   - 200 steps from the same init and batches should track the CPU loss
     curve.
   - Step time is timed against the CPU step, best-of-N if the eGPU is
     shared.
   - Launches per step are counted.
   - Utilization is checked on the discrete GPU.
   - Timing runs hold an exclusive GPU lease (`gpu_lease::hold`). Long
     training runs hold a shared lease and call
     `gpu_lease::pause_while_exclusive()` at each checkpoint.
   - GPU runs go at Normal CPU priority, not BelowNormal: with every
     core busy, BelowNormal made round trips ~25× slower
     (`gpu_cpu_priority_check.rs`).
   - **Done (2026-09-24).** `gpu_train_check` runs both tapes in
     lockstep on the same batches (training_recipe_check's setup: seed 1,
     batch 8, lr 0.3, 200 steps).
   - **Exact tracking stops at real size, and neither tape is wrong.**
     Step 0 matches (loss difference 9.5e-7). By step 20 the losses
     differ by 0.56. The cause is ReLU kinks: the real step has ~524k FFN
     pre-activations, and about once a step one lands within float
     rounding of 0, so the two tapes' ReLUs disagree. Seed 13, step 1:
     one unit, block 3 row 127, is +7.0e-7 on the CPU and ≤ 0 on the
     GPU. That flips the whole gradient of that unit. The embedding
     gradient error sits on the same row, and the parameters then differ
     by ~1e-4. From step 2, dozens of units flip each step. The GPU
     backward is deterministic (three reruns, bit-identical). The small
     test config has ~600× fewer pre-activations and stays at 7e-7 for
     6 steps. So `device_training_tracks_cpu_over_steps` checks the
     small config only, and at real size the check is the training
     outcome.
   - **Yardstick: CPU vs a nudged CPU** (`gpu_train_check <s> 200
     control [index]`). This is the CPU tape again, from an init with
     one weight moved by 1e-6. The first attempt nudged token 0, which
     never occurs in the corpus, so it changed nothing. Nudging a block
     weight gives a loss gap of 0.38 at step 40, the same shape as the
     GPU's 0.81.

     | final CE (train-probe / held-out) | CPU | GPU | CPU, nudged 1e-6 |
     |---|---|---|---|
     | plain | 2.6914 / 2.8318 | 2.6892 / 2.8332 | 2.6988 / 2.8400 (1 site) |
     | softmax1 | 2.6551 / 2.7888 | 2.6428 / 2.7863 | 2.6436–2.6657 / 2.7871–2.7961 (4 sites) |

     The GPU's gaps (≤ 0.0022 plain; 0.0123 / 0.0025 softmax1) are the
     size of a 1e-6 nudge's. The softmax1 train-probe gap is at the edge
     of the four nudges' range (−0.0115 to +0.0106).
   - **Timing (idle eGPU, exclusive lease, 100 steps back to back,
     `gpu_train_check <s> 101 time`):**
     - GPU step: 9.60 ms best / 10.18 ms median (plain), 9.38 / 10.05
       (softmax1).
     - Launches: 175 per step.
     - CPU step, in the same runs: 259 / 292 ms best (single-threaded
       tape).
     - The kill criterion (~130 ms) is beaten ~14×.
     - The GPU step is still 2–5× over the 2–5 ms estimate. The launch
       floor is only ~1.75 ms (175 × ~10 µs).
   - **Contention dominates any timing.** With 5 other jobs on the eGPU,
     the lockstep runs' GPU steps took 1–25 s. With one other job, back
     to back, they took ~208 ms. Idle, they took ~9.5 ms. Any GPU time
     measured without first checking the eGPU's process list is
     meaningless.
   - **Where the time goes: kernels, not launches**
     (`gpu_train_check 0 41 profile`, idle eGPU). gpu_step's profiler
     gives each launch its own cubecl device-timestamp window and charges
     it to the launching line (`#[track_caller]` on `count_launch`).
     Summed kernel time is 9.09 ms per step, against a 9.5 ms unprofiled
     step. A first idle run read 12.3 ms, probably clocks ramping.

     | launch site | ms/step | launches | share |
     |---|---|---|---|
     | matmul (all) | 3.02 | 75 | 33% |
     | `col_sum` (bias grads) | 1.87 | 17 | 21% |
     | LayerNorm backward dx | 1.21 | 9 | 13% |
     | LayerNorm backward gamma/beta | 1.06 | 9 | 12% |
     | LayerNorm forward | 0.89 | 9 | 10% |
     | Softmax backward | 0.50 | 4 | 6% |
     | Softmax forward | 0.15 | 4 | 2% |
     | everything else (68 launches) | ~0.4 | 68 | 4% |

     The row kernels take 5.7 ms (62%). The cause is the one-unit-per-row
     (or per-column) design, which serially loops over d = 128 (or 512
     rows) three times:
     - 512 rows is 2 cubes of 256, on a GPU with 32 compute units.
     - Neighbouring units read addresses d floats apart, so no load is
       coalesced.
     - Each launch takes ~100–130 µs.
     - Softmax, with 2048 rows of 64, is the cheapest of them per launch
       (38 µs).
     This is the caveat left open in milestone 2 ("Milestone 5 decides
     whether it matters"): it does.
     Fix: a cube per row (or per group of columns) with a fixed-order
     shared-memory tree reduction, which keeps results deterministic.
   - **Row reductions (done 2026-09-24).** `rows.rs` now runs:
     - one 64-unit cube per row for LayerNorm forward, LayerNorm dx and
       both softmaxes, each unit striding the row by 64 (coalesced), then
       a 6-level shared-memory tree (`row_reduce`);
     - 16 columns × 16 lanes per cube for the column sums (bias grads,
       gamma/beta), each lane striding rows by 16, then a 4-level tree
       across lanes (`lane_reduce`).
     The summation order is fixed, so runs stay bit-reproducible, but it
     differs from the CPU's serial order. Mutation check: dropping the
     last tree level fails every row test.

     Results, idle eGPU: a step takes **4.62–4.67 ms best** (5.8 ms
     median) for both plain and softmax1, down from 9.4–9.6 ms. Summed
     kernel time is 3.47 ms, of which the row kernels are now 0.43 ms
     (was 5.7). Launch count is unchanged (175).

     Gotchas:
     - **cubecl SPIR-V validation** ("Expected operand type cube.ptr, but
       found cube.index") came from the first version of the tree helpers.
       What compiled: u32 indices cast `as usize` at each access, literal
       shared sizes (`64usize`), `#[unroll]` comptime loops instead of a
       `while` over usize, and helpers that take a value and allocate
       their own `Shared` rather than receiving one. The exact trigger
       wasn't bisected.
     - **The one-step parity test hit a ReLU tie.** The new reduction
       order moved one FFN pre-activation (of ~524k at real size) to
       within rounding of 0 on seed 12, flipping a hidden unit's gradient.
       `device_step_matches_cpu_tape` now compares the post-ReLU hidden
       values first, requires every disagreement to be a tie (|z| < 1e-5),
       and moves to the next seed if there is one. Plain real size uses
       seed 13 (worst gradient error 8.1e-6); the rest pass at seed 12.
   - **Matmul per shape** (`gpu_train_check 0 41 profile`; the profiler
     tags each matmul launch with batch, shape, transposes and epilogue).
     Matmul is now ~3.0–3.4 ms of the step. The idle run's top lines were
     lost; the table is from a run where another job joined near the end
     (total 4.26 ms), so read the ranking, not the absolute values:

     | shape (out = a·b) | role | launches | µs each | cubes |
     |---|---|---|---|---|
     | [128×512]ᵀ·[512×256], += | dW, FFN1 / output proj | 5 | 108 | 8 |
     | [128×512]ᵀ·[512×384], += | dW, QKV | 4 | 115 | 12 |
     | [128×512]ᵀ·[512×128], += | dW, attn out | 4 | 114 | 4 |
     | [256×512]ᵀ·[512×128], += | dW, FFN2 | 4 | 110 | 8 |
     | [512×384]·[384×128]ᵀ | dX, QKV | 4 | 59 | 16 |
     | [512×256]·[256×128]ᵀ | dX, FFN1 / output proj | 5 | 40 | 16 |
     | [512×128]·[128×256 or 128]ᵀ | dX, FFN2 / attn out | 8 | 22 | 16 |
     | 64×[64×64]·[64×16] (two), 64×[64×16]·[16×64]ᵀ | attention, per head | 24 | 9–20 | 64 |
     | forward Linears | | 17 | 26–49 | 16–48 |

     **The thin shapes that starve are the weight gradients, not the
     attention matmuls.** Each is Xᵀ·dY with k = 512 (the batch's rows)
     and a 128–256 × 128–384 output, so 64×64 tiles give only 4–12 cubes
     for 32 compute units, and each cube walks all 512 of k serially. The
     four dW shapes take ~1.9 ms, 44% of the step. The per-head attention
     matmuls do fill 64 cubes but use a quarter of each 64×64 tile
     (n = 16); they total ~0.4 ms.

     Next fix: split-k for the dW matmuls. Partition k = 512 into s
     slices across the grid's z, write partials to scratch, then sum them
     in a fixed order into the gradient (keeps determinism; one extra
     launch per dW). s = 4–8 gives 16–96 cubes. Smaller tiles (32×32) are
     the alternative but shorten each cube's reuse; measure both.
   - **Split-k (done 2026-09-24).** `matmul` splits k itself when a call
     has batch 1, no epilogue but `+=`, and fewer than `SPLIT_TARGET` = 64
     output tiles: slices = ceil(64 / tiles), capped so each slice keeps
     ≥ 64 of k. Each slice is one z of the grid and writes its own [m, n]
     partial to a scratch buffer; `k_split_sum` then adds the partials in
     ascending order into out. Results stay deterministic, at one extra
     launch per split call (175 → 209 launches). The dW matmuls split
     6–8 ways; the unmasked dX matmuls (512 × 128 outputs, 16 tiles) split
     2–4 ways, which the rule picked up on its own.

     Target sweep, plain, 101 steps, idle eGPU, best of two rounds
     (best ms per round):

     | SPLIT_TARGET | round 1 | round 2 |
     |---|---|---|
     | off | 4.61 | 4.73 |
     | 32 | 3.69 | 3.82 |
     | **64** | **3.25** | 3.53 |
     | 128 | 3.57 | 3.49 |

     64 and 128 are within noise of each other; 64 needs half the
     scratch. Profile at 64 (idle): kernel time 3.47 → **2.36 ms**. The
     four dW shapes went from ~1.9 ms to 0.32 ms, plus 0.14 ms of sums.
     The wall step (~3.3 ms) is now ~1 ms above kernel time, so launch
     and submit overhead matters again: with 209 launches, the DX12
     comparison (milestone 6) and fusing the split sums into a following
     kernel are the next levers (but see "Head views": the gap turned out
     not to be per-launch). Checks: new matmul test cases (split 6,
     split 8, and split 4 with a ragged last slice); dropping the first
     partial from the sum fails the test (error 50 at scale 109).
   - **Head views (done 2026-09-24).** `MatRef` gained a row stride
     (`ld`) and a two-level batch offset, matrix z at off + (z / group)·
     stride + (z % group)·inner, so `MatRef::heads` addresses head h of
     batch row b in place in a [B·T, cols] buffer. The tape's
     `batched_matmul` takes a `Layout` (Stacked or Heads) per operand and
     for its output: scores read Q and K straight from the QKV output,
     the attention output is written straight into the merged [B·T, D]
     layout, and backward writes dQ, dK, dV into the QKV gradient (zeroed
     once, then accumulated). SplitHeads/MergeHeads and `gpu_step::heads`
     are gone; a block is 9 tape ops, not 13. Checks:
     `matmul_head_views_match_reference` (Q Kᵀ, P V into the merged
     layout with +=, Pᵀ dCtx into V's columns); dropping the in-group
     offset fails it (error 1.16); full-step parity passes unchanged.

     Result: launches 209 → 177 and kernel time 2.45 → **2.16 ms**
     (profiles back to back, idle). **The wall step didn't move.**
     Interleaved A/B, 101 plain steps, idle-gated (medians were spoiled
     by bursty other users the 2 s idle check misses, so best-of):

     | | synced best, 4 runs | pipelined best |
     |---|---|---|
     | before (2b4034a) | 3.37 / 3.53 / 3.53 / 3.61 | 3.43 |
     | head views | 3.38 / 3.41 / 3.47 / 3.74 | 3.36 |

     **Negative result: the ~1 ms between wall step and kernel time is
     not per-dispatch GPU cost.** Cutting 32 dispatches and 0.3 ms of
     kernels left the floor at ~3.35 ms, and the gap grew to ~1.2 ms. It
     isn't the loss readback either: `gpu_train_check ... time` now also
     runs the steps pipelined (one readback at the end), and that is no
     faster than syncing every step (3.31 vs 3.38 ms). The host spends
     ~1.7 ms queueing a step (~10 µs per launch). Pipelining should then
     give max(host, GPU) ≈ 2.2 ms, not 3.35, so host queueing and GPU
     execution are serialized somewhere (cubecl's submission or
     allocation path, or the driver). Next: one device-timestamp window
     around a whole step (GPU span vs wall), then a host-side look at
     where the 1.7 ms of queueing goes.
   - **Negative result: a sync-after-every-launch profiler didn't work.**
     Its step ran at ~70–100 ms. The ~0.5 ms round trip charged to each
     launch swamped the kernels, so it couldn't rank them. Device
     timestamps replaced it. With them, a profiled step takes 50–60 ms of
     wall time, because each window flushes the queue, but kernel times
     are unaffected.
6. **Optional:** cmma f16 matmul behind a flag, a DX12 comparison, and a
   longer run of `training_recipe_check` on the GPU.

**Kill criterion:** if milestone 5's step isn't clearly faster than
~130 ms (a well-threaded CPU batch-8 step), stop and record why. The
expectation is ~2–5 ms: ~1.5 ms of launch floor plus kernel time.

## Out of scope for v1

qk-norm, the attention gate, sink logits, Adam, checkpoint I/O, and the
other examples. The ops table leaves room for them later.

## Risks

- **cubecl churn.** The API changed between pre.3 and pre.4. The version
  is pinned exactly, and upgrades are handled one at a time.
- **Shared eGPU.** Other sessions' jobs inflated timings up to ~100×.
  Check utilization before any timing.
- **Determinism.** Embedding and bias gradient reductions are written as
  ordered per-row loops, not atomics, so the parity tests stay exact
  enough.

## Open questions

- None open. The CPU batched-heads question is settled below.

## CPU model: batched heads (done 2026-09-24)

The CPU model moves to batched heads before milestone 2, so the GPU ops'
parity tests compare against a CPU tape with the same layout and op
boundaries.

**Two implementations, on purpose, for now.** The CPU tape stays the
readable reference, composed from small gradient-checked primitives. The
GPU step uses coarse cubecl kernels checked against it.

Could one cubecl source run on both? **No, measured 2026-09-24**
(experiment below). Launch cost turned out not to be the limit:
- The GPU's tiled matmul can't run on the CPU runtime at all. It uses
  `sync_cube` barriers, and the CPU runtime runs a barrier kernel's cube
  as one spinning OS thread per unit.
- The barrier-free naive kernel runs, but a step-shaped chain is 2.7×
  slower than single-threaded `NdArray`, with or without the launch
  patches.
- So a CPU-worthy cubecl kernel would have to be a different kernel
  from the GPU one. That is two implementations again, only both in
  cubecl.

Superseded reasoning, kept for the record: an earlier version of this
section argued from launch cost alone. First it said stock launch cost
ruled a single source out; then it said the `IDLE_POLL` patch's
0.05–0.4 ms launches kept it open. It also cited a 227 s JIT, which was
the whole smoke run; the JIT itself is ≤0.2 s per kernel.

Re-evaluate only if a CPU-shaped cubecl kernel (vectorized `Line<f32>`
rows, no barriers) gets near `NdArray`. See "Next" in the experiment.

**Decisions:**
- **Scope: full batched heads.**
  - `TransformerBlock` stores one `qkv: Linear` `[D, 3D]` in today's
    fused column order (`part·D + h·d_k + j`), replacing
    `q_heads`/`k_heads`/`v_heads`.
  - New tape ops `split_heads` (`[B·T, H·d_k]` → `[B·H·T, d_k]`) and
    `merge_heads` (its inverse). Both are permutations, so each backward
    is the inverse permutation; both are gradient-checked.
  - Attention is one `batched_matmul` over B·H, with the mask built for
    B·H.
  - `TransformerBlockOut` gets one `qkv_out` in place of
    `q_outs`/`k_outs`/`v_outs`. `head_weights` becomes one
    `[B·H·T, T]` Var, with a helper returning head h's `[B·T, T]` slice.
- **Extras: ported with `gather`.**
  - Attention gates fuse into one `[D, D]` Linear, applied after
    `merge_heads`.
  - Per-head qk-norm gamma/beta (`[H, d_k]` tables) and sink logits
    (`[H, 1]`) are broadcast to rows with the existing `gather`, using a
    row-to-head index.
- **Checkpoints: switch, no legacy reader.** Same flat length, fused
  order. No checkpoints are tracked in git.
- **Init: unchanged draws.** Draw each head's Q, K and V in today's RNG
  order, then pack them, so a given seed gives identical parameters.
- **Acceptance:**
  - The forward is bit-identical to the old per-head path. That's
    achievable because `NdArray::matmul` accumulates each output element
    over k in the same order at any width. The reference outputs are
    captured before the switch.
  - Gradients match to 1e-4 relative (the plan said 1e-5; measured
    worst case 1.05e-5 for parameter gradients, 2.6e-5 for dX). They
    can't be bit-exact: dX now sums 3D columns in one dot product
    instead of adding 24 per-head partial sums.
  - A short `training_recipe_check` loss curve tracks the old one.
  - `step_profile` records the CPU step time before and after.
- **Blast radius:**
  - 6 examples read `head_weights[h]`; they switch to the helper.
  - 4 memory_tier examples iterate `q_outs`/`k_outs`/`v_outs` leaves;
    that becomes one leaf.
  - `softmax1_divergence_diagnosis` reads per-head Q/K weight norms;
    those become column blocks of `qkv.w`.
  - `softmax1_qknorm_fix` has its own block copy and is unaffected.
  - `to_fused_flat`/`from_fused_flat` become `to_flat`/`from_flat`.

**Results (2026-09-24):**
- `batched_heads_match_per_head_reference` (in `src/nn.rs`) checks a
  plain block and a softmax1 block with QK-norm, gate and sinks against
  `src/testdata/batched_heads_reference.bin`, captured from the
  per-head code before the switch. The forward is bit-identical;
  gradients and dX are within 1e-4 relative.
- `training_recipe_check`, 200 steps (batch 8, lr 0.3): step 0 is
  identical (6.2731 / 6.3213). Train-probe / held-out end at
  2.6857 / 2.8260 before and 2.6914 / 2.8318 after, so the curve tracks.
- Time for 200 steps, back to back on an idle CPU: ~92–105 s before,
  ~72–73 s after, about 1.35× faster. That is inside the survey's ≤1.8×
  bound.
- `step_profile 8 10`: 280 tape nodes per step (was 888); forward
  144.8 ms, backward 204.3 ms, total 349.4 ms, 37% in matmul.
- `step_profile 8 1 census`: ~728 launches per step unfused, ~361 fused
  (7.3 vs 3.6 ms at ~10 µs per launch). The survey had estimated 776 /
  385.
- Old checkpoints hold per-head order at the same length, so they load
  without error but compute garbage. The ones in `runs/` were converted
  once, with the per-head originals kept as `*.ckpt.perhead`. There is
  no legacy reader, as decided.

### Survey: alternatives to full batched heads (2026-09-24)

Re-checked before implementing. Nothing beat the decision above.

| option | verdict |
|---|---|
| **Full batched heads** (`split_heads` → `batched_matmul` over B·H → softmax → `batched_matmul` → `merge_heads`) | **Keep.** It's exactly llm.c's CUDA attention pipeline (`permute_kernel`, batched matmul, softmax, batched matmul, `unpermute_kernel`, with permutes as copies), so the op boundaries are the standard ones. |
| Skip it: keep per-head CPU, bridge with `to_fused_flat` | Works for parity (M1 already checks the bridge). It forgoes the CPU speedup and leaves M2's per-op tests converting layouts ad hoc. |
| llm.c's CPU style: one fused attention op that indexes heads inside `[B·T, 3D]` by stride, with no copies and a hand-written backward | Fastest CPU, but the reference tape would then hold a coarse hand-written backward, the very thing it exists to check. |
| Strided views / einsum in `NdArray` (the PyTorch way) | General, but a large infrastructure change for one use. |
| Fuse only QKV into one `Linear`; keep the per-head attention loop via column slices | Smaller blast radius (`head_weights` and extras unchanged). It gets the wide-matmul gain but not the tape-node or launch-structure gain, and it's a second migration later. |

Upper bound on the CPU gain: `step_profile 8 10` with `CENSUS_HEADS=1` (a stand-in removed once real batched heads landed)
(same matmul work, 8× less softmax work, so a best case), CPU contended,
best of 5: 835 ms for 8 heads vs 471 ms for 1 head, **≤1.8×**. Real
batched heads keeps the per-head softmax work and adds permute copies,
so expect less. Measured afterwards: ~1.35×.

### Experiment: one kernel source on the CPU runtime (2026-09-24)

`cubecl_spike cpu-step` runs a step-shaped chain:
- 144 launches, alternating matmul (512,128)@(128,128) and elementwise;
- one sync, best of 5;
- next to the same chain on `NdArray`.

It also times each kernel's first launch (the JIT) against its second,
and checks results against `NdArray` (max error ≤ 5e-8).

Setup:
- Three `cubecl-cpu` builds, recreated per `spikes/cubecl_spike/PATCHES.md`:
  stock pre.4; #1658 applied; `IDLE_POLL` = 20 ms.
- Runs were pinned to 12 of 16 cores (`start /affinity FFF`) at
  BelowNormal, on an idle machine.

| variant | chain (ms) | vs NdArray 1 thread | JIT, first launch (mm / ew) | second launch + sync (mm / ew) |
|---|---|---|---|---|
| stock | 165 | 2.6× slower (64) | 44 / 21 ms | 24 / 15 ms |
| #1658 | 769 | 11× slower (67) | 93 / 188 ms | 101 / 91 ms |
| `IDLE_POLL` | 157 | 2.7× slower (59) | 46 / 24 ms | 1.1 / 0.46 ms |

Queued launch floor (`cubecl_spike cpu`, same setup), in ms per naive
matmul dispatch; `NdArray` is single-threaded:

| shape | stock | #1658 | `IDLE_POLL` | NdArray |
|---|---|---|---|---|
| 64 | 5.13 | 1.11 | 0.43 | 0.04–0.07 |
| 512x128 | 3.83 | 3.71 | 2.89 | 0.74–0.86 |
| 512x256 | 9.57 | 11.3 | 7.22 | 2.3–3.2 |
| 2048x128 | 7.92 | 7.49 | 8.54 | 3.5 |

What it shows:
- **Kill criterion met.** The chain had to come in clearly under
  ~130 ms. The best variant took 157 ms, and one thread of `NdArray`
  took 59 ms.
- **In a queued chain the kernel is the bottleneck, not the launch.**
  - `IDLE_POLL` only speeds up the sync round trip: 1.1 ms against 24 ms
    stock. Its chain matches stock.
  - The naive kernel reaches ~5–9 GFLOP/s on 12 cores, against ~20 on
    one core for `NdArray`, whose ikj loop vectorizes. The naive kernel
    walks `b` by column, one scalar unit per output element.
- **#1658 is a regression on Windows:** a 4.6× slower chain, and every
  sync costs ~0.1 s. Its gains were measured on Linux; it wakes and
  blocks workers per launch.
- **Barrier kernels are unusable on the CPU runtime.**
  - The tiled matmul uses 16×16 = 256 units per cube, with a barrier per
    k-step. It never finished one launch in 12 minutes: 258 threads, ~10
    cores spinning, 7,200 CPU-seconds.
  - By design, the CPU runtime (`threadpool/mod.rs`) grows its pool to
    one worker per unit and spins at `sync_cube`.
  - Any GPU kernel that uses shared memory and barriers therefore needs
    a separate CPU kernel.
- **JIT is not a blocker:** ≤0.2 s per kernel. #1527's missing cache
  would cost seconds per process start, not minutes.

Next, if anyone reopens this: write a CPU-shaped cubecl matmul, where
each unit owns output rows and runs an ikj loop over `Line<f32>`, with
no barriers. See whether it approaches `NdArray` there. Even if it does,
it's a second kernel, and single source was the point.

### Survey: cutting cubecl CPU launch overhead (2026-09-24)

Sources: tracel-ai/cubecl and tracel-ai/burn issues and PRs.

- **Upstream's diagnosis.** The pool's workers take every logical CPU
  and starve the client thread that feeds them; queued launches also pin
  their buffers until they run. Our `IDLE_POLL` patch worked the other
  way, keeping workers awake. That helps our chained small launches, but
  it's the spinning upstream is removing.
  - **#1545** (merged 2026-08-21, in pre.3/pre.4): idle polls yield
    instead of spinning. The author calls it a stopgap.
  - **#1658** (open, unreviewed): the client parks while waiting, and
    drained workers block immediately.
    - Gains: mobilenet batch 1 goes 37 → 21 ms; an 8-thread f32 matmul
      goes 21–32 → 18 ms on a Ryzen 5700X, and 104 → 50 ms on a Xeon.
    - Costs: tiny all-thread launches get a few µs slower from waking
      workers. Not measured on Windows or with more units than cores.
  - **#1566** (open draft; a superset of #1658): runs ahead without
    syncing, and releases a launch's buffers at enqueue. An 11-deep
    unsynced chain cost 40 ms before, against 9.5 ms synced after every
    op. Mobilenet batch 1 goes 38 → 7 ms (5.3×), and resnet50 batch 8
    only 1.04×.
- **#1527 (open):** the CPU runtime ignores the kernel cache config, so
  every process start pays the full JIT again.
- **#1635 (open):** the CPU runtime assumes a 512-bit load width, so
  vector widths are wrong on AVX2.
- **burn #4993 (open):** users report burn's cubecl CPU backend is far
  slower than burn's non-cubecl CPU backend on real models.
- **Not a CPU lever:** graph capture (#1505) is wgpu-only. See the GPU
  note below.
- **Branch status, checked again the same day.**
  - #1658 lives in `tracel-ai/cubecl`: mergeable, CI green, unreviewed.
    It only touches 5 files, all in `cubecl-cpu`. Its head has diverged
    from pre.4 (4 commits ahead, 22 behind).
  - #1661 (the load-width fix for #1635): mergeable and green, but it
    spans 10 files across 9 crates.
  - #1566: a draft that conflicts with main.
  - Nothing on main since pre.4 touches CPU launch cost or #1527. The
    big changes are an adaptive memory pool (#1685) and turso
    persistence (#1677).
  - Windows CI is still off (#104).

**Git dependency or local patch?** (Moot since the experiment below: the
local patches were enough to test, and single source failed.) The user
leaned toward a single source (2026-09-24), so the question is how to get upstream's CPU fixes before
they're released.
- **A git dependency is reasonable for a pre-release**, but not for
  this:
  - `cubecl` and `cubecl-runtime` must move to the same `rev` together.
  - Pinning #1658's head means a tree 22 commits behind pre.4, with API
    drift in both directions.
  - Combining #1658 with #1661 needs a merge branch we'd maintain,
    which is a fork.
  - PR branches get deleted after merging.
- **A local patch fits better.** Keep pre.4 from crates.io and apply
  #1658's 5-file `cubecl-cpu` diff as
  `[patch.crates-io] cubecl-cpu = { path = ... }`, the same mechanism
  as the existing bundler patch.
- **Switch to a git dependency** when a needed fix spans crates (like
  #1661), or to track main between pre-releases once the single-source
  path is adopted.

### Found along the way: graph capture for the GPU step

cubecl pre.4 has wgpu graph capture (#1505, merged 2026-08-18) on the
public client: `graph_prepare()` before a warm-up step, then
`start_capture()` / `stop_capture() -> Graph` around a step, then replay.
It records a whole step's launches once and replays them, which removes
per-launch encoding and metadata uploads (#1504 caches the uniforms).
Candidate for milestone 5 or 6, measured against plain queued launches.
Unverified here: shapes must stay fixed between replays, and the Vulkan
SPIR-V path may not support it.
