# GPU training step: design (2026-09-24)

Goal: run one tiny_lm training step (batch 8, `d_model` 128, 8 heads,
`seq_len` 64, 4 blocks, `d_ff` 256, vocab 256, SGD, cross-entropy)
entirely on the RX 9060 XT eGPU. It must match the CPU tape's losses and
gradients, and be faster than the CPU step. Status: **milestone 1 done
(2026-09-24)**. Next: move the CPU model to batched heads (section
below), then milestone 2 (kernels).

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
   - `TransformerBlock::to_fused_flat` / `from_fused_flat` convert to and
     from the fused-QKV layout; `gpu_step::pack` flattens a whole model.
     A whole-model unpack waits for milestone 5, the first thing that
     needs one.
   - Checks: `fused_qkv_layout_round_trips_and_matches_per_head_projections`
     (CPU, runs by default) and `device_sgd_matches_optim_sgd` (GPU,
     `cargo test --lib gpu_step -- --ignored`, since it needs the
     discrete GPU).
2. **Kernels one by one.** Each forward is compared with the CPU tape's
   composed ops on the same inputs. Each backward is checked two ways:
   against CPU gradients, and by finite differences through the GPU
   kernels. That's the branch's discipline.
3. **Forward parity.** A block's output and the loss match
   `forward_full` (plain and softmax1) to 1e-4 relative.
4. **Gradient parity.** Every parameter gradient after one step matches
   the CPU tape to 1e-4 relative.
5. **Training parity and timing.**
   - 200 steps from the same init and batches should track the CPU loss
     curve.
   - Step time is timed against the CPU step, best-of-N if the eGPU is
     shared.
   - Launches per step are counted.
   - Utilization is checked on the discrete GPU.
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

## CPU model: batched heads (decided 2026-09-24)

The CPU model moves to batched heads before milestone 2, so the GPU ops'
parity tests compare against a CPU tape with the same layout and op
boundaries.

**Two implementations, on purpose, for now.** The CPU tape stays the
readable reference, composed from small gradient-checked primitives. The
GPU step uses coarse cubecl kernels checked against it.

Could one cubecl source run on both? Not yet:
- **Stock launch cost.** The cubecl CPU runtime costs ~3.4 ms per
  launch (pre.3; still milliseconds on pre.4, `spikes/cubecl_spike/
  PATCHES.md`). That's ~0.5 s for the 145-launch step, and its matmul
  lost to single-threaded `NdArray::matmul`.
- **Patched launch cost.** Our `IDLE_POLL` patch got a launch down to
  0.05–0.4 ms, which is 7–58 ms for 145 launches. That's competitive with
  the ~130 ms CPU step, and the patched matmul beat `NdArray`. So launch
  cost alone doesn't rule it out. (An earlier draft of this section said
  it did, using only the stock figure.)
- **The blockers are elsewhere** (online survey, 2026-09-24, below):
  - The fixes are unmerged and untested on Windows.
  - There's no kernel cache, so every process start re-pays the JIT
    (227 s in our smoke run).
  - The patch keeps 16 threads spinning on a machine other sessions
    share.

Re-evaluate when #1566 or #1658 merge and #1527 is fixed.

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

### Found along the way: graph capture for the GPU step

cubecl pre.4 has wgpu graph capture (#1505, merged 2026-08-18) on the
public client: `graph_prepare()` before a warm-up step, then
`start_capture()` / `stop_capture() -> Graph` around a step, then replay.
It records a whole step's launches once and replays them, which removes
per-launch encoding and metadata uploads (#1504 caches the uniforms).
Candidate for milestone 5 or 6, measured against plain queued launches.
Unverified here: shapes must stay fixed between replays, and the Vulkan
SPIR-V path may not support it.

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
  - Gradients match to 1e-5 relative. They can't be bit-exact: dX now
    sums 3D columns in one dot product instead of adding 24 per-head
    partial sums.
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
