# GPU training step: design (2026-09-24)

Goal: run one tiny_lm training step (batch 8, `d_model` 128, 8 heads,
`seq_len` 64, 4 blocks, `d_ff` 256, vocab 256, SGD, cross-entropy)
entirely on the RX 9060 XT eGPU. It must match the CPU tape's losses and
gradients, and be faster than the CPU step. Status: **design, nothing
built yet.**

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
- **Placement:** the code goes in `src/gpu_step/` behind a `gpu` cargo
  feature, so the default build stays free of the pre-release dependency.
  The CPU runtime (`cubecl/cpu`) is not used.

## Milestones and checks

Each milestone leaves a runnable check behind.

1. **Scaffolding.** Feature flag, device selection and logging, the flat
   parameter and gradient buffers, and pack/unpack against the CPU model.
   The SGD kernel is checked against `optim::Sgd`.
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

- Should `gpu_step` live in the main crate behind a feature (the current
  plan), or in a separate crate like the spike?
- Should `nn.rs` move to batched heads on the CPU too? GPU parity doesn't
  need it, because pack/unpack bridges the layouts. It would speed up the
  CPU step and let the census's "heads batched" row become the real tape.
