# Gating the memory tier by uncertainty

Status: design, awaiting review (2026-10-07). Context: `docs/tiers_design.md` ("Order", step 3: how the tiers interact);
README "Direction: tiered AI".

## Question

Today the kNN memory is mixed in with a fixed weight: `p = (1 - lambda) p_model + lambda p_knn`, lambda = 0.5, temperature 15,
k = 256 (`Knn::adjust` in `examples/tiny_lm/tier_eval.rs`), after the CPU tiers (lexicon, word bigram) have rewritten
`p_model`. Does a per-position weight, set from signals the other tiers already produce, lower held-out CE? This is the
first test of one tier gating another.

Deployable stack and baseline: online memory + lexicon 0.3 + words 1:0.25 + knn 256:0.5:15, warm 32, stride 1,
held-out CE **1.1313** (model alone 1.1764; `docs/tiers_design.md`, tier_full9).

## Key observation: a position's CE needs five numbers

The mixture is linear, so a position's loss at any lambda is `-ln((1 - l) p_t + l k_t)` where `p_t` is the target's
probability under the distribution entering the memory tier and `k_t` its probability under the kNN distribution. Gate
signals need only: model entropy `H` (of the distribution entering the memory), nearest-key squared distance `d0`, k-th
distance `dk`. So one run of `tier_eval` can dump `(chunk, p_t, k_t, H, d0, dk)` per scored position (503,808 rows, ~12 MB)
and every gate is then fitted and scored offline in seconds, with no further GPU time. This holds because the memory is the
last tier in the stack; the dump is only valid for specs that end in `knn`.

## Parts

1. **Dump** (`tier_eval.rs`, ~25 lines): option `dump=<path>` on the scoring pass writes the rows above (little-endian f32,
   chunk as u32) for the last `knn`-terminated spec. No change to scoring.
2. **Fit and score** (new `examples/tiny_lm/gate_fit.rs`): reads the dump and reports, fitting on even chunks and scoring
   on odd chunks (both halves see the same memory growth, so no leakage):
   - fixed lambda = 0.5 CE, and the best fixed lambda (reference);
   - **binned**: bins of `H` x `d0` (quantile edges from the even half), the best lambda per bin by 1-D search, out-of-sample
     CE on the odd half, and the table of per-bin lambda (shows the gate's shape and whether it varies at all);
   - **parametric**: `lambda = sigmoid(a + b H + c ln d0 + d (dk - d0) / d0)`, 4 parameters, fitted by gradient descent on the
     even half with analytic gradients, out-of-sample CE on the odd half.
3. **Record** the table, CE numbers and the fitted parameters in `docs/tiers_design.md`.

## Checks and success criteria

- **Gate 0:** the dump's own CE at lambda = 0.5 over all rows must equal the run's logged CE for that spec (1.1313) to 1e-4,
  or nothing after it counts. `gate_fit` prints both.
- A unit test of the parametric gradient against finite differences (the one small runnable check).
- **Success:** odd-half CE of the parametric gate below odd-half CE at fixed lambda 0.5 by a margin worth keeping
  (> 0.002 nats; the memory's whole gain is 0.041). **Kill:** if the binned (more expressive) gate gains < 0.002 out of
  sample, per-position lambda from these signals does not help; stop, record, and move to a richer signal or tier.

## Out of scope (YAGNI)

A learned MLP mixer; gating the lexicon and word-bigram weights (a follow-up if memory gating works; they are not last in
the chain, so their dump needs a different shape); changing k or temperature; non-causal signals.

## Cost

One `tier_eval` run to produce the dump (~14 min GPU, same as tier_full9, on a quiet machine with a lease and
`load_log.ps1`); fitting is CPU seconds. Files touched: `tier_eval.rs`, `gate_fit.rs` (new, with a Cargo `[[example]]`
entry), `docs/tiers_design.md`.
