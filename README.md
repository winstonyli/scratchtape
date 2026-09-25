# scratchtape

A machine learning engine built from scratch in Rust: tensors, a
reverse-mode autodiff tape, neural network layers, optimizers, and a GPU
backend, with no ML framework underneath any of it. Every primitive
(matmul, softmax, cross-entropy, gather, attention) is hand-written and
gradient-checked, not a call into an existing library. Layered on top:
a byte-level transformer LM, several alternative architectures compared
against it, and an extended investigation into what symbolic structure
(if any) a small trained transformer's internals actually contain.

The project's own history is part of its documentation. Every commit
message narrates what was tried, what was measured, and what the result
actually was — including the negative ones. `git log` reads like a lab
notebook; this README is the map, not a replacement for it.

## Quick start

```bash
cargo test --release              # 24 gradient-check / correctness tests (+1 GPU-only: -- --ignored)
cargo build --release --examples  # build everything under examples/
cargo run --release --example tiny_lm
```

Most examples follow the same shape: encode a small corpus, train a
tiny transformer, print what happened. A few (`checkpoint_save`/
`checkpoint_load`, `memory_tier_*`) are pairs of programs that write a
file and read it back in a genuinely separate process.

`examples/` is grouped into subdirectories by investigation line
(`fundamentals/`, `tiny_lm/`, `alternative_mechanisms/`,
`memory_tiers/`, `differentiable_reasoning/` — matching this section's
own headings below), each wired up via an explicit `[[example]]` entry
in `Cargo.toml` (Cargo only auto-discovers examples at the top level).
Names are unchanged from the original flat layout, so every
`cargo run --example <name>` below still works exactly as written.
`examples/common/` holds the handful of helpers that crossed this
project's own "2+ consumers" promotion bar many times over
(`encode_bytes` alone was byte-identical across 22 files) — plain
functions, not a trait, since none of them have more than one real
implementation to swap between. Kept in `examples/`, not `src/`: this
is example glue (byte encoding, window sampling, the standard tiny-
transformer forward/apply_grad pair), not an engine primitive.

## Engine (`src/`)

| module | what it is |
|---|---|
| `tensor.rs` | `NdArray` — a minimal n-dimensional array (data + shape), the value type everything else operates on. |
| `tape.rs` | `Tape` — reverse-mode autodiff. Every op (`add`, `matmul`, `gather`, `softmax`, `softmax1`, `cross_entropy`, `batched_matmul`, `max_last_axis`, ...) is a node with a forward and backward rule, gradient-checked against finite differences. `max_last_axis` is differentiable on purpose, unlike `NdArray::max_last_axis` (kept non-differentiable, used only for softmax's shift-invariant stability trick) — added for a real OR-module in the differentiable-reasoning line below, where gradient through *which candidate wins* is the whole point. |
| `nn.rs` | Layers built from tape ops: `Linear`, `LayerNorm`, `Embedding`, `TransformerBlock` (multi-head causal self-attention + FFN), plus `Rng` (hand-rolled xorshift, no external RNG crate). |
| `optim.rs` | `Sgd` and `Adam`. |
| `gpu.rs` / `matmul.wgsl` | A wgpu compute-shader matmul kernel — verified correct, but not wired into the autodiff path; see the persistent-GPU-backend note below. On wgpu 30, the same version cubecl uses. |
| `gpu_lease.rs` | Taking turns on the shared eGPU: one lease file per GPU job in `%LOCALAPPDATA%\gpu-leases`, the same format as the LONG_RUNS.md snippet. Benchmarks hold an exclusive lease, and long runs pause for one at checkpoints. |
| `gpu_step/` | The device-resident training step (cubecl, Vulkan, discrete GPU only): flat parameter and gradient buffers, one-launch SGD. In progress, see [`docs/gpu_step_design.md`](docs/gpu_step_design.md). |

Design stance, held consistently throughout: build the primitive
yourself before reaching for a library, verify it against a
finite-difference gradient check or a direct numerical comparison, and
promote example-local code into the library only once a second consumer
actually needs it.

## Examples (`examples/`)

**Fundamentals** — `xor_mlp.rs`, `xor_mlp_adam.rs`, `xor_es.rs` (evolution
strategies vs backprop on the same task), `linear_regression.rs`,
`byte_tokenizer.rs`.

**The transformer LM family** — `tiny_lm.rs` is the original: byte-level
tokenizer, embeddings, causal transformer blocks, trained on a 172-byte
Shakespeare excerpt. `tiny_lm_scaled.rs` / `tiny_lm_scaled2.rs` scale the
model up in isolation; `tiny_lm_batched.rs` adds minibatching via
`BatchedMatMul` (measured net-negative at this scale, kept as
infrastructure anyway). `tiny_lm_corpus.rs` moves to a real ~238KB
corpus (Aesop's Fables) with a 90/10 train/held-out split — and is also the
single largest file in the project (see below).

**Alternatives to attention** — `attention_recall.rs` /
`multihead_attention_recall.rs` (single- vs multi-head recall, showing
one head can be blind while another is sighted on an identical query),
`ssm_recall.rs` (a diagonal linear SSM, measuring its vanishing-gradient
shape directly rather than citing it), `softmax1_comparison.rs`
("Attention Is Off By One" — tried against a real frequency-sink finding
later, diverged regardless of learning rate, reverted; the mechanism
itself is kept as an opt-in `TransformerBlock::forward_full` flag).

That divergence was originally parked with a guess ("same failure
class as the PC-precision instabilities") never actually tested -
`softmax1_divergence_diagnosis.rs` tested it directly by instrumenting
a real run instead of citing the comparison. The guess was wrong: this
isn't a discontinuous warm-up jump like PC's (no smooth ramp would fix
it). It's a continuous, accelerating runaway from step 0 — block0's
Q/K weight norms and their own gradient grow together (confirmed via a
side-by-side run against plain softmax from identical seeds: plain
softmax's Q/K gradient stays bounded for 2000+ steps, softmax1's
accelerates from ~6 to ~195 before NaN at step 775). One clean
candidate explanation for *why* softmax1 lacks plain softmax's
self-limiting saturation — the local derivative of the max-attention-
weight term — was checked by direct calculation and is identical for
both variants, ruling it out rather than confirming it; the real
mechanism is a multi-layer effect (softmax1's variable row-sum
interacting with the residual stream) not fully decomposed here.
`softmax1_qknorm_fix.rs` tested the natural fix anyway — L2-normalizing
Q/K before the dot product, bounding every score regardless of
underlying weight-norm growth — using a self-contained reimplementation
of the block (`TransformerBlock`'s own fields are private, unlike its
`Out` struct) rather than another library change. It works completely:
a full 2000-step run with zero divergence, Q/K weight norm and gradient
both staying flat the whole time, against the same setup's step-775 NaN
without it.

*(Retracted 2026-09-23: this divergence was a `Tape::softmax1` bug; see
the correction further down. Rerun with the fixed op, the diagnosis
completes all 2000 steps with no NaN and no QK-norm. Block0's Q+K
norm stays flat at ~90.24, its gradient stays in plain softmax's range,
and scores stay within ~±15. softmax1 itself was never unstable here;
QK-norm "fixed" a bug.)*

That proof lived entirely in a standalone reimplementation, not the real
`TransformerBlock` any actual model uses. Closed the loop: QK-norm is
now a `use_qknorm` flag on `TransformerBlock::forward_full` itself
(opt-in, every existing call site unchanged, gradient-checked against
finite differences), then re-ran the frequency-sink question on
`tiny_lm_corpus.rs`'s real scaled corpus rather than the demo — a
matched control/treatment pair from an identical seed, full 16000
steps (8x the demo's own proof). Zero divergence either run. The named
sink target ('a') stayed a 1-byte edge case under both, consistent
with it never being a reliable seed-stable sink to begin with, but
distinct bytes chosen as anyone's top-1 attention target rose from 48
to 68 (of 92) under softmax1+QK-norm — attention spreading across more
distinct targets instead of concentrating, real evidence for the
original theory on the model it was always about.

That first pass used one seed — the same shape of claim this project's
own methodology already flagged as unreliable ([ccd411e] found the
original single-seed 'a'-sink itself was seed-arbitrary), and never
checked whether the fix costs anything in modeling quality. Reran with
3 fresh seeds, each a control/treatment pair from identical seed/data/
steps: the diversity result replicates cleanly (3 of 3 seeds higher,
48→68/48→68/44→72, mean +22.7) — real, seed-stable structure. But it
isn't free: held-out loss is consistently worse under softmax1+QK-norm
on every seed (mean delta +0.216 nats). The fix does what it was
designed to do — attention stops concentrating onto a small shared
set — but at this matched step count that costs real modeling quality,
not a free improvement. Left open: whether that gap is a step-budget
confound (this project's own 2x-model-scale precedent closed almost
its entire gap once given proportionally more steps) or a genuine
tradeoff intrinsic to the mechanism.

That question is answered, not left open: it's a step-budget confound.
Retrained the same 3 treatment seeds at 4x the steps (64000, same
extension factor as the 2x-model-scale precedent), control unchanged
(QK-norm adds no meaningful per-step compute, so this tests
convergence speed, not FLOPs-matching). The mean held-out-loss delta
against control collapses from +0.216 at matched steps to -0.010 at
4x steps — crossing to a hair better, with 2 of 3 seeds beating their
own control outright. Softmax1+QK-norm just converges slower per
step; given a fair budget it reaches parity or better while keeping
the attention-diversity gain. The fix works, replicates, and costs
nothing once fairly trained. *(Retracted below: this compared 64000
treatment steps to 16000 control steps; at equal steps plain softmax
wins by ~0.28 nats.)*

*(Promotion later reversed — see "Fixing QK-norm doesn't rescue
softmax1" below.)* Promoted to `tiny_lm_corpus.rs`'s sole attention path (softmax1+QK-norm
always on, `steps` at the proven-necessary 64000), then re-ran every
downstream KR&R check against the old plain-softmax baseline. Several
findings sharpen: the k-means vowel cluster, previously a case-mixed
grab-bag, is now exactly the 7 vowels and nothing else; block-0/head-0
cross-seed attention stability jumps from 7 unanimous/43 majority/37
all-different (plain softmax) to 63/24/0 (softmax1+QK-norm) out of 92
bytes; is-vowel decision-tree accuracy rises from 0.815 to 0.924. None
of that can be credited to the mechanism alone from this run — the
mechanism and the step budget both changed in the same commit, and
this project's own step-budget-confound precedent already showed step
count alone moves these numbers. A plain-softmax control at the same
64000 steps would be needed to isolate it. One result cuts against the
fix's own motivation: `attention_graph_all_heads()` now shows all 32
(block, head) combinations converging on the identical top sink target
byte, with uniformly high self-attention rates (0.64–0.78), versus the
baseline's per-head-varied sinks and widely spread rates (0.07–0.74) —
a *more* homogeneous sink pattern, not less, at the all-heads level.
Reported as found: softmax1+QK-norm doesn't obviously reduce sink
concentration once every head is examined; it may just relocate which
byte the sink converges on.

Ran the isolating control both open questions raised: plain softmax at
the same 64000 steps (a temporary local flip, reverted right after),
same seeds, same everything else. Cross-seed attention stability goes
7→20→63 unanimous (16k plain softmax → 64k plain softmax → 64k
softmax1+QK-norm); is-vowel decision-tree accuracy goes 0.815→0.859→
0.924, same shape; the vowel cluster stays a case-mixed grab-bag under
plain softmax even at 64000 steps, only softmax1+QK-norm produces the
clean 7-vowel cluster. More training alone buys a real but partial
gain — the mechanism adds substantially more on top of it, resolving
the confound. And the uniform-sink surprise turns out not to be a
training-length artifact either: plain softmax at 64000 steps still
shows 32 different per-head sinks and a wide self-attention-rate
spread (0.10–0.74), unchanged in kind from the 16k baseline. All 32
heads converging on one shared sink byte is a genuine, previously-
unseen behavior specific to softmax1+QK-norm at this step budget —
sitting alongside, not resolved by, the top-1-diversity gain the
mechanism was adopted for. Both are real; they measure different
things.

Traced why `'¼'` specifically, from data already on hand: it occurs 8
times in the 237,882-byte corpus (0.0034%) — among the rarest bytes
that still clear the noise filter — with accumulated |gradient| 113.2
over 64000 steps, 7th-lowest of the 92 filtered bytes (~300–450x
smaller than common letters). Barely ever sampled as a target, its
embedding stays near initialization while every well-trained byte's
embedding moves substantially, and the top-2 attention dump shows most
of the defaults to it sitting at weight 0.02–0.11 — barely above
uniform (1/64), a weak tie-break, not confident attention. That's the
mechanistic split from plain softmax: its score is an unnormalized dot
product scaling with magnitude, so an undertrained embedding produces
a weak score and rarely wins — plain-softmax sinks instead form around
whichever byte's key happens to develop a large norm in a given run,
hence seed-arbitrary and head-varied. QK-norm discards magnitude
entirely; an embedding that stayed near small-random init (from almost
never getting a gradient) sits in a comparatively generic direction,
which tends toward moderate, non-negative cosine similarity to a broad
range of other directions — a structural "least-bad default." Every
block/head reads the same shared embedding table, so that byte's
peculiarity is available identically everywhere, explaining the
cross-block, cross-head, and cross-seed convergence plain softmax
never showed. Not verified directly: this chains corpus frequency →
gradient starvation → attention weight, not the trained K-vectors' own
cosine geometry — that would need a checkpoint dump, left as an
optional deeper confirmation.

**Correction — the sink-uniformity finding and the `'¼'` mechanism
above are largely a measurement artifact.** Before spending an hour on
a checkpoint retrain, ran the cheapest discriminating test: the same
sink metric with no model at all, on exactly uniform causal attention
(`attention_null_baseline.rs`). It reproduces the reported result: the
same 19953 usable windows, self-attention rate 0.72 (reported
0.64–0.78), and `'¼'` as top sink with 10 votes (reported 9–11).
Cause: softmax1+QK-norm bounds every score to ±0.25, so no weight in a
row can exceed another by more than e^0.5 ≈ 1.65× — attention is
near-uniform by construction. The metric averages weight per (query
byte, key byte) pair over one fixed window set shared by every head and
seed and takes the argmax; with almost no content signal that argmax
is decided by where each key byte sat in those windows plus small-
sample noise for rare bytes (`'¼'` occurs 6 times in train). Plain
softmax's peaky attention overrides that baseline, hence its varied
sinks. The "undertrained embedding → generic K direction" story is
retracted — it explained a pattern the null already produces (and never
explained why `'¼'` beat bytes with lower gradient sums). Also in
doubt, since they read attention argmax over the same windows: the
top-1-target diversity gain (48→68), the cross-seed stability jump
(7→20→63 unanimous — shared windows make a window-determined argmax
agree across seeds by construction), and the attention-neighbor KG/
graph-probe results. Standing: everything computed from embeddings
alone (the clean 7-vowel k-means cluster; is-vowel decision-tree
0.815→0.859→0.924, isolated from the step budget by the plain-softmax
control) and zero divergence. (This paragraph originally also listed
held-out-loss parity at 64000 steps as standing — retracted in the next
paragraph.)
The real mechanism-specific fact is that QK-norm makes attention
near-uniform; whether the trained model deviates meaningfully from
uniform at all is the open question, needing a checkpoint and a direct
per-head KL-from-uniform measurement rather than another argmax metric.
The promotion rested partly on the diversity evidence, now unproven.
Lesson, same shape as the earlier single-seed one: an argmax-style
metric needs a null baseline before its output is read as structure.

**Measured directly, and the promotion doesn't survive.**
`attention_uniformity_check.rs` retrains the seed-1 model under both
conditions at the same 64000 steps (reproducing tiny_lm_corpus.rs's
runs exactly: identical loss trajectories), saves checkpoints, and
measures every head over the sink metric's windows. Softmax1+QK-norm:
mean KL from uniform 0.0043 nats, peak weight 1.16× uniform (the
construction allows 1.65×), row mass 0.94, and 93% of query bytes'
argmax keys identical to the uniform null's — every head in every
block is effectively a causal mean-pool. Plain softmax: KL 2.24, peak
21×, 36% null agreement — genuinely selective, so its varied per-head
sinks (newline, `;`, `:`, `.`) are mostly real content, and its 7/20
unanimous cross-seed counts are not the artifact the 63 is. Worse, the
"parity" leg was a step-unmatched comparison all along: the step-budget
test gave the treatment 64000 steps but compared it to the control's
16000. At matched steps (seed 1) held-out loss is 2.026
(softmax1+QK-norm) vs 1.738 (plain), plain lower at all 40 evals after
init — a ~0.28-nat cost (mean of the last 8 evals: 2.145 vs 1.868), not
a slower route to the same place. "Costs nothing once fairly trained"
above is wrong: the fair comparison is equal steps, and it costs. Root
cause is an implementation gap, not the idea: `l2_normalize_rows` in
`src/nn.rs` normalizes Q/K with no learnable scale, then divides by
√d_k, capping logits at ±0.25. Published QK-norm (Henry et al. 2020)
replaces that fixed 1/√d_k with a learned scalar exactly so attention
can sharpen. It still eliminates softmax1's divergence — by making
attention too flat to diverge. What stands from the promotion: zero
divergence, and the embedding-only results (vowel cluster, 0.924
decision tree) as measurements of *this* model, though with attention
this flat those may reflect the model leaning on per-byte embeddings
because it can't select context — not a better attention mechanism.
One seed; the held-out eval is noisy (±0.2 between evals), but the
sign held at every one.

**Fixing QK-norm doesn't rescue softmax1; it exposes it.** Replaced the
library's L2 QK-norm with per-head LayerNorm on Q and K
(`TransformerBlock::with_qk_norm`, ViT-22B's form: learnable γ, 1/√d_k
kept, so logits start near ±√d_k instead of ±0.25; nGPT independently
diagnoses the same bug: unit-normalized q·k has variance 1/d_k, so the
right factor is √d_k). Reran `attention_uniformity_check.rs` with three
conditions, seed 1, 64000 steps:

| Condition | KL from uniform | Peak | Null agreement | Held-out | Train |
|---|---|---|---|---|---|
| plain softmax | 2.24 | 21.4× | 0.36 | 1.738 | 1.362 |
| plain + LN QK-norm | 1.71 | 15.8× | 0.47 | 1.740 | 1.239 |
| softmax1 + LN QK-norm | 0.40 | 5.0× | 0.61 | 2.976 | 3.013 |

LN QK-norm under plain softmax is stable, keeps attention selective, and
ties plain on held-out loss (fits train better, generalizes no better).
Softmax1 with it tracks the other two to step 1600 (2.82 held-out), then
jumps to ~3.0 and stays there — barely better than a unigram model (3.27)
— with no NaN. The checkpoint shows why: block 0's Q/K norm gains grew to
~1e9 and its K-norm bias to ~2e8, while blocks 1–3 sit at their init
norms. A key bias shifts every logit in a row by the same amount, which
has exactly zero gradient under plain softmax (its K biases stayed 0.00)
but moves softmax1's row mass, so nothing bounds it: the same unbounded
pressure behind softmax1's original divergence. The L2 version's "zero
divergence" came from capping logits at ±0.25, which also removed
selectivity. Net: softmax1 isn't viable in this setup (SGD, lr 0.3)
without bounding the logit scale some other way, and the promotion is
reversed: `tiny_lm_corpus.rs` is back on plain softmax. Every
`tiny_lm_corpus.rs` result recorded between `e4820bd` and this change
came from the L2 softmax1+QK-norm model. For context, recent work puts
softmax1's upside at small scale in the hundredths of a nat (abstention
~0.019 nats at 10M params, shrinking with scale; Wang 2026). Gated or
sink-logit variants (Qwen's gated attention, NeurIPS 2025; a learnable
phantom key) get abstention without unbounded logits. One seed.

**Correction: the runaway was a `Tape::softmax1` bug, present since
`20e5038`.** Writing a finite-difference test for a new sink-logit op
exposed it. softmax1 subtracted the row max for overflow safety but kept
the literal "+1", computing exp(x)/(Σexp(x) + exp(max x)) — a phantom key
pinned at the row's max, not at 0 (0.44 instead of 0.71 on a test row).
That forward pass ignores a uniform shift of the row; the gradient,
taken with the max detached, doesn't. So any parameter that shifts a
whole row (a key bias, a K-norm β) received a steady gradient that
changed nothing in the loss, and drifted without bound: exactly the
~2e8 K-norm bias and ~1e9 gains above, and the "Q/K-weight-norm runaway" behind softmax1's original step-775 NaN
(`softmax1_divergence_diagnosis.rs`; confirmed by rerun: the fixed op
never diverges). The paragraph above explains it
as softmax1 legitimately rewarding shifts; that's wrong for the code
that ran. Fixed: shift by max(max x, 0) and use exp(−m) for the phantom
term — exactly softmax over [x, 0], where detaching the shift is valid.
A new test checks values against the definition and gradients against
finite differences (it fails on the old code). Every softmax1 result in
this project — `softmax1_comparison.rs`, the divergence diagnosis, the
QK-norm fix, the promotion, and the run above — used the broken op. The
project never had a direct gradient check on softmax1 because it
"composed already-checked ops"; the composition itself was the bug.
Rerun with the fixed op pending.

**What `tiny_lm_corpus.rs` reports now (plain softmax, 64000 steps).**
The plain-softmax control run above *is* the current file's output, so
its numbers replace the headline figures from the promotion paragraph.
The is-vowel decision tree scores 0.859, not 0.924. Block-0/head-0
cross-seed attention stability is 20 unanimous of 92 bytes, not 63.
k-means gives no clean vowel cluster: 'A' 'I' 'O' 'U' 'u' cluster
together with '-' and two stray bytes, while 'a' 'e' 'i' 'o' sit in a
large lowercase cluster. Per-head sinks vary across all 32 heads, with
self-attention rates spread over 0.10–0.74. Seed-1 held-out loss is
1.738 (1.93 at 16000 steps). The 0.924, clean 7-vowel, and 63-unanimous
figures belong to the L2 softmax1+QK-norm model, run with the buggy
softmax1, and aren't reproducible from the current code.

**A count model beats the transformer.** `ngram_baseline.rs` fits
interpolated Kneser-Ney over bytes (Ney discounts per order, uniform
floor, sum-to-1 checked at every order). It scores held-out with the
transformer's own windowing: 64-byte windows, context reset at each
window. Held-out CE falls from 3.266 (unigram) to **1.655 nats/byte at
order 7**, then flattens. The plain-softmax transformer logged 1.738 at
step 64000 and averaged 1.868 over its last 8 evals. That eval is a
noisy 20-window sample, so a deterministic full-held-out eval of each
checkpoint on the same windows is next. Either way, this model isn't
clearly beating a 1998-era count model, and every mechanism comparison
here is happening below that bar.

**Rerun with the fixed softmax1, scored deterministically.**
`attention_uniformity_check.rs` now reports held-out CE over all 371
non-overlapping 64-byte windows, the same windowing as the n-gram. Seed
1, 64000 steps, all variants stable:

| Model | Held-out (nats/byte) |
|---|---|
| Kneser-Ney 7-gram | **1.655** |
| softmax1 | 1.803 |
| sink logit (learnable phantom key) | 1.818 |
| plain + output gate (Qwen G1) | 1.821 |
| softmax1 + LN QK-norm | 1.824 |
| plain softmax | 1.852 |
| plain + LN QK-norm | 1.856 |

Every abstention mechanism beats plain by 0.03–0.05 nats, the order
Wang 2026 reports at 10M params. softmax1 alone is best, and QK-norm adds
nothing once softmax1 is correct. The gates never go sparse (mean
0.39–0.46), matching Rizwan et al.'s 1M-param observation. The training
log's 20-window evals ran optimistic by 0.1+ nats (plain logged 1.738).
The headline stands: every transformer here trails the count model by
0.15–0.20 nats, so training (not attention variants) is the bottleneck.
One seed. Over 5 seeds at batch 1 (below), softmax1's lead over plain
holds but is 0.021, not 0.049: this plain run was an unlucky draw.

**Minibatching doesn't close the gap.** `training_recipe_check.rs` varies
the recipe one factor at a time at a fixed data budget (64000 windows,
so batch changes step count, not data), seed 1, SGD lr 0.3, same
deterministic eval. Train-probe is the first 371 windows of train, for a
like-for-like overfitting gap. Runs take hours, so they save a resumable
checkpoint every 10 minutes; relaunching the same command continues
bit-identically (checked by killing and resuming a short run).

| Recipe | Steps | Train-probe | Held-out |
|---|---|---|---|
| plain, batch 1 | 64000 | — | 1.852 |
| plain, batch 8 | 8000 | 1.390 | 1.858 |
| plain, batch 32 | 2000 | 1.806 | 2.128 |
| softmax1, batch 1 | 64000 | — | 1.803 |
| softmax1, batch 8 | 8000 | 1.375 | 1.846 |
| softmax1, batch 32 | 2000 | 1.813 | 2.115 |

On seed 1, batch 8 matched batch 1 in 8× fewer updates; over 5 seeds it
doesn't (below: batch 1 is 0.03–0.04 better). Batch 32 at unscaled lr is
under-stepped, 0.27 behind at equal data and still falling steeply.
softmax1's lead over plain shrinks from 0.021 (batch 1, 5 seeds) to
nothing detectable at batch 8 (5 seeds, below). The limiting factor is
overfitting: at batch 8 the train/held-out gap is 0.47 and was still
widening at the end, while held-out gains had slowed to 0.03 per 8000
windows. Regularization (weight decay, dropout) is the next lever, ahead
of the optimizer.

**On the GPU the same run takes ~20 s, not ~4.5 h** (`training_recipe_check
... gpu`, 2026-09-25): ~1.9–2.0 ms/step uncontended, with evaluation on the
device too (~0.3 s each, checked against the CPU at the end). Batch 8 ends at plain 1.357 / 1.857 and softmax1 1.378 /
1.860 (train-probe / held-out). The GPU and CPU runs are chaotic twins
(milestone 5 below), so their gap measures seed-level noise: 0.001
held-out for plain, 0.014 for softmax1, which trails plain on the GPU.
That noise is as big as softmax1's batch-8 lead, so 5 seeds each were
run (`... gpu <seed>`, seeds 1–5, ~20–40 s each). Held-out CE:

| batch 8 | mean ± sd over 5 seeds | per seed (1–5) |
|---|---|---|
| plain | 1.8521 ± 0.0108 | 1.8573 1.8448 1.8652 1.8555 1.8379 |
| softmax1 | 1.8481 ± 0.0091 | 1.8603 1.8553 1.8405 1.8403 1.8443 |

Paired by seed (same init and batches), plain − softmax1 = 0.004, 95% CI
[−0.015, 0.023]: **at batch 8 softmax1 makes no detectable difference.**
Train-probe CE is also level (1.372 vs 1.376). The seed spread is the
yardstick for any recipe change here: sd 0.01 over seeds 1–5, 0.02 over
6–10 (whose plain mean is 1.871), so compare changes paired by seed.

**Weight decay helps a little** (`... gpu <seed> <weight_decay>`,
2026-09-25). Every parameter shrinks by 1 − lr·wd each step, LayerNorm
and biases included. Plain, batch 8, held-out CE:

| wd | seeds 1–5, mean ± sd | paired gain over wd 0 |
|---|---|---|
| 0 | 1.8521 ± 0.0108 | — |
| 1e-4 | 1.8353 ± 0.0020 | 0.017 ± 0.005 |
| 2e-4 | 1.8334 ± 0.0073 | 0.019 ± 0.003 |
| 3e-4 | 1.8353 ± 0.0081 | 0.017 ± 0.008 |
| 5e-4 | 1.898 (seeds 1–2) | worse |
| 1e-3 and up | 1.96–2.75 (seeds 1–2) | underfits |

2e-4 was chosen on those seeds, so it was checked again on fresh seeds
6–10: 1.8405 vs 1.8708 without decay, a paired gain of **0.030** (95% CI
[0.007, 0.054]). Train-probe CE barely moves (1.392 vs 1.399), so the
overfitting gap narrows only from 0.47 to 0.45. Decay also improves the
fit to the training set a little, rather than just trading training
loss for held-out loss. It closes ~15–20% of the gap to the 7-gram
(1.655).

**Exempting LayerNorm and biases, and dropout, don't help** (`... gpu
<seed> <wd> weights [dropout]`, 2026-09-25). Plain, batch 8, held-out
CE, seeds 1 and 2:

| setting | seed 1 | seed 2 | mean | train-probe |
|---|---|---|---|---|
| no regularization | 1.8573 | 1.8448 | 1.8511 | 1.36 |
| decay 2e-4, all parameters | 1.8401 | 1.8361 | 1.8381 | 1.39 |
| decay 2e-4 / 5e-4 / 1e-3 / 2e-3, weights only | 1.8447 / 1.8466 / 1.8606 / 1.9194 | 1.8377 / 1.8549 / 1.8925 / 1.9940 | 1.8412 / 1.8508 / 1.8766 / 1.9567 | 1.35 / 1.40 / 1.50 / 1.64 |
| dropout 0.05 / 0.1 / 0.2 | 1.8676 / 1.9048 / 1.9825 | 1.8689 / 1.9020 / 1.9808 | 1.8683 / 1.9034 / 1.9817 | 1.47 / 1.55 / 1.67 |

Exempting LayerNorm and biases doesn't permit a larger rate: the
weights-only curve peaks at the same 2e-4 and underfits the same way above
it, so the decay that helps is on the weights (sparing the gains and
biases fits the training set slightly better, 1.35 vs 1.39, for the same
held-out CE). Dropout (on each block's
attention output and FFN hidden layer) is worse at every rate, and worse
the higher the rate. It does narrow the train/held-out gap (0.49 → 0.40
at 0.05), but by slowing the fit to the training set more than it helps
held-out: in 8000 steps the model hasn't fit enough for dropout's noise to
pay. A longer run is where it could; untested.

**At batch 1, softmax1 does beat plain** (`... <softmax1> 1 0.3 64000
600 gpu <seed>`, seeds 1–5, 64000 steps, ~2 min each uncontended,
2026-09-25). Held-out CE:

| batch 1 | mean ± sd over 5 seeds | per seed (1–5) | train-probe |
|---|---|---|---|
| plain | 1.8247 ± 0.0138 | 1.8072 1.8227 1.8174 1.8332 1.8429 | 1.333 |
| softmax1 | 1.8037 ± 0.0089 | 1.7989 1.8050 1.7923 1.8062 1.8162 | 1.271 |

Paired by seed, plain − softmax1 = **0.021**, 95% CI [0.011, 0.031],
positive on all 5 seeds. That's under half the single-seed 0.049 above:
the CPU references came from `attention_uniformity_check.rs`, which also
draws its eval windows from the training RNG, so its batch order differs
from this example's after step 1600, and its seed-1 plain (1.852) sits
~2 sd above this plain mean while its softmax1 (1.803) is typical. softmax1
also fits the training set better (0.06 lower train-probe), so at batch 1
it's an optimization gain, not a regularizer.

Batch 1 beats batch 8 at equal data, paired by seed: plain by 0.027 (95%
CI [−0.001, 0.055], 4 of 5 seeds) and softmax1 by 0.044 ([0.028, 0.061],
5 of 5). Batch 8 at lr 0.3 costs softmax1 more, which is why its lead
vanishes there; whether it returns at batch 8 with the higher lr below is
untested. Batch-1 softmax1 (1.804) also beats batch 8 with
the best weight decay (1.833). The gap to the 7-gram (1.655) is still
0.15. That batch-1 lead was batch 8's too-small lr, not the extra
updates: see the next paragraph.

**Batch 8 matches batch 1 once its lr is raised, at ~6× the data
throughput** (`... gpu <seed> 0 all 0 <warmup_windows>`, 2026-09-25). A
step costs nearly the same at any batch up to 8 (the step is bound by its
~145 dispatches, not arithmetic), so batch 8 was only losing on lr. Plain,
64000 windows, warmup = lr ramped linearly over the first 6400 windows:

| recipe | held-out, seeds 1–5 | train-probe | ms/step | s per run |
|---|---|---|---|---|
| batch 1, lr 0.3 | 1.8247 ± 0.0138 | 1.333 | 1.9 | ~120 |
| batch 8, lr 0.3 | 1.8521 ± 0.0108 | 1.372 | 2.5 | ~20 |
| batch 8, lr 0.6 + warmup | 1.8241 ± 0.0051 | 1.271 | 2.5 | ~20 |
| batch 8, lr 1.2 + warmup | 1.8187 ± 0.0091 | 1.244 | 2.5 | ~20 |

Paired by seed, batch 1 − batch 8 at lr 1.2 = 0.006, 95% CI [−0.014,
0.026]: no detectable difference (lr 1.2 was picked on seeds 1–2; 3–5 are
fresh). Warmup is what makes the higher lr work: without it lr 0.6 ends
worse than lr 0.3 (1.864, seeds 1–2) and lr 1.2 diverges. Warmup doesn't
help batch 1 itself (1.8095 vs 1.8150, seeds 1–2, one up and one down).

Larger batches hit plain SGD's stability ceiling, not the warmup's:
batch 16 reaches 1.8459 at lr 1.2 (seeds 1–2) and diverges at 1.7;
batch 32 reaches 1.923 at lr 1.2 and diverges at 2.4 and above. Every
divergence came after the warmup ended, which fits an lr above SGD's
curvature limit (~2/sharpness) that no batch size raises. Going past
batch 8 needs momentum, Adam or gradient clipping.

Step cost by batch (1500 steps, best of 2, CPU 34–92% loaded by other
sessions, so upper bounds): 1.86, 2.54, 4.01, 5.27, 8.22, 11.15 ms for
batch 1, 8, 16, 32, 64, 96, or 1.86 → 0.12 ms per window. Past batch 8
arithmetic starts to show, ~0.1 ms per extra window.

**Momentum removes the ceiling: batch 32 matches batch 1 at ~11× the
throughput** (`... <warmup_windows> 0.9`, 2026-09-25). Heavy-ball
momentum (v = μv + g, p −= lr·v, μ = 0.9, so the effective lr is 10·lr),
same 6400-window warmup. Plain, 64000 windows, held-out CE over seeds
1–5 (settings picked on seeds 1–2, confirmed on fresh seeds 3–5):

| recipe | held-out | train-probe | batch 1 − this, paired (95% CI) | ms/window |
|---|---|---|---|---|
| batch 1, SGD lr 0.3 | 1.8247 ± 0.0138 | 1.333 | — | 1.86 |
| batch 8, SGD lr 1.2 | 1.8187 ± 0.0091 | 1.244 | 0.006 [−0.014, 0.026] | 0.32 |
| batch 8, momentum, eff. lr 1.2 | 1.8088 ± 0.0089 | 1.223 | 0.016 [−0.001, 0.032] | 0.32 |
| batch 16, momentum, eff. lr 2.4 | 1.8110 ± 0.0115 | 1.238 | 0.014 [−0.004, 0.031] | 0.25 |
| batch 32, momentum, eff. lr 4.8 | 1.8175 ± 0.0155 | 1.278 | 0.007 [−0.016, 0.030] | 0.16 |

Nothing diverged with momentum, up to effective lr 9.6, where CE got
worse instead (batch 16: 1.851, batch 32: 1.841, seeds 1–2). Plain SGD
diverged at 1.7 (batch 16) and 2.4 (batch 32). Momentum at batch 8
beats plain SGD at batch 8 by 0.010 (95% CI [−0.009, 0.029]), so its
main value here is making larger batches work, not a better optimum.
A 64000-window run is now 2000 steps of ~5 ms, ~11 s of training.
Untested: momentum without warmup, other μ.

softmax1 under the tuned batch-8 SGD recipe (lr 1.2 + warmup, seeds
1–5): 1.8263 ± 0.0115 vs plain's 1.8187, paired plain − softmax1 =
−0.008, 95% CI [−0.020, 0.004]. Its batch-1 lead (0.021 at lr 0.3)
doesn't survive tuning the lr (tuned on plain).

**GNN over an extracted graph** — `gnn_byte_classification.rs`: the
original is-vowel probe, message-passing over `tiny_lm.rs`'s attention
graph on the tiny 172-byte corpus. Inconclusive there by a stated data
limit (only 34 labeled nodes); rerun and resolved at scale inside
`tiny_lm_corpus.rs` below.

**Predictive coding** — `predictive_coding.rs`: Whittington & Bogacz PC
as a biologically-motivated alternative to backprop, including genuinely
learned per-layer precision weighting. Took three attempts to stabilize
(raw MLE precision collapses toward zero on large early error, or
explodes toward infinity once error gets small, or NaNs on the
discontinuous jump between them) - fixed with a MAP estimate under a
Gamma prior (self-bounding, no clamp wall to hit) plus a smooth linear
ramp across the warm-up boundary (removes the jump itself).

**Continual learning / memory tiers** — `catastrophic_forgetting.rs`
demonstrates the problem and a bounded-replay-buffer mitigation in one
process; `checkpoint_save.rs`/`checkpoint_load.rs` prove cross-process
weight persistence (hand-rolled flat-file format, no serialization
crate); `memory_tier_save.rs`/`memory_tier_load.rs`/
`memory_tier_load_no_replay.rs`/`memory_tier_diff.rs` integrate the two
— persisting the replay buffer alongside the weights so a genuinely
separate process can resume replay-mitigated training, then diffing the
resulting weights layer-by-layer against a no-replay control.

Diffing weights instead of just loss surfaced a counter-intuitive
result: replay causes *more* overall L2 drift from the phase-1
baseline than no-replay (+12.2%), not less — only `LayerNorm` shows the
naive "protection" pattern. `memory_tier_sweep.rs` followed up by
sweeping `replay_prob` from a single loaded baseline: drift turns out
to be a threshold effect, not a dial (flat from 0.05 to 0.50; forgetting
drops sharply the moment *any* replay exists and barely improves with
more of it), and the one setting that does slash drift (`1.0`, training
on nothing but the 8-window snapshot) does it by failing to learn the
new corpus at all, not by protecting the old one. `memory_tier_multigen.rs`
extended this to a 4-corpus sequence with a half-life replay-curation
policy and found even actively-replayed corpora can still forget almost
completely once corpora start competing for a fixed 8-window budget,
and that learning each new task gets *harder* as generations
accumulate — replay isn't free.

All of the above ran at toy scale (`d_model=32`, 4-line nursery-rhyme
corpora). `memory_tier_multigen_scaled.rs` reran the same 4-corpus
multigen setup at `tiny_lm_corpus.rs`'s real scale (`d_model=128`,
Aesop's Fables) — but split one book into 4 contiguous quarters instead
of using distinct texts, and found essentially *no* forgetting: every
phase's loss on earlier "corpora" stayed flat or improved, since all
four quarters share the same vocabulary and register. Swapping in 4
genuinely different real texts (`memory_tier_multigen_diverse.rs`:
Aesop's Fables, Sherlock Holmes, *On the Origin of Species*, *Leaves of
Grass*) restored real forgetting — mild (~0.1-0.4 loss drift), nowhere
near the toy demo's ~8x blowup. A no-replay control
(`memory_tier_multigen_diverse_no_replay.rs`) then closed the loop:
budget=8 replay made no measurable difference at this scale, so the
mildness is saturation (real prose shares enough universal byte-level
structure that there's a floor to how much any phase can forget), not
replay mitigation — 8 fixed windows replayed 15% of the time is a
negligible fraction of a ~59KB corpus, unlike the toy case where it
covered a real share of the whole task.

That pointed at budget as the fix, not a fundamental mismatch, so
`memory_tier_multigen_diverse_bigbudget.rs` reran the same setup at
budget=128 (16x) and found a real, consistent ~0.15-0.28 loss
reduction across every retained corpus — the mechanism does transfer
to real scale, it just needed real coverage.
`memory_tier_multigen_diverse_budget256.rs` doubled budget again to
256 and found meaningfully smaller further gains (~0.03-0.065) —
diminishing returns, not a threshold or a continued linear win.

`memory_tier_multigen_diverse_consolidate.rs` then tried a completely
different mechanism on the same setup: no replay buffer at all, just
blending each phase's trained weights 70/30 back toward a snapshot
taken before that phase started (an EWC-lite stand-in — no
per-parameter importance weighting, just a uniform pull-back). It ties
budget=128 replay on one corpus, beats it on another, and matches it on
new-task cost — competitive retention from a mechanism with no stored
history or curation policy at all. No trait unifies this with replay:
the two don't share a call shape (replay hooks into training's
sampling; this only runs between phases) and each has exactly one
consumer so far, nowhere near the "2+ consumers" bar the `Optimizer`
trait needed before it was worth building.

`memory_tier_multigen_diverse_consolidate_fisher.rs` built the real
importance weighting that file's uniform pull-back explicitly named as
missing — per-segment empirical Fisher (average squared gradient over
the phase's own data) instead of one hand-picked alpha. Took three
rounds of actually finding bugs rather than accepting the first
plausible-sounding explanation for a negative result: the first
attempt looked *worse* than the uniform blend, diagnosed at the time as
per-parameter averaging diluting large layers' scores — true in part,
but a real counting bug (`counts` accumulated with `+=` every training
step instead of being set once) was doing most of the damage, silently
shrinking every Fisher value ~4000x and mistuning the blend strength by
four orders of magnitude. Once fixed, Fisher-weighting beat the uniform
blend on every metric, including new-task cost. Extending it further —
`TransformerBlockOut`'s fields were private, so the 4 transformer
blocks had fallen back to the old fixed alpha; made them `pub` (nothing
about reading a gradient needs write access to a layer's internals) —
gave each block its own genuine importance score instead of one shared
guess. Result changed again, honestly: retention improved further (the
best of any method in this whole line), but new-task cost got *worse*
than every other method tried, including no mitigation — the real
stability/plasticity trade-off, no longer averaged away. `block0`
(earliest layer) gets the strongest protection in every phase, a
specific finding that echoes — via a completely different method —
`tiny_lm_corpus.rs`'s own finding that attention behaves very
differently by depth.

`memory_tier_reconsolidation.rs` tested a prediction the Fisher work's
own EWC lineage raises but nothing here had checked: reconsolidation —
retrieving a consolidated memory briefly returns it to a labile state
before it re-stabilizes, rather than being a passive readout. Trained
on corpus A alone (last 500 steps = an "actively learning, nearly
converged" baseline), then on corpus B while replaying A at `p=0.15`,
bucketing every step's per-segment squared gradient by A-replay/B-fresh
× early/late phase 2. Mixed result, reported honestly: retrieving A
does elevate gradient above the phase-1 baseline in every segment but
`output_proj` — real lability, not a frozen readout — but that can't be
cleanly separated from ordinary interference (B-training has already
pulled shared weights away from A, so some corrective gradient on
replay is expected regardless of any reconsolidation-like effect). In
the embedding/attention-block layers, gradient grows late vs early for
*both* A-replay and B-fresh (B-fresh growing faster) — a general
within-phase trend, not a retrieval-specific one. But `final_ln`
(0.39×) and `output_proj` (0.43×) show A-replay's gradient shrinking
sharply late vs early while B-fresh stays flat or grows — repeated
retrieval specifically re-stabilizing the output head, the one place
matching real reconsolidation's "windows narrow with repetition"
signature. `block0` again has the largest raw gradient of any segment
under every condition — a third independent metric landing on the same
earliest-layer-most-sensitive finding. Left open: the natural control
(continue phase 2 on corpus A alone, no B, same step count) to check
whether A's gradient decays on the same schedule from continued
optimization alone — not run here.

Ran that control (`memory_tier_reconsolidation_control.rs`): identical
phase 1, then phase 2 continues on corpus A alone, no B, no replay. The
embedding/attention-block growth trend survives unchanged (1.15–1.63×,
if anything larger) — confirming that part is generic optimization
dynamics, unrelated to retrieval. But `final_ln`/`output_proj` invert:
under uninterrupted A training they hold flat or grow slightly (1.12×,
0.93×) instead of shrinking, unlike the sharp decay under interference
(0.39×, 0.43×). That isolates the effect: `final_ln`'s reconsolidation-
like re-stabilization is genuinely retrieval-specific, not continued-
training artifact. `output_proj` is partial — the control's own mild
decay (0.93×) explains some but not most of the interference run's
shrinkage (0.43×), so a real, smaller retrieval-specific component
survives there too. The open question is answered: the output-facing
layers show genuine reconsolidation-like lability, not confounded
generic convergence.

Sharpened the early/late split into what "windows narrow with
repetition" actually claims: `memory_tier_reconsolidation_repetition.rs`
keys each replay by which of the 8 buffer windows was drawn and how
many times THAT window has itself been retrieved so far, pooled into
one dose-response curve per repetition count rather than by phase-2
wall-clock position. The curve confirms and sharpens the finding:
`final_ln` drops from 0.165 (1st replay) to 0.027 (21st+) and
`output_proj` from 0.613 to 0.171 — both largely-monotonic declines
substantially resolved by ~5 repetitions, at an average phase-2 step of
only ~320 of 4000, well before elapsed time could explain it.
`token_emb` and all 4 blocks show no comparable trend — noisy, flat, or
mildly rising by the 21+ bucket — consistent with their behavior being
the generic within-phase growth already isolated by the control. The
more direct confirmation: re-stabilization tracks repetition count of
the specific retrieved memory, not just elapsed training time.

A different brain analog from the same line: Josselyn & Frankland's
neurogenesis-forgetting hypothesis — new hippocampal neurons
integrating into an existing circuit are proposed to *cause* forgetting
of memories that circuit already held, independent of time or how much
else gets learned (the leading explanation for infantile amnesia).
Adult neurogenesis specifically occurs in the dentate gyrus, an early
hippocampal stage — motivating `memory_tier_neurogenesis.rs` to
reinitialize `block0` specifically (also the segment Fisher-
consolidation and reconsolidation independently found most sensitive).
Trained once on corpus A, snapshotted, then forked into two phase-2
branches seeing the *identical* corpus-B training sequence — one
control, one with `block0` reinitialized to fresh random weights
first.

A real but modest effect, with a real confound. Neither branch
actually forgets A outright at this scale (matching the established
saturation finding — real prose shares enough universal byte-level
structure that there's a floor on how much any phase can forget); both
branches' loss on A slightly *improves* after phase 2 (control:
2.382→2.334; neurogenesis: 2.382→2.358). But the improvement is
roughly half as large under reinitialization (−0.048 vs −0.024), and
loss on B is *worse* too (2.127 vs 2.168) — reinitializing block0 cost
both retention and new-task learning simultaneously, not a clean
stability/plasticity trade-off, because the reinitialized block has to
relearn its role from random init in the same step budget rather than
arriving "unencumbered." Honest caveat: reinitializing an entire
existing block is a harsher "wipe and regrow" than the biological
analog — real adult-born neurons integrate *alongside* existing,
undisturbed synapses, they don't erase them. This conflates "new
growth disrupts old memories" with "a block relearning from scratch is
just a worse starting point." A more faithful test would add new
capacity (an extra head or block) without erasing what's already
there — not run here.

Ran that more faithful test (`memory_tier_neurogenesis_additive.rs`):
blocks 0-3 stay byte-for-byte untouched, phase 2 simply appends a fresh
block4, and every parameter — old and new — stays trainable. The
effect disappears: retention on A is statistically indistinguishable
from control (−0.048 control vs −0.050 additive), and learning B is
also slightly better with the extra block (2.118 vs 2.127). Side by
side, the two experiments say something more precise than either
alone: the reinit run's real cost wasn't caused by the mere presence of
new, untrained capacity — it was caused by the *destructive* act of
erasing already-useful weights and forcing that segment to relearn
from scratch. Made faithful to what real neurogenesis actually does
(add without erasing), this system shows no measurable neurogenesis-
forgetting effect at all — a genuine, clarifying negative result that
locates the disruptive ingredient precisely, not a failed replication.

**Differentiable reasoning / neural theorem proving** — surveyed against
recent NTP (neural theorem prover) literature, testing whether making
`tiny_lm_corpus.rs`'s rule-chaining engine's combination rule *learned*
instead of hand-written fixes its "hollow recall" finding below (a
majority-vote rule that beat a decision tree on accuracy but never once
caught a true positive on several categories).

`differentiable_forward_chaining.rs` rebuilds that rule on a self-
contained relation graph (raw byte co-occurrence, not attention/
embeddings/clustering) and learns the combination weights instead of an
untrained OR. Fixes real recall on `is-uppercase` and weakly on
`is-digit`, but stays at exactly zero on every vowel category — not
noise: English alternates consonants and vowels, so a vowel's textual
neighbors are mostly consonants, and no weighting scheme can learn
signal that isn't there. Separates two causes of hollow recall that
looked identical under the hard rule: an untrained combination
(fixable by learning) vs. evidence with no real predictive signal for
the target (not fixable by any weighting — garbage in, garbage out).

`differentiable_backward_chaining.rs` goes further — a real NTP-style
prover (OR-module = existential search over a candidate entity,
AND-module = product of two atoms' scores) on a toy 12-entity kinship
KB: one relation (`parent`), one rule
(`grandparent(X,Z) :- parent(X,Y), parent(Y,Z)`). Phases 1-2: embeddings
trained on `parent` facts alone achieve perfect, stable-across-4-seeds
zero-shot compositional generalization to `grandparent` — genuine 2-hop
reasoning from 1-hop training, with the correct bridge entity printed
for every true pair. Phase 3 asks the harder question: can `parent` be
learned from *only* 2-hop supervision, never shown directly? That
needed a real differentiable OR-module, so `Tape::max_last_axis` was
added (gradient-checked like every other op here). Result: the trained
objective converges fine, but the recovered `parent` relation is a
complete miss (0% precision/recall on every inferred fact) — the model
finds a different, self-consistent-but-wrong bridge structure, matching
the literature's own documented local-minima failure mode for greedy
max-pooling. Phase 4 tried the literature's own named fix (a
softmax-weighted "beam" over candidates instead of hard max) — it
didn't fix it, and hallucinated *more* spurious relations, not fewer,
distinguishing an optimization pathology (what beam supervision
actually fixes) from an information-theoretic one (what this task
actually has — existence-only labels never specify *which* y works).

`differentiable_backward_chaining_dense.rs` tested whether that
identifiability gap was specific to the toy KB's clean symmetry (26
entities, deliberately irregular branching, no two branches shaped the
same) — it wasn't: still 0% recovery, and greedy max-pooling now failed
to even solve the trained objective at the larger scale, an
optimization artifact isolated by retuning the learning rate (which
fixed the trained objective cleanly while recovery stayed at exactly
0%). A temperature sweep on the softmax-weighted variant found one real
but noisy exception — a small nonzero recovery signal at specific
temperatures (best f1=0.137), not a smooth function of temperature —
consistent with genuinely limited recoverable information rather than
a tunable knob.

**`tiny_lm_corpus.rs`** is the project's largest and most-iterated file
by far — a byte-level LM plus an extended symbolic/KR&R (knowledge
representation & reasoning) investigation built on top of it:
k-means clustering of trained embeddings, an attention-derived
relational graph, an embedding-geometry nearest-neighbor graph, a GNN
that message-passes over the attention graph, a from-scratch decision
tree, a hand-written rule-chaining inference engine, a formal
consistency check across all three extraction methods, a
hierarchy/lattice check on compound categories, and a multi-hop
knowledge-graph query engine — each with cross-seed stability testing,
each reported honestly including where the methods disagree or fail.
`git log --oneline -- examples/tiny_lm_corpus.rs` is the actual
narrative; a few of the sharper findings:

- Attention behaves very differently by depth: early layers lean on a
  seed-arbitrary frequency sink, deep layers lean on structural
  landmarks (e.g. `'\n'` as a paragraph boundary).
- Embedding geometry finds real structure raw clustering misses
  entirely — a clean punctuation split by grammatical role
  (`{,;:}` vs `{.?!}`), and a compositional "uppercase AND vowel"
  direction that no coarser extraction method (clustering, a decision
  tree, rule-chaining) manages to preserve, even though it demonstrably
  exists in the raw embeddings.
- A hand-written logical rule (majority vote over a combined relation
  graph) beats a from-scratch decision tree in every tested category —
  but precision/recall reveals several of those "wins" were hollow
  (zero recall, i.e. the rule never once caught a true positive).

Two more findings worth surfacing: a 2x model-size scale-up at a fixed
step count measured *worse* on both held-out loss and structural
fragmentation than the smaller model - a compute-optimal-scaling
confound, confirmed by rerunning it with proportionally more steps too,
which closed almost the entire gap (and revealed capacity doesn't
clearly help either, once fairly trained - it just catches up to
parity, at ~20x the wall-clock cost). And a real design gap found by
checking which architecture parts are genuinely swappable: every
layer's `apply_grad` was hardcoded to `Sgd` even though Adam had grown
5 real consumers, past this project's own stated bar for promoting an
abstraction - fixed with an `Optimizer` trait unifying both behind
`Linear::apply_grad_with`.

`Linear`/`Embedding`/`LayerNorm::forward()` used to leaf a fresh copy of
their parameters on every call, so reusing one layer across multiple
`forward()` calls in a single tape (RNN/SSM-style weight tying) silently
dropped gradient contributions — real enough that `ssm_recall.rs` had to
work around it by hand. Fixed: each layer now has a `forward_shared`
taking pre-leafed `Var`s for tied reuse, and `apply_grad` panics instead
of silently computing a partial gradient if it's ever called with a
stale output from before a later `forward()` call.

## Related work

Literature (surveyed 2026-09-23) and a sibling project, each with what
it means here.

**QK-norm**
- Henry et al. 2020, *Query-Key Normalization for Transformers*
  ([arXiv 2010.04245](https://arxiv.org/abs/2010.04245)): L2-normalize Q/K, then
  *multiply* by a learnable scale in place of 1/√d. This project's first
  version kept 1/√d and dropped the learnable scale; that's why attention
  went uniform.
- Loshchilov et al. 2024, *nGPT* ([arXiv 2410.01131](https://arxiv.org/abs/2410.01131)):
  unit-normalized q·k has variance 1/d_k, so the softmax factor should be
  √d_k, plus a learnable per-head s_qk. An independent diagnosis of the
  same bug.
- Dehghani et al. 2023, *ViT-22B* ([arXiv 2302.05442](https://arxiv.org/abs/2302.05442)):
  LayerNorm on Q/K, 1/√d kept. The form `TransformerBlock::with_qk_norm`
  uses. Per-head beats layer-wise Q/K norm at suppressing logit growth.
- Wortsman et al. 2023, *Small-scale proxies for large-scale Transformer
  training instabilities* ([arXiv 2309.14322](https://arxiv.org/abs/2309.14322)):
  attention-logit growth reproduces in small models at high LR, and
  qk-layernorm fixes it there. Relevant to lr 0.3 SGD here.
- Zhai et al. 2023, σReparam ([arXiv 2303.06296](https://arxiv.org/abs/2303.06296)):
  attention *entropy collapse* destabilizes training. The L2 bug hit the
  opposite pole (maximum entropy, heads as mean-pools).

**softmax1, attention sinks, abstention**
- Miller 2023, *Attention Is Off By One*
  ([blog](https://www.evanmiller.org/attention-is-off-by-one.html)): softmax1.
  Follow-up tests found only small outlier reductions.
- Bondarenko et al. 2023, *Quantizable Transformers* ([arXiv 2306.12929](https://arxiv.org/abs/2306.12929)):
  outliers come from heads learning no-ops. The fixes are clipped softmax
  and gated attention.
- Gu et al., ICLR 2025, *When Attention Sink Emerges* ([arXiv 2410.10781](https://arxiv.org/abs/2410.10781)):
  sinks stem from softmax normalization. Sigmoid attention without
  normalization removes them up to 1B params.
- Qiu et al. 2025, *Gated Attention for LLMs* (NeurIPS 2025 best paper,
  [arXiv 2505.06708](https://arxiv.org/abs/2505.06708)): an elementwise sigmoid
  gate after attention removes sinks and improves loss. The form
  `with_attn_gate` uses.
- Zuhri et al. 2025, *Softpick* ([arXiv 2504.20966](https://arxiv.org/abs/2504.20966)):
  rectified, not-sum-to-one softmax. 0% sink rate at 340M/1.8B.
- Wang 2026, *Abstention and Noise Filtering* ([arXiv 2609.22005](https://arxiv.org/abs/2609.22005),
  preprint): abstention (off-by-one, sink logit) is worth ~0.019 nats at
  10M params and shrinks with scale. Gated noise filtering grows with
  scale, and sink logit + gate is best. At this project's ~0.6M params,
  softmax1-style abstention should help slightly, if at all.
- Rizwan et al. 2026 ([arXiv 2609.08574](https://arxiv.org/abs/2609.08574), preprint):
  at 1M params gated attention never learned sparsity (mean gate 0.69 vs
  0.12 at scale). Sinks were driven by the training objective as much as
  the architecture. A caution for mechanisms tested at this scale.

**Continual learning / memory tiers**
- McClelland, McNaughton & O'Reilly 1995 (complementary learning
  systems); Kirkpatrick et al. 2017 (EWC); Zenke et al. 2017 (Synaptic
  Intelligence): the lineage of the replay and Fisher-consolidation
  experiments.
- Arani et al. 2022, *CLS-ER* ([arXiv 2201.12604](https://arxiv.org/abs/2201.12604)):
  fast (plastic) and slow (stable) EMA copies of the working model plus a
  consistency loss. A concrete, published version of the untried
  "two-speed" tier idea.
- Replay + EWC together is standard in continual language learning
  (e.g. Distill-and-Replay, [COLING 2020](https://aclanthology.org/2020.coling-main.318.pdf)).
  Never tried together here.
- *EWC Done Right* 2026 ([arXiv 2603.18596](https://arxiv.org/abs/2603.18596)):
  Fisher-based importance can suffer gradient vanishing and misallocate
  protection. A candidate factor in the Fisher run's new-task cost,
  unverified.

**GPU step layout**
- Karpathy, *llm.c* ([GitHub](https://github.com/karpathy/llm.c)): the CPU
  reference (`train_gpt2.c`) indexes heads inside the fused `(B, T, 3C)`
  QKV by stride, with no permute. The CUDA path (`llmc/attention.cuh`)
  permutes to `(B, NH, T, HS)`, runs batched matmuls and a softmax, then
  unpermutes. The CPU batched-heads plan mirrors the CUDA pipeline
  (`docs/gpu_step_design.md`, "Survey: alternatives to full batched heads").

**Sibling project: `humble-cortex`** (`../humble-cortex`, predictive
coding in Rust). Its 95 checks overlap here in three places. First, every
model is benchmarked against the simplest baseline for the task
(persistence, OLS, backoff n-gram, kNN). scratchtape's tiny LMs had none
until `ngram_baseline.rs`, which the current model doesn't clearly beat. Second, partial replay + consolidation
compounded (37.6% / 41.2% alone, 54.0% combined, over 3 repeats), which is
untested here. Third, retention is normalized by how much was learned,
because raw forgetting deltas hid a floor effect.

## Open threads

- **Replay + Fisher consolidation together** — humble-cortex found them
  synergistic. Here each was only tried alone, and Fisher's new-task cost
  might be offset by replay.
- **CLS-ER-style two-speed memory** — fast and slow EMA copies of the
  model with a consistency loss (Arani et al. 2022).
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
  Vulkan, but DX12 has ~7× lower per-call overhead (128×128 round trip
  1.1 ms vs 7.8 ms), so `gpu.rs` now defaults to DX12 on Windows
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
  wgpu 23 build at `a37604c` stalled the same way (after 31), so the port
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

## A note on the name

The crate is `scratchtape`: from-scratch, and literally built around a
`Tape` struct at its center — not a metaphor.
