# Examples: what each one is and what it measured

Moved verbatim from the README (2026-10-14). The README keeps a one-line index.

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
own methodology already flagged as unreliable ([43366bd] found the
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
`tiny_lm_corpus.rs` result recorded between `22437d0` and this change
came from the L2 softmax1+QK-norm model. For context, recent work puts
softmax1's upside at small scale in the hundredths of a nat (abstention
~0.019 nats at 10M params, shrinking with scale; Wang 2026). Gated or
sink-logit variants (Qwen's gated attention, NeurIPS 2025; a learnable
phantom key) get abstention without unbounded logits. One seed.

**Correction: the runaway was a `Tape::softmax1` bug, present since
`6f22b36`.** Writing a finite-difference test for a new sink-logit op
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
(Training was: long runs with dropout later cut the gap to 0.02; see
"Long runs" below.)
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
pay. In longer runs it does, and is the biggest gain found (below,
"Long runs").

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

**Long runs: dropout closes most of the gap to the 7-gram**
(`... 32 0.48 <windows> 600 gpu <seed> <wd> all <dropout> 6400 0.9`,
2026-09-25). With batch 32 + momentum a run costs ~5.6 ms per 32
windows, so a million windows (~150 passes over the ~214 KB of training
text) takes ~3 minutes. More steps alone don't help: without dropout,
held-out bottoms out at 64–88k windows (1.809, ~20 passes) and climbs to
2.16 by 256k while train-probe falls to 0.82, pure memorization. Weight
decay only delays it. Dropout (on each block's attention output and FFN
hidden layer) does what it didn't at 64k windows. Held-out CE, mean of
seeds 1–3 (best = the lowest of the run's evaluations, chosen on
held-out, so slightly optimistic; final is honest):

| regularization | windows | best | final | final train-probe |
|---|---|---|---|---|
| none | 256k | 1.8088 | 2.1628 | 0.82 |
| wd 2e-4 | 256k | 1.8082 | 2.0827 | 0.86 |
| wd 1e-3 | 256k | 1.7864 | 1.8561 | 1.05 |
| dropout 0.1 | 256k | 1.7156 | 1.7530 | 0.99 |
| dropout 0.1 + wd 2e-4 | 256k | 1.7112 | 1.7430 | 1.02 |
| dropout 0.2 + wd 2e-4 | 512k | 1.6851 | 1.7052 | 1.02 |
| dropout 0.3 + wd 2e-4 | 512k | 1.6803 | 1.6918 | 1.11 |
| dropout 0.3 + wd 2e-4 | 1M | 1.6633 | 1.6864 | 1.04 |
| dropout 0.4 + wd 2e-4 | 1M | 1.6702 | 1.6769 | 1.12 |
| Kneser-Ney 7-gram | — | 1.655 | | |

Dropout 0.1's best beats no regularization by 0.093 on every seed (95%
CI [0.049, 0.137]). The gap to the 7-gram falls from 0.15 to 0.02
(final) or 0.008 (best), but it isn't crossed: 512k → 1M windows gains
only ~0.005–0.02, a plateau. The final value sits 0.01–0.02 above the
run's best, the noise of a constant lr, so an lr decay at the end was the
next lever. (Tried 2026-09-30, below: it made held-out worse. The 7-gram
is crossed by averaging models instead.)

**Update (2026-09-30): the 7-gram is crossed, by averaging K models.**
`fused_models_check` trains K = 4 models in the same launches from one
init, averages their parameters every 100 steps (local SGD) and adds
mutual distillation with α = 0.1:

```
fused_models_check <name> 4 32 0.48 1024000 <seed> 0.0002 0.3 6400 0.9 0.1 0 100 1
```

(args: name K batch lr windows seed wd dropout warmup momentum α
α-ramp sync-steps shared-init). Final held-out CE on aesop, three seeds:
**1.610** (1.605–1.615), against the 7-gram's 1.655 and 1.686 for one
model of the earlier recipe. Without α: 1.651; a single model with the
same data per step (batch 128): 1.755. Costs 3.4 ms/step per model, all
four in 13.5 ms. Sync period H = 10 / 30 / 100 gave 1.634 / 1.621 / 1.610
(seed 1 for the first two). Experiments and caveats in
`docs/fusion_results.md`.
It does not transfer cleanly to a small corpus: on `sherlock_holmes`
(54 KB train, 93 held-out windows) at 256k windows the recipe overfits
(single model: best 1.85 mid-run, final 2.04). Final held-out there:
single 2.038, averaged 2.065, averaged + α 1.883, ensemble of 4
independent models 1.727, against a 7-gram-family 1.695. α helps, plain
averaging doesn't, and the windows count wasn't tuned for that corpus.
At 64k windows on the same corpus averaging does help (1.805 vs 1.877
for one model; α adds 0.007), so the effect needs a run length matched
to the corpus. A final lr decay did not help on aesop (1.623–1.632 vs
1.610: it overfits more), and a 4× larger model (d 256, d_ff 512; lr 0.48
diverges, so lr 0.05) did worse at the same 1M windows: 1.675 averaged,
1.81 single, against 1.610 for the small one; 512k windows gave 1.664,
256k 1.762 and dropout 0.5 1.794, so no setting tried rescued it. Two
averaged groups of 4, ensembled, reach 1.568, short of 4 independent
models' 1.548: averaging and ensembling don't stack. On all four bundled
books (373 KB train, the last 10% of each held out, 7-gram 1.628) the
recipe carries: one model 1.637, K = 4 averaged + α = 0.1 **1.579** at 1M
windows and 1.568 at 2M, while the d = 256 model gets 1.642 single and
1.597 averaged.

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
