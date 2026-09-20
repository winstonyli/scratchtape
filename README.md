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
cargo test --release              # 20 gradient-check / correctness tests
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
| `gpu.rs` / `matmul.wgsl` | A wgpu compute-shader matmul kernel — verified correct, but not wired into the autodiff path; see the persistent-GPU-backend note below. |

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

## Open threads

- **A persistent, on-device GPU training backend** — the current GPU
  kernel is correct but pays a full host round-trip per call, which is
  why it doesn't win at this model's scale. Scoped as its own
  multi-day project, not a small addition.

## A note on the name

The crate is `scratchtape`: from-scratch, and literally built around a
`Tape` struct at its center — not a metaphor.
