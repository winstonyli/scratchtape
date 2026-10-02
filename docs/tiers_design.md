# Tiered AI: first cross-tier experiments

Direction and tier map: README, "Direction: tiered AI". Related work: README, "Tiered and neuro-symbolic
architectures". Status: design, 2026-10-02; nothing here is built or measured yet.

## What exists and what does not

- GPU tier: the byte-level transformer and its device-resident step. Settled recipe on `novels6`: d = 256,
  dropout 0.1, K = 1; held-out CE **1.2408** nats/byte at 8M windows (7-gram 1.3567). Checkpoint:
  `runs/nov_big_k1_d0.1_8m_m0.ckpt` (flat parameters, `flatten_all` order).
- CPU/KR&R tier: **no symbolic checker over byte sequences exists.** `tiny_lm_corpus.rs` extracts structure *from*
  the trained network's internals (probes, embedding-NN graphs, rule chaining over them); `differentiable_reasoning/`
  works on toy logic problems. Neither runs against the LM's output.
- Working-memory tier (registers/cache): nothing built.
- Long-term tier (RAM/disk): `examples/memory_tiers/` stores replay windows and weights, i.e. memory *for
  training*; no tier the model reads at inference.

## Common measurement

Every experiment is scored the same way as the rest of the project: mean CE (nats/byte) over the non-overlapping
64-byte windows of the `novels6` held-out split (7873 windows), against the checkpoint's own number (1.2408),
so a tier "helps" only if it lowers that number. A new example, `examples/tiny_lm/tier_eval.rs`, loads a
checkpoint, runs the held-out windows on the GPU and reads back, per position, the logits and the final residual
stream (`model_forward`'s last block output; no API change). The tiers are then applied on the CPU to those
arrays. **Gate 0:** with no tier applied it must reproduce 1.2408 to 1e-4, or nothing after it counts.

## Experiment A: a symbolic lexicon tier (CPU, propose-and-check)

The knowledge: a word list built from the training text (a trie of the words seen), as symbolic structure the
network does not store explicitly. At each position inside a word, the checker knows which next bytes keep the
prefix a prefix of some known word, or end a known word.

- Combine as a mixture so unseen words (names, held-out vocabulary) are not given probability 0:
  `p'(b) = (1 - eps) * renorm(p restricted to checker-allowed bytes)(b) + eps * p(b)`; sweep eps.
- Also report the plain "propose-and-check" view: of the bytes the LM ranks top-1, what fraction does the
  checker reject, and does rejecting and taking the next-best fix them? (Accuracy of top-1 before/after.)
- Honest risks: held-out words absent from the lexicon (Dracula and Moby-Dick names) push the optimum eps up; a
  win that comes only from the training lexicon covering held-out text of the *same books* is a leak-shaped
  result, so the lexicon is built from the **train split only** and the split is by position within each book.
- Cost: one extra CPU pass; readback of ~129M logits in chunks. No training.

## Experiment B: an external memory tier (RAM, kNN-LM-style)

Key = final residual stream at a position, value = the next byte; datastore built from training windows run
through the same frozen model (Khandelwal et al. 2020, Wu et al. 2022). At a held-out position take the k nearest
keys, form a distribution over their next bytes (softmax of negative distance), and interpolate:
`p'(b) = (1 - lambda) * p(b) + lambda * p_knn(b)`; sweep k, lambda and the datastore size.

- Datastore size: 4.5M train positions x d = 256 is 4.6 GB in f32 (free RAM 38 GB at last check); start with a
  1M-position subsample (1 GB) and a subsample of held-out windows for the search (brute force: 1M x 256 MACs per
  query), scored against the base CE **on the same subsample**.
- Honest risks: the datastore comes from the training split, which the model has fit (train-probe 1.1172 vs
  held-out 1.2408), so its keys are overfit; the published gains were on models trained once on far more data.
  A negative result is a real result and gets recorded.

## Order

0. `tier_eval.rs` with Gate 0 (needs the GPU for a forward pass only; the d = 384 run is using it, shared lease).
1. Experiment A (smallest, most distinctly a CPU/symbolic tier).
2. Experiment B.
3. Only then: how the tiers interact (a tier gating another, working-memory contents), which needs A and B's
   numbers to be worth designing.

Open questions for the user: is the lexicon a reasonable first "KR&R" content, or should the CPU tier hold
something else (rules the LM must satisfy, e.g. quote/bracket matching)? Is the final residual stream the right
"working memory" to expose to the long-term tier?
