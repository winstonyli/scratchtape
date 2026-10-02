# Tiered AI: first cross-tier experiments

Direction and tier map: README, "Direction: tiered AI". Related work: README, "Tiered and neuro-symbolic
architectures". Status: step 0 and Experiment A built and measured (2026-10-02); Experiment B measured on a tune/test split.

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

## Results so far (2026-10-02)

`examples/tiny_lm/tier_eval.rs` (plug-and-play: a `Tier` trait, specs `tier=<part>+<part>`, `tap=<block>`
selects the hidden state tiers receive). User decision: the word lexicon is a reasonable first CPU tier and the
final residual stream a reasonable first working-memory tap, but both stay swappable.

```
tier_eval runs/nov_big_k1_d0.1_8m_m0.ckpt expect=1.2408 tier=lexicon:<eps> ...
```

- **Gate 0 passed:** CE from the logits 1.2408 = the device's row losses = the recorded number; lexicon with
  eps = 1 (no masking) reproduces it exactly (+0.0000).
- **Experiment A (lexicon, 28728 train-split words; the whole run takes ~20 s):** held-out CE by eps
  (model alone 1.2408): 0.005 1.2420, 0.02 1.2389, 0.05 1.2369, 0.1 1.2357, 0.15 1.2351, 0.2 1.2348,
  **0.3 1.2347**, 0.5 1.2354. Best **-0.0061** nats/byte.
- The tier constrains 360476 of 503808 positions (those inside a word after a boundary). The model's top-1 byte
  is rejected at only 1676 of them (0.5%), and the next-best allowed byte is then right 1065 times (64%).
- Reading: the model already spells almost everything, so a train-split lexicon adds little; the high optimal eps
  (0.3) means a lot of held-out words are outside the lexicon (names in the held-out tails), not that the mask
  is wrong. Held-out OOV (measured, printed by `tier_eval` at every run): 1294 of 92358 word
  tokens (1.40%), covering 1.97% of held-out bytes, are absent from the train lexicon. That is the leak-free cost
  of the mask and explains the high optimum eps. Not measured: a lexicon that also scores word frequency or
  context (a bigram tier) rather than only masking.

- **Experiment B (kNN memory, first sweep; log `runs/tier_knn_1m.log`):** 960000 keys (15000 evenly spaced train
  windows, 21% of train positions; key = final residual stream, block 3 output), scored on every 16th held-out
  window (492 windows, 31488 positions; model alone **1.2565** on this subsample, not 1.2408). Grid k in {8, 32},
  lambda in {0.05..0.4}, temp in {10, 30, 100} (squared-L2 distances: nearest ~41, 32nd ~70). Best **k=32,
  lambda=0.2, temp=30: 1.2462 (-0.0103)**; k=8 best -0.0065 (lambda 0.1, temp 30). lambda 0.4 hurts everywhere
  except k=32/temp 30 (flat). Larger k was better, and k=32 is the grid's edge (max supported 64), so the optimum is
  not bracketed. The gain is ~1.7x the lexicon's, from a different source (train-text continuations of similar
  states), but note: a single subsample, hyperparameters picked on the same positions they are scored on (a small
  optimistic bias), a 21% datastore (more keys probably help), and nothing yet on a disjoint tuning split.

- **Experiment B, tune/test split (`scripts/tier_knn_sweep.sh`; logs `runs/tier_knn_2m_o0.log`, `_o8.log`):**
  1.92M keys (30000 train windows, 43% of train positions), k in {32, 64}, lambda in {0.1, 0.2, 0.3}, temp in
  {20, 30, 50}, tuned on every 16th held-out window from offset 0 (model alone 1.2565) and tested on the disjoint
  offset 8 (1.2524). Best on tune **k=64, lambda 0.3, temp 30: -0.0168**; the same setting on test **-0.0181**,
  so the gain is not tuning bias. More keys and larger k both help (k=32 at 1M keys: -0.0103; at 1.92M keys:
  -0.0147), and the optimum is still at the grid edge (k cap 64, lambda 0.3). **Stacked** `lexicon:0.3+knn:64:0.2:30`:
  tune -0.0215, test **-0.0212** (lexicon alone -0.0076 / -0.0053, knn alone at those settings -0.0160 / -0.0169):
  the tiers largely add. Cost: ~25 min per 31488-position subsample at 4 threads on a shared CPU (brute-force
  search, 1.92M x 256 per query), so a full-held-out run is out of reach without an index or the GPU.

## Order

0. `tier_eval.rs` with Gate 0 (needs the GPU for a forward pass only; the d = 384 run is using it, shared lease).
1. Experiment A (smallest, most distinctly a CPU/symbolic tier).
2. Experiment B.
3. Only then: how the tiers interact (a tier gating another, working-memory contents), which needs A and B's
   numbers to be worth designing.

Open questions for the user: is the lexicon a reasonable first "KR&R" content, or should the CPU tier hold
something else (rules the LM must satisfy, e.g. quote/bracket matching)? Is the final residual stream the right
"working memory" to expose to the long-term tier?
