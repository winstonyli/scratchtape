# Tiered AI: first cross-tier experiments

Direction and tier map: README, "Direction: tiered AI". Related work: README, "Tiered and neuro-symbolic
architectures". Status: step 0 and Experiment A built and measured (2026-10-02); Experiments A and B and a second CPU tier (word bigram) measured on the full held-out split.

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

- **Experiment B on the GPU, full datastore and full held-out split.** The brute-force search moved to the
  device (`src/gpu_step/knn.rs`: keys in 16384-key tiles, one matmul per tile giving transposed distances, a
  top-k kernel keeping the `KMAX` = 128 nearest per query; unit test against a CPU brute force). The same 1.92M-key
  query set that took ~25 min on the CPU takes 35 s (reproduces -0.0168 exactly). With all 4,534,720 train positions
  as keys, tuned on offset 0 of stride 16 (`knn:128:0.4:25` best: -0.0265 vs -0.0168 at 1.92M keys/k=64), the
  **full held-out split** (7872 windows, 503808 positions; `scripts/tier_full.sh`, `runs/tier_full.log`;
  Gate 0 passed at 1.2408):

  | Tier | CE | vs model |
  |---|---|---|
  | model alone | 1.2408 | |
  | lexicon:0.3 | 1.2347 | -0.0061 |
  | knn:32:0.4:25 | 1.2204 | -0.0205 |
  | knn:64:0.4:25 | 1.2151 | -0.0257 |
  | knn:128:0.3:25 | 1.2143 | -0.0265 |
  | knn:128:0.4:25 | 1.2126 | -0.0283 |
  | lexicon:0.3 + knn:128:0.3:25 | 1.2110 | -0.0298 |
  | lexicon:0.3 + knn:128:0.4:25 | **1.2099** | **-0.0309** |

  k is still improving at the cap (32 -> 64 -> 128: -0.0205, -0.0257, -0.0283), so the optimum is not bracketed.
  Together the two CPU/RAM tiers move the model from 0.116 under the 7-gram to 0.147 under it.
- **Which hidden state is the key (`runs/tier_tap*.log`; stride 16, offset 0, k = 128, temp re-tuned per tap
  because distance scales differ):** block 1 output -0.0117, block 2 -0.0241, block 3 (final residual stream)
  -0.0265. The gain grows with depth, so the memory is useful because the late representation is close to the
  prediction, not only because nearby text is similar. Not separated: how much of the gain is the same
  train-text continuation the model already fit (train-probe 1.1172 vs held-out 1.2408), which a datastore
  built from data the model never trained on would answer.

- **Second CPU tier: word statistics (`words:<order>:<lambda>`; `runs/tier_words.log`, full held-out).** Inside a word,
  the train words extending the current prefix give a next-letter / end-of-word distribution (order 1: the words that
  followed the previous word, backing off to all words by frequency). Order 0 (unigram frequency) is no better than the
  lexicon (best -0.0063 at lambda 0.1; lambda 0.3 hurts). **Order 1 (word bigram): -0.0158 at lambda 0.25-0.3**
  (1.2250), 2.6x the lexicon; its bigram list applied at 259061 of the 360476 in-word positions. A 1.4%-of-tokens OOV
  rate and the model already knowing spelling limit the lexicon; context (the previous word) is what the model's
  CPU-side counterpart adds.
- **Memory contents: unseen vs trained-on, and same document vs other documents** (`runs/tier_unseen16.log`,
  `runs/tier_parts.log`; k = 128, best of a small lambda/temp grid; scored on a subsample, so compare rows within a block):

  | Memory | keys | gain (stride 16, all books; model 1.2565) |
  |---|---|---|
  | train windows, equal size | 472k | -0.0077 |
  | held-out windows not scored (same books, unseen by the model) | 472k | **-0.0366** |
  | both | 945k | -0.0364 |
  | all train windows | 4.53M | -0.0265 |

  | Memory (scored: first half of the held-out text, model 1.2140) | keys | gain |
  |---|---|---|
  | held-out of the **other** books (unseen, different documents) | 220k | **worse, +0.0057 at best** (all 6 settings hurt) |
  | held-out of the **same** books (unseen, adjacent) | 220k | **-0.0355** |
  | train, equal size | 230k | -0.0036 |
  | all train windows | 4.53M | -0.0268 |

  So the memory's gain is **document-specific retrieval**: recent text of the same book (names, scenes, phrasing)
  helps a lot, even in small amounts, and more than 10x as much train text from the earlier 90% of those books; text of
  other books hurts. "Unseen by the model" is not what matters; "from the same document" is. Caveats: the held-out
  memory here includes text *after* the scored window as well as before it, which a streaming system would not have, so
  the -0.036 is an upper bound on a causal (past-only) in-document memory, not a deployable number; and the
  neighbouring windows share rare words and names with the scored ones.
- **k = 256 and a disjoint split (`runs/tier_o8_final.log`; tuned on offset 0, tested on offset 8; model 1.2524 on
  offset 8).** knn k=128 -0.0315, **k=256 -0.0324**; words:1:0.25 -0.0150; words+knn256 -0.0368;
  **lexicon+words+knn256 -0.0381**. Tuning-split numbers were smaller (k=128: -0.0265, k=256: -0.0276), so there is
  no tuning bias, and k's gain is flattening (128 to 256: +0.0009 to +0.0011).
- **Full held-out split, all tiers** (`scripts/tier_full2.sh`, `runs/tier_full2.log`; Gate 0 passed at 1.2408; 4.53M keys):

  | Tier | CE | vs model |
  |---|---|---|
  | words:1:0.25 | 1.2247 | -0.0161 |
  | lexicon:0.3 + words:1:0.25 | 1.2225 | -0.0184 |
  | knn:256:0.4:25 | 1.2118 | -0.0291 |
  | words:1:0.25 + knn:256:0.4:25 | 1.2066 | -0.0342 |
  | lexicon:0.3 + words:1:0.25 + knn:256:0.4:25 | **1.2053** | **-0.0356** |

  The three tiers largely add (-0.0184 and -0.0291 alone-ish, -0.0356 together). The model plus tiers is 0.151 nats
  under the 7-gram (1.3567).
- **Engineering note:** a `KMAX` = 256 top-k launch over a whole 16384-key tile lost the device (OS GPU watchdog; the
  `Failed to map buffer` / `Parent device is lost` panics). The kernel now takes 1024-key slices per launch; the
  unit test is unchanged.

- **Protocol caveat (found after the rows above; read this before comparing numbers).** The standard held-out score
  gives position t of each 64-byte window only t bytes of context, so early positions are scored almost blind.
  `tier_eval warm=32` scores every byte with at least 32 bytes of context (`runs/tier_warm.log`): the **model alone
  is 1.1764, not 1.2408**. Consequences: (1) the `h` flag (CPU tiers reading 256 preceding bytes) looked like a big
  win under the standard protocol (`runs/tier_ctx.log`: lexicon:0.3:h -0.0217, words:1:0.35:h -0.0395, lexicon+words
  -0.0463) but is **nothing under `warm=32`** (lexicon -0.0066 with or without `h`; words -0.0174 vs -0.0159 with
  `h`): it only compensated for the missing context. The `f` flag (first-letter prediction between words) did not help
  either way (-0.0137 vs -0.0161 without it). (2) The kNN memory is **not** an artifact: its gain is slightly larger
  under `warm=32` (-0.0328 vs -0.0276 on the same subsample, `runs/tier_warm_knn.log`). (3) Every "vs model" number
  in the earlier rows is valid only for the standard protocol; the 7-gram's 1.3567 is scored the same windowed way
  (`ngram_baseline.rs`), so that comparison is like for like, but model + tiers on the fair protocol has no 7-gram
  number yet. **Now measured** (`runs/ngram_novels6.log`, `ngram_baseline novels6`): the order-7 Kneser-Ney model scores
  1.3567 windowed and **1.3237 with the full left context** (the stream number, the n-gram counterpart of `warm`; it
  needs only 6 bytes, so `warm=32` is the same). The model alone at `warm=32` (1.1764) is therefore **0.147 under
  the 7-gram on the fair protocol**, against 0.116 under it on the windowed one (1.2408 vs 1.3567): the windowed
  numbers understated the model's lead, and the earlier 7-gram comparisons in the README and `fusion_results.md`
  (all windowed on both sides) remain like for like.
- **Causal in-document memory (`memory=causal`; `runs/tier_causal*.log`, `tier_full3_causal.log`).** The memory is
  the train keys plus the held-out windows before the scored one in the same book (past-only; GPU search masks each
  query's allowed key ranges, `KnnStore::search_masked`, unit-tested against a CPU brute force). On the `warm=32`
  subsample (model 1.1844): train keys only -0.0328; **train + past same-book windows -0.0448**; past same-book
  windows alone (503k keys) -0.0201 (its best temp 40 is interior; the mean neighbour distance is 93 vs 36, so the
  memory is sparse). So a deployable in-document memory adds about 0.012 on top of train keys.
- **d = 384 (round 18 checkpoint, 1.2413; `runs/tier_d384_flat.log`).** Gate 0 passed. Lexicon -0.0063, words:1
  -0.0152 (standard protocol). Memory under `warm=32` (model 1.1817 on the subsample): best -0.0324 at lambda 0.5,
  temp 30 (the lambda grid's top), against -0.0328 for d = 256 (model 1.1844): the memory's gain does not depend
  much on the model width, and the d = 384 absolute score is slightly better (1.1493 vs 1.1516).
- **Fair-protocol full runs (`warm=32`, every other held-out window = 251904 positions, 4.5M train keys; scripts/
  tier_full3.sh; settings tuned on stride-16 offset 0):**

  | Tiers | CE | vs model (1.1778) |
  |---|---|---|
  | lexicon:0.3 | 1.1712 | -0.0066 |
  | words:1:0.25 | 1.1601 | -0.0177 |
  | lexicon + words | 1.1578 | -0.0201 |
  | flat memory knn:256:0.4:25 | 1.1462 | -0.0316 |
  | lexicon + words + flat memory | 1.1393 | -0.0386 |
  | causal memory knn:256:0.5:15 | 1.1353 | -0.0425 |
  | words + causal memory | 1.1318 | -0.0461 |
  | lexicon + words + causal memory | **1.1311** | **-0.0467** |

  The memory is the largest single tier, the causal in-document memory beats the flat one by 0.011, and the CPU
  tiers add about 0.004-0.007 on top of the memory.

- **Recency (`recent=<windows>`, past-only memory; `runs/tier_recent.log`).** Limiting a query to the last n
  windows of its book never helped: held-out keys only (warm 32 subsample, model 1.1844): n=8 and 32 hurt (+0.195,
  +0.138: too few keys for k=256), n=128 -0.0115, n=512 -0.0246, unlimited **-0.0305** (best at lambda 0.2, temp 40; an
  earlier -0.0201 was an under-wide grid). With train keys: n=512 -0.0436, n=2048 -0.0448, unlimited -0.0448. So the
  whole document so far is the best memory; there is no recency effect to exploit at this scale.

- **Online memory (`memory=online`; `runs/tier_online_*.log`).** The memory starts as the train keys and each scored
  chunk's scored positions are appended after scoring (write latency one 32-window chunk; stride 1, `warm=32`,
  `part=3/6`, `store=100000`; model alone 1.1453). Same slice, same specs:

  | memory | knn:256:0.5:15 | 0.4:15 | 0.5:25 |
  |---|---|---|---|
  | flat (train keys) | -0.0316 | -0.0323 | -0.0314 |
  | online (written as read) | -0.0374 | -0.0374 | -0.0361 |
  | causal (past-only held windows, precomputed) | **-0.0407** | -0.0403 | -0.0386 |

  Online recovers about 0.006 of the 0.009 causal-over-flat gain; the rest is the chunk write latency (the 32 most
  recent windows are invisible) plus keys being written from the scored positions only (t >= 32). Online is the
  deployable form (no held-out text needed in advance), so the realistic in-document memory gain is ~-0.037.

- **Eval speed (phase timers in `tier_eval`; `runs/tier_time_{online,flat}.log`, `part=5/24`, 21 chunks of 32 windows, 4.5M
  keys, one knn spec).** 160 s per run, **99.9% in tier prepare** (the GPU kNN search plus its readback; ~7.6 s per
  2048-query chunk); row scoring 0.1 s, online writes ~0 s. Online and flat cost the same. The forward pass is queued
  asynchronously, so its time lands in prepare too; the split inside the search (matmul, top-k, readback) is not yet
  measured. Contended: CPU 35% at launch, another process (`WizardGraphicalClient`) on the eGPU, Defender real-time off.
  So the suspected per-window flush and CPU softmax are not the bottleneck; the search is.

- **Where the search time goes (`KNN_TIMING=1`, staged syncs in `KnnStore::search_masked`; 4.53M keys, k=256).** Per
  2048-query block: matmul 1.3 s (16%), **top-k 6.4-7.9 s (84%)**, readback 3 ms. A 256-query block still takes 5.4 s
  of top-k (matmul 0.4 s): top-k time barely depends on the number of queries, so the kernel (one thread per query, a
  serial scan over all 4.5M keys) is latency-bound and leaves the GPU mostly idle; the matmul is already efficient.
  Contended run (CPU 85%, `WizardGraphicalClient` on the eGPU). Fix to try: split each query's key range across
  several threads (partial top-k per slice, then merge), which should scale until the GPU is saturated.

- **Top-k is insertion-bound, not scan-bound (`KNN_PARTS`, same 2048 x 4.53M block; contended, CPU 40-85%).** Splitting
  each query's slice over 8 threads (`PARTS`, lists merged on the host; SLICE 4096) is exact (same CE 1.0949) but only
  cut top-k 6.4 -> 5.5 s. Sweep: 1 part 6.1 s, 8 parts 5.5 s, 32 parts **18.5 s** (128 parts did not finish). Scan-only
  (insertions disabled): 1 part ~2 s, 8 parts ~0.6-0.9 s. So ~4-5 s is the sorted-list insertions (rank count over 256
  entries + shift, divergent within a wave, global memory), and each extra part adds its own ~k(1+ln(n/k)) insertions,
  which is why more parts get slower. Next lever: cut insertions, e.g. a per-query threshold from a sampled pass 1
  (only keys under it are candidates; fall back when fewer than k qualify), or binary-search insertion.

- **Faster search: thresholds + 16 parts (`src/gpu_step/knn.rs`).** Binary-search insertion alone changed nothing (6.1 vs
  6.1 s), so the cost is the shifting/divergence of insertions, not the rank count. What worked: a pass 1 over every
  61st key (sample on the host, uploaded when it grows) gives each query a threshold (its 32nd nearest sample key ~ the
  1950th nearest key); the main pass inserts only keys under it, and a block with any query that has fewer than 256 keys
  under its threshold is searched again without thresholds (exact either way; test
  `thresholded_search_matches_cpu_brute_force`, masks included). Two bugs found on the way, both from structure in the
  keys: a stride of 64 equals the window length, so the sample was only window position 0 (stride 61 fixed it), and
  position-0 keys are exact duplicates (they see only their own byte), so a threshold equal to a tie distance excluded
  the ties (`widen` adds 1e-4 relative). With neither, every block fell back. Result, 2048 queries x 4.53M keys, same
  contended machine (CPU ~40%, another process on the eGPU): block 6.7-7.1 s -> **3.5-3.7 s** (top-k 5.3 -> 1.6-2.2 s;
  matmul 1.4 s and the sample pass ~0.15 s are now the floor); the 6-chunk slice 49 s -> 22.6 s. CE unchanged
  (1.0949). Parts sweep with thresholds: 2: 3.6 s, 4: 2.7, 8: 2.2, 16: 1.9, 32: 1.6, 64: 1.6 (host merge/readback grows
  with parts: 6 -> 110 ms), so the default is 16.

- **End-to-end check of the faster search (`scripts/tier_online.sh` rerun, same slice).** Every CE matches the earlier
  run (the one 1.1046 -> 1.1045 is the 4th decimal; all deltas identical to 4 places), and the wall time per run fell
  from ~11-12 min to **281 s (online), 248 s (causal), 232 s (flat)**, still 99% in the kNN search. Machine busy (CPU
  ~100% from other sessions at launch), so these are upper bounds.

- **Skip unscored rows (`warm>0`).** `Store::prepare` now searches only rows at window position >= warm (the others were
  searched and never scored). Same slice (`part=5/96`, warm 32): CE identical (flat 1.0949, online 1.0943), 22.6 s ->
  **14.1 s** flat (16.3 s online). Top-k does not halve with half the queries (1.5 s for 1024 vs 2.2 s for 2048: fewer
  threads, latency-bound), so a wider `PARTS` for small blocks may help. Contended machine (CPU ~32%).

- **Store build time (`memory build:` line in `tier_eval`).** Building the 4.53M-key memory costs ~42 s per launch
  (CPU ~55% from other sessions): forward + hidden readback 34.8 s (82%), host copy + norms 2.2 s, upload 4.0 s, other
  1.4 s. The build forwarded 32 windows per launch and also read back logits and row losses it never uses; skipping
  those reads and forwarding 64 windows per launch gives **30.0 s** (forward + readback 24.0 s), identical CE (1.0797).
  Remaining: ~22 ms per 64-window launch of GPU compute plus a 4 MB readback over USB4, each launch serialised with the
  host copy; overlapping them (queue the next forward before reading the last) could recover a few seconds. Sharing the
  store across memory modes in one process would save the whole build per extra mode (needs `KnnStore::truncate`).

- **Full held-out, deployable stack (`scripts/tier_full4.sh`, `runs/tier_full4_online.log`).** `warm=32`, stride 1, whole
  held-out text (15744 windows, 503808 scored positions), online memory (train keys, then the scored text written as
  it is read; write latency 32 windows), the CPU tiers stacked. Model alone 1.1764.

  | tier | CE | vs model |
  |---|---|---|
  | lexicon | 1.1697 | -0.0066 |
  | words (bigram) | 1.1590 | -0.0174 |
  | lexicon + words | 1.1566 | -0.0197 |
  | online memory knn:256:0.5:15 | 1.1352 | -0.0412 |
  | words + memory | 1.1318 | -0.0445 |
  | lexicon + words + memory | **1.1313** | **-0.0451** |

  Against the offline causal stack at stride 2 (1.1311, -0.0467 vs 1.1778) the deployable online stack gives up 0.0016
  nats/byte (-0.0451 vs -0.0467) and needs no held-out text in advance. The online memory alone is -0.0412 on the full
  text (-0.0374 on the earlier one-sixth slice, mostly one book; the difference is not explained), against -0.0425 for the causal
  memory. Run: 2502 s (42 min), 98% in the kNN search (~5 s per 32-window chunk), build 29 s, CPU tiers 27 s; the machine
  was shared (CPU ~60% from other sessions), Defender real-time off.

- **f16 matrix cores for the search (default since 2026-10-02; `KNN_F32=1` selects the exact f32 path).** Keys and queries are rounded to f16 and multiplied
  on the matrix cores (cubecl cmma, f16 x f16 -> f32, 16x16 tiles, one plane per tile; the spike's kernel); norms come
  from the rounded vectors, so a distance is exactly the squared distance between the rounded vectors. Unit test
  `f16_search_matches_cpu_brute_force_on_rounded_vectors` (padding included). Timing, same slice (`part=5/96`, warm 32;
  CPU ~30%): matmul per 1024-query block 0.9-1.1 s -> **0.53 s** (~1.8x; the README's 2-3x was an upper bound), slice
  17.2 s -> 13.4 s, CE 1.0949 -> 1.0950 (a 1e-4 change). Key memory halves (4.6 GB -> 2.3 GB), which matters for larger
  memories on the 16 GB card. Top-k (1.5 s per block) is now ~75% of the search. Not default because results differ in
  the 4th decimal; the f32 path stays the exact reference.

- **Full held-out stack with the f16 search (`scripts/tier_full5.sh`, `runs/tier_full5_f16.log`; 2026-10-02 20:04,
  CPU 17% at launch).** All six specs and the model alone match the f32 run (`tier_full4`) to the 4th decimal
  (lexicon+words+memory 1.1313, -0.0451; memory alone 1.1352, -0.0412). Wall time of the scoring loop **2502 s -> 1355 s**
  (23 min; search 1306 s), build 30 s. So `KNN_F16=1` costs nothing measurable at the reported precision and nearly halves
  the full run; it is now the default (falls back to f32 if the device lacks the f16 matrix-core configuration or d, tile are not multiples of 16).

- **Top-k floor and parts (f16, 1024-query blocks of the `part=5/96` slice; contended: another process on the eGPU, CPU
  27-100% from other sessions, so read these as rough).** With insertions disabled the scan alone costs **0.5-0.6 s**
  per block (32 parts) against 1.4-2.4 s with insertions: even with thresholds, insertions are ~70% of top-k. Parts for a
  1024-query block, best of 2 rounds: 16: 1.63 s, 24: 1.44, 32: 1.48 (second round 2.08, 1.79, 1.76), so 24-32 is ~10-15%
  better than 16; `parts` is now `32768 / queries` clamped to 8..=32 (`KNN_PARTS` forces a value).
- **Top-k insertion shifts only live entries (2026-10-02).** `k_topk` used to shift the whole INF-padded tail on every
  insertion (~200 entries); it now finds the live count once (binary search for the first INF, so it resumes across
  slices and tiles) and shifts only to it. With thresholds a part sees ~120 survivors, so lists stay short. Same slice
  (`part=5/96`, warm 32, online; `runs/tier_cnt.sh`; contended machine, CPU ~30%, Defender off), CE identical (1.0943):
  top-k per 1024-query block **1.38-1.45 s -> 0.82-0.85 s** (~1.7x; first block of each run is warm-up). Whole slice
  15.7 / 22.3 / 17.7 s (3 baseline runs) -> 13.7 / 14.9 s; noisy. Split of a block now (`KNN_TIMING=1`, which also prints
  the pass-1 time): total ~2.2 s = matmul ~0.6 + **pass-1 thresholds ~0.6** + top-k ~0.9. The threshold kernel (`k_thresh`)
  is one thread per query (1024 threads) scanning ~74k sample keys serially: it now costs as much as the matmul. Fused
  matmul+filter kernel stays parked: dist traffic is ~0.06 s per block, and the gain would be the scan floor (~0.5 s) only.
- **Pass 1 was upload-bound, not scan-bound (2026-10-02).** Splitting `k_thresh` across parts changed nothing (kept: it is
  correct and cheap). Staged syncs (`KNN_TIMING=1` now prints `knn pass 1: ...`) showed ~340 ms per block in "sample
  upload": an online memory grows every chunk, and `sample_chunks` re-converted and re-uploaded the whole ~74k-key sample
  (host f16 conversion + 38 MB). Now only the last partial chunk is rebuilt. Same slice: pass 1 ~370-420 ms -> ~70-90 ms,
  block total ~2.2 s -> ~1.3 s (matmul ~0.5, pass 1 ~0.08, top-k ~0.7), CE identical (1.0943). Test
  `search_stays_exact_while_the_store_grows` covers search / add / search across sample-chunk boundaries.
- **Full held-out rerun with the top-k and pass-1 fixes (2026-10-02, `scripts/tier_full6.sh`, `runs/tier_full6.log`).** CE
  identical to the f16 run (full stack 1.1313; memory alone 1.1352), but **1464 s vs 1355 s: not faster**, although the
  `part=5/96` slice ran 15.7 -> 10.4 s (same exe, minutes later; CPU ~54% from other sessions at that check). The full run
  costs 2.86 s per chunk (492 chunks) against ~1.3-1.7 s per chunk on the slice, so the slice speedup did not carry over.
  Unexplained: contention during the run was not measured (the eGPU users and CPU load were checked only before launch),
  and the full run's memory is larger (5.04M keys at the end vs 4.5M). Do not quote the full-run speedup until it is
  reproduced under a logged quiet machine.
- **tier_full7 (2026-10-02 22:06 -> 2026-10-03 00:44; `scripts/tier_full7.sh`, `runs/tier_full7.log`, load samples in
  `runs/tier_full7_load.log`): finished, but 2 h 38 min, not ~25 min.** CE identical again (full stack 1.1313). The
  `KNN_TIMING` blocks sum to 9255 s and **grow through the run**: 1.97 s at block 100 (4.53M keys), 5.8 s at 250, 17.4 s at
  400, 31-73 s over the last ~100 (5.03M keys); 329 of 492 blocks took over 10 s, 7 under 2 s. Matmul and top-k slow together
  (equal shares each block). `WizardGraphicalClient` was on the eGPU in nearly every sample, CPU averaged 71% (Defender off).
  The growth is explained in the next entry (one tiny device tile per window), not by VRAM spill or store size.
  tier_full6's 1464 s (2.86 s/chunk) was probably a milder case of the same thing. `tier_eval` now prints
  `speed: chunks a..b X s/chunk` every 20 chunks and pauses while another job holds an exclusive GPU lease.
  Note: `tier_full5.sh` / `tier_full6.sh` named above no longer exist; `tier_full7.sh` is the same stack.

- **Root cause of the growing block time: one tiny tile per window (2026-10-07; `examples/gpu/knn_scale_check.rs`,
  `runs/knn_scale_check*.log`).** tier_full8 (rerun on a then-quiet machine, killed at chunk ~200 of 492) was slow too:
  2.3 s/chunk for chunks 0-20, 8-9 at 40-100, 16 at 180-200, with only 3.2 GB VRAM (spill ruled out) but other sessions'
  jobs arriving mid-run (CPU 65% average; `tier_eval_run` used ~1 core). Store-size ladder at the tier_eval shapes (d 256,
  tile 16384, 1024 queries, f16, Gaussian keys, best of 5 on an exclusive lease, CPU 27-60% from another session's job):
  0.5M keys 0.38-0.47 s, 2M 0.65, 5M 0.97-1.01 s, so **size alone is linear and mild**, and no drift (same 5M store after
  60 s idle: 1.0 s; fresh 0.5M store: 0.44-0.56 s). `write_chunk` calls `finish()` after every window, so each 32 keys
  (SEQ_LEN 64 - warm 32) become their own device tile and a chunk adds 32 tiles (~15.7k by the end of a run, against ~300
  full tiles). Adding 125k / 250k / 500k keys that way to the 5M store: **1.79 / 2.76 / 4.43 s per block** (~0.22 ms per
  extra tile per search, matmul + top-k launches). That is the time-driven growth seen in tier_full5-8; with the host
  starved by other jobs each launch costs far more (tier_full7: 31-73 s blocks), which fits LONG_RUNS' "round trips ~25x
  slower with all cores busy". The earlier slice speedups (`part=5/96`, few chunks) never hit it. Not yet fixed. Options:
  (a) keep the partial tail in `pending` and have `search` upload it as a transient tile (no `finish()` needed; `len()`
  must count pending keys, and `flush`'s `first` and the doc-start indices must stay consistent); (b) cheaper, `finish()`
  once per chunk instead of per window (492 tiles of 1024 instead of 15.7k of 32; needs `len()` to count pending for
  `doc_first_key`). Then rerun the full stack; expect ~1-1.5 s/chunk and a run of ~10-15 min, to be verified with the
  `speed:` lines and a quiet-machine log.

- **Fix: pending keys are searched as a transient tile (2026-10-07).** `KnnStore::len()` counts pending keys, `search`
  uploads the pending tail as one extra tile, and `write_chunk` no longer calls `finish()` per window (test
  `pending_keys_are_searched_without_one_tile_per_add`, exact against brute force, plain and masked). `knn_scale_check`
  with 500k keys added 32 at a time: **1.04 s per block (was 4.43 s)**, flat at 5.0M -> 5.5M keys (CPU 76% average from
  another session's job). `flush_time[1]` is now always zero (conversion and upload are timed together in `[0]`).
- **tier_full9: the fix, full held-out stack (2026-10-07 18:50 -> 19:04; `scripts/tier_full9.sh`, `runs/tier_full9.log`,
  `runs/tier_full9_load.log`).** **825.6 s for the scoring loop (13.8 min), down from 1355 s (tier_full5, f16) and 2502 s
  (f32); 1.44-2.21 s per chunk, flat from chunk 0 to 492** (tier_full7: 2 s growing to 73 s). CE identical to the
  earlier runs (full stack 1.1313, -0.0451; memory alone 1.1352). Phases: forward 6 s, tier prepare (search) 799 s (97%),
  row scoring 19 s, online writes 1 s. Machine: CPU 36% average (another session's `bv_sbo` and `WizardGraphicalClient`
  present, so not fully quiet), Defender off, dedicated VRAM 3.3 GB, no shared-VRAM spill. Per chunk ~1.7 s against the
  ~1.3 s of the `part=5/96` slice, which had ~4.5M keys and less load. The search is still ~97% of the run: matmul ~0.5 s
  and top-k ~0.7 s per 1024-query block (see the top-k and pass-1 entries above) are the next places to look.

- **The matmul is ~a quarter of the search; top-k is the target (2026-10-07; `examples/gpu/cmma_blocking_check.rs`).**
  One 16384-key x 1024-query x 256 product (the real tile shape, f16 -> f32): `k_dots_cmma` (one plane per 16x16 tile,
  fragments loaded from global memory, no reuse) **0.77 ms = 11 TFLOP/s**; a plane holding 2x2 accumulator tiles
  **0.46 ms = 18.6 TFLOP/s**, output identical (quiet machine: CPU 15-20%, no other eGPU users, exclusive lease). A 5M-key
  block has ~307 tiles, so the whole matmul is ~0.24 s of the ~0.95 s an undisturbed block takes (`knn_scale_check`, no
  timing syncs). 2x2 would save ~0.1 s per block (~10%); even a 4x faster kernel saves under 0.2 s. The `KNN_TIMING`
  "matmul 0.55 s" overstates it: timing mode syncs after every tile. The remaining ~0.7 s per block is top-k insertion
  plus launch overhead; the fused matmul+filter kernel (parked above) is not worth building for the matmul alone. Not
  done: a 4x4 or shared-memory variant, and the 2x2 kernel is not wired into `KnnStore`. If search speed matters again,
  look at top-k insertions (a coarser threshold, fewer parts' lists, or a survivor-compaction pass) before the matmul.

- **Keys per top-k launch: 4096 -> 16384 (2026-10-07; `KNN_SLICE`, `scripts/knn_topk_grid.ps1`, log in `runs/`).**
  Block time at 5M keys (`knn_scale_check`, `QUICK=1`, best of 5, CPU 15-41%): tile 16384 / slice 4096 **884-905 ms**
  (two runs), slice 8192 762, **slice 16384 706**; tile 65536 / slice 4096 883, 16384 697, 65536 **625** (its worst round
  was 3.8-4.5 s, a warm-up or watchdog-adjacent stall, so not adopted). So launch count explains ~20-30% of a block; the
  other ~0.6 s is the scan itself. 16384 is the default now (one launch per default tile; 512 keys per thread per launch
  at 32 parts). **Full run tier_full10 (`scripts/tier_full10.sh`, 19:28-19:46): no device loss, CE identical (1.1313),
  but 1119 s with the machine at 86% CPU (another session's `bv_sbo`), against 826 s at 36% CPU for tier_full9; mean
  top-k per block 746 ms vs 726 ms, so the full run neither confirms nor refutes the gain.** The 826 s figure stands as
  the best measured; a full run under similar load is needed to compare the two slices fairly.

- **Uncertainty gating: tooling built, run held (2026-10-07; spec `docs/superpowers/specs/2026-10-07-uncertainty-gating-design.md`,
  plan `docs/superpowers/plans/2026-10-07-uncertainty-gating.md`).** `tier_eval ... dump=<path>` writes six f32 per scored
  position of the last spec's knn tier (chunk, p_t, k_t, entropy, d0, dk); `gate_fit <dump> [logged=<CE>]` checks Gate 0
  (dump CE at lambda 0.5 == logged CE to 1e-4), then fits a 4x4 binned gate and a 4-parameter sigmoid gate on even chunks
  and scores odd chunks. Features are `ln(1+d0)` and `(dk-d0)/(1+d0)` plus entropy (the spec said `ln d0`; harmless, distances
  are >= 0), standardised on the fit half. Success and kill are judged against the gain over the **best fixed lambda**, not
  0.5 (a gate can reproduce any constant); both are printed. Caveats: even/odd are interleaved chunks of the same books, so
  the held-out test is weak. **Slice check done (`part=5/96`, 5248 positions, 6 chunks, 48% CPU; correctness only):
  Gate 0 passed (dump CE 1.08876 vs logged 1.0888; 5248 rows = 125952 bytes).** Gate numbers on it are noise: binned
  -0.0018, sigmoid -0.0001 vs best fixed 0.42 (odd half, ~2k rows); the binned table has 0.00/1.00 bins from a few rows,
  and the sigmoid weights were negative on entropy, distance and spread (lambda falls as uncertainty rises), opposite to the
  expectation, so check the sign on the full run.
  **Larger slice (`part=5/24`, 20992 positions, 656 windows, CPU 56%, `runs/gate_slice24.bin`): Gate 0 passed again (1.01596 vs
  logged 1.0160).** Gain over the best fixed lambda (0.45 even/odd, 0.42 early/late), held-out half: binned +0.0002 / +0.0001,
  sigmoid **+0.0012 / +0.0010** nats; over fixed 0.5 the sigmoid gets +0.0024 / +0.0020. Both are below the 0.002 success bar
  against best fixed, and the binned gate is under the kill line, so on this slice the gate is at best a marginal gain; the
  full run (about 24x the positions) decides. Shape of the binned table (stable across both splits): lambda rises from ~0.2
  at low entropy to ~0.7 at mid-high entropy, then falls to ~0.3 in the top entropy bin (the model is lost and so are its
  neighbours); the sigmoid cannot express that hump, which is why it gets a negative entropy weight. If the full run shows
  the same hump, a quadratic entropy term (or more entropy bins) is the next gate to try. Also added: `dump=` is written
  atomically every 20 chunks, `gate_fit` takes several dump files (chunk ids offset by 100000 per file) and reports a second,
  harder early/late split next to even/odd.
  **Full run done (`scripts/tier_gate_dump.sh`, 2026-10-07 23:35-23:43, contended: `manifold`/`WizardGraphicalClient` on the eGPU,
  CPU 36-48%): 503808 positions, CE 1.1313, 448.6 s, flat 0.8-1.2 s/chunk, no device loss. Gate 0 passed (dump 1.13127 vs
  logged 1.1313).** Held-out gain in nats (`gate_fit runs/tier_gate_dump.bin logged=1.1313`; best fixed lambda 0.40 / 0.41):

  | split | binned vs 0.5 | binned vs best fixed | sigmoid vs 0.5 | sigmoid vs best fixed |
  |---|---|---|---|---|
  | even/odd | +0.0022 | +0.0011 | +0.0022 | **+0.0011** |
  | early/late | +0.0021 | +0.0008 | +0.0023 | **+0.0010** |

  **Verdict: below the success bar (0.002 over best fixed) and the binned gate is under the kill line (0.002), so gating the
  kNN lambda by entropy and neighbour distance is not worth wiring in: ~0.001 nats (0.1%) of the 0.045 the stack already
  gains over the model.** Most of the "gain vs 0.5" is just moving the constant to 0.40. Shape: lambda is highest (0.6) at
  mid entropy and short distance, lowest (~0.3) at the top-entropy bin and at large distances; the sigmoid fits negative
  weights on entropy and ln(1+d0), so the hump seen on the 21k slice is weaker here. The 4x4 table is smooth and stable
  across both splits, so the signal is real but small; a quadratic entropy term might add a few 1e-4, not 2e-3. Practical
  takeaway: use a kNN weight of 0.4 in future `tier=...knn:256:0.4:15` specs (no code default to change; lambda is only a spec
  argument in `scripts/`; the online memory does not depend on lambda, so the dump already gives the exact CE at 0.4, about
  1.1301 against 1.1313 from the odd/even halves, and a confirming run adds nothing) and look for the next
  tier gain elsewhere (gate the CPU tiers, or a richer memory), not in lambda. Timing (448.6 s vs 825.6 s for tier_full9)
  is not like-for-like: this run scored one spec instead of six, dropped `KNN_TIMING`, and used slice 16384, so it neither
  confirms the slice gain nor the 826 s figure. The even/odd halves share books, hence the early/late split next to it.
  Tests: `cargo test --release --example gate_fit -j 8` (6 pass). Commits 3808d38..807957d.

- **Words-weight gating: tooling built, full run in flight (2026-10-08; spec `docs/superpowers/specs/2026-10-08-words-gating-design.md`,
  plan `docs/superpowers/plans/2026-10-08-words-gating.md`).** `tier_eval ... dump=<k> wdump=<w>` writes the kNN rows and, per
  scored position, the words tier's inputs (applied, p_in, q, entropy_in, ln(1+count mass), bigram hit, prefix length);
  `words_gate_fit <w> <k> logged=<CE>` joins them (Gate 0b: 0.75 p_in + 0.25 q == the kNN row's p_t), checks the logged CE (Gate
  0), then fits a bigram x entropy-quartile gate and a 6-weight sigmoid gate (also seeing count mass, prefix length, ln(1+d0))
  for the words weight with the kNN weight fixed at 0.4, on even/odd and early/late splits, and prints one verdict
  (success: sigmoid > 0.002 on both splits; kill: binned < 0.002 on either; else inconclusive) with per-chunk standard errors.
  Slice check (`part=5/96`, 5248 positions, quiet machine): both dumps 5248 rows, Gate 0b and Gate 0 passed (1.08783 vs
  logged 1.0878); the slice gains are noise (sigmoid -0.0033/+0.0005 +-0.002).
  **Full run done (`scripts/tier_words_gate_dump.sh`, 2026-10-08 04:54-05:02, quiet at launch: CPU 13%, no leases; during the run
  mean CPU 49% (the run itself) and `WizardGraphicalClient` on the eGPU for ~5 of 14 samples): 503808 positions (379168 with words
  applied), 417.2 s (flat 0.72-1.02 s/chunk), CE 1.1301 (-0.0462 vs the model), which confirms the predicted CE at kNN weight 0.4
  (1.1313 at 0.5). Gate 0b and Gate 0 passed.** Words weight gains over the best constant (0.19) on the held-out half, nats per
  position, mean +- per-chunk standard error:

  | split | binned | sigmoid |
  |---|---|---|
  | even/odd | +0.00021 +- 0.00004 | +0.00039 +- 0.00008 |
  | early/late | +0.00012 +- 0.00005 | +0.00042 +- 0.00008 |

  **Verdict: KILL (binned < 0.002); the sigmoid gate is also ~5x short of the bar.** The effects are real in the statistical
  sense (several standard errors) but four hundredths of a percent of the CE: a per-position words weight is not where the
  remaining gain is. Shape: the best weight rises with the model's entropy (0.06 at the lowest quartile to 0.2-0.26 at the
  highest) and slightly with a bigram hit; moving the constant from 0.25 to 0.19 alone gains ~0.0002-0.0003, so retuning
  `words:1:0.19` is the cheap leftover (about +0.0003 nats; confirm with one run). With the kNN gate (also killed, +0.001)
  this closes mixture-weight gating for the current tiers: the weights are within ~0.001 nats of optimal either way, so
  gains must come from what the tiers contain (richer memory, better word model), not how they are weighted. Logs
  `runs/wg_dump.log`, `runs/wg_dump_load.log`; dumps `runs/wg_dump_k.bin`, `runs/wg_dump_w.bin`.

- **Richer memory, step 1: remove the online write latency (`write=first`; 2026-10-08).** Survey: the memory's gain is
  document-specific, and online lost 0.0033 to the offline causal memory on `part=3/6`. With `warm=32` the windows overlap
  (a new window every 32 bytes), so the old online path already wrote every text byte once; the gap was only the
  32-window write latency (up to ~1 KB of the most recent text invisible, because a chunk was searched before its keys
  were written). `write=first` writes the chunk's keys before its search and gives each window's queries only the keys of
  earlier windows (same causal rule, nothing from the future), so it stays deployable. `part=3/6`, `knn:256:0.5:15`: write
  after -0.0374, **write first -0.0406**, causal -0.0407 (`scripts/tier_write_first.sh`; `part=5/96` is too small a slice
  to test this: online starts a slice with an empty in-document memory, causal does not). Full held-out
  (`scripts/tier_full_write_first.sh`, `runs/tier_full_wf.log`; 503808 positions, 384 s at 0.79 s/chunk, steady; CPU ~27%,
  GPU otherwise idle, Defender off):

  | stack | write after | write first | gain |
  |---|---|---|---|
  | model alone | 1.1764 | 1.1764 | |
  | knn:256:0.5:15 | 1.1352 | 1.1322 | 0.0030 |
  | lexicon + words + knn 0.5 | 1.1313 | 1.1282 | 0.0031 |
  | lexicon + words + knn 0.4 | 1.1301 | **1.1274** | 0.0027 |

  Best deployable stack is now 1.1274 (-0.0489).

- **Retune under `write=first` (`scripts/tier_retune_wf.sh`, `runs/tier_retune_wf.log`; 2026-10-08).** One full run, 24 specs
  sharing the search: words weight {0.25, 0.19} x knn weight {0.4, 0.5, 0.6} x temp {10, 15, 20, 25}. Best **0.19 / 0.4 / 15:
  1.1272** (-0.0492), against 1.1274 at the current 0.25 / 0.4 / 15: the stack is already at its optimum (gain 0.0002, less
  than the ~1e-4 selection bias of picking the best of 24 on the scoring text). The words weight 0.19 is worth 0.0001-0.0002
  everywhere. Ridge: (0.4, 15) 1.1274, (0.5, 20) 1.1279, (0.5, 15) 1.1282, (0.4, 20) 1.1282; temp 10 and 25 and weight 0.6
  are clearly worse (1.1296-1.1448). Weight 0.3 was not tried (0.4 beat 0.5 at temp 15 by 0.0008, so a slightly lower
  weight could add ~1e-4). Defaults left at `words:1:0.25` / `knn:256:0.4:15`. Run: 502 s (row scoring 27% with 24 specs);
  another session's `bv_cd` was running, CPU 27-57%. Mixture weights and temperature are now closed; the remaining gain has
  to come from the memory's contents (option 2: separate train and in-document keys; option 3: richer values).

- **Richer memory, step 2: weight in-document neighbours separately (`knn:k:λ:T:T_doc:shift`; 2026-10-08).** Keys carry a
  source tag (`KnnStore::add_doc`, `search_tagged`: the value travels as `DOC + byte`, so no extra search cost); in-document
  neighbours (keys written from the scored text's own book, causal as in `write=first`) use temperature `T_doc` and have
  `shift` taken off their squared distance before the exp(-(d - d_ref)/T) weight, `d_ref` = the smallest adjusted distance.
  `T_doc = T`, shift 0 is the plain tier (checked: identical CE). Why: only **1.0% (slice) / 2.0% (full) of the 256 nearest
  neighbours are in-document keys**; their distances are larger (sparser memory), so train keys outvote the informative ones.
  Slice `part=3/6`, memory alone, lambda 0.4 / T 15: plain -0.0402; (T_doc 25, shift 40) -0.0491; a plateau over T_doc 25-40
  and shift 30-50, worse at 70-80 (the few doc keys then dominate) (`scripts/tier_doc_split.sh`, `runs/tier_doc_split*.log`).
  Full held-out (`scripts/tier_full_doc_split.sh`, `runs/tier_full_doc_split.log`; 564 s, 1.04-1.52 s/chunk, CPU
  contended by another session; 7 finalists picked on the slice):

  | stack (lexicon 0.3 + words 1:0.25 + knn:256:lambda:T:T_doc:shift) | CE | vs model |
  |---|---|---|
  | plain 0.4:15 (control, reproduces tier_full_wf) | 1.1274 | -0.0489 |
  | 0.4:15:25:30 | 1.1191 | -0.0572 |
  | **0.4:15:25:40** | **1.1182** | **-0.0582** |
  | 0.4:15:25:50 | 1.1189 | -0.0575 |
  | 0.4:15:40:40 | 1.1187 | -0.0577 |
  | 0.5:15:25:50 | 1.1210 | -0.0554 |
  | 0.5:20:40:50 | 1.1188 | -0.0576 |
  | 0.5:20:25:40 | 1.1192 | -0.0572 |

  Memory alone at (0.4:15:25:40): 1.1238 (-0.0526; was 1.1322 plain). Gain **0.0092** over the plain stack, by far the largest
  since the memory itself and 8x any gating result; the finalists agree within 0.0009 so the optimum is flat and the
  selection bias is negligible. Best deployable stack: **1.1182 (-0.0582)**. Not done: a per-source top-256 (an in-document
  key outside the merged top-256 is still invisible; only 2% of neighbours are in-document), re-tuning lambda and the words
  weight with this, and a learned (instead of constant) shift.

- **Retune under per-source weighting (2026-10-08, slice part=3/6): flat.** Best words 0.19/0.4/15/25/40 = 1.1179 on the
  full run's grid vs the 0.25 default 1.1182; lambda, words weight and temperatures are at their optimum.

- **Step B: per-source top-256 (`search=split`, 2026-10-08).** Train keys and in-document keys are searched separately
  (`KnnStore::search_tail` scans only the tiles from `n_train` on), KMAX=256 each, merged by distance. On the slice
  +0.003, 50% of the neighbours then in-document. Full held-out (`scripts/tier_full_search_split.sh`,
  `runs/tier_full_search_split.log`; 696 s, 1.1-1.7 s/chunk, ~1.5x the merged search; CPU 31% from other jobs):

  | stack (knn:k:lambda:T:T_doc:shift) | CE | vs model |
  |---|---|---|
  | memory alone 256:0.4:15:25:50 | 1.1196 | -0.0568 |
  | **+lexicon+words, 256:0.4:15:25:50** | **1.1138** | **-0.0626** |
  | 256:0.4:15:25:40 | 1.1143 | -0.0621 |
  | 256:0.4:15:25:60 | 1.1145 | -0.0619 |
  | 256:0.4:15:40:50 | 1.1148 | -0.0615 |
  | 256:0.5:20:40:50 | 1.1149 | -0.0615 |
  | 64:0.4:15:25:50 | 1.1153 | -0.0611 |
  | 256:0.4:15:25:70 | 1.1163 | -0.0600 |

  Gain **0.0044** over the merged control (1.1182); memory alone 1.1238 -> 1.1196. The optimum is again flat (top five
  within 0.0011). Best deployable stack: **1.1138 (-0.0626)**.
  **Engineering note (crash):** the first version of the tail search skipped the sample-threshold pass ("the sample is
  mostly older keys") and lost the GPU device (`BufferAsyncError`, "Parent device is lost") in every full run within
  ~140 chunks, also with 1024-key slices. The cause was the missing thresholds (the sample includes in-document keys, so
  they stay finite and cheap); with the pass restored the slice that crashed three times finishes (CE 1.1262) and the full
  run completes, and tile skipping was not at fault.

- **Cleanup (2026-10-08): `write=after` removed.** The online memory always writes a chunk's keys before its search now
  (`write=first` was strictly better and nothing is pending against it); the `write=` option, `Online.first` and
  `scripts/tier_write_first.sh` (the A/B) are gone. Check: `part=3/6`, `knn:256:0.5:15` gives 1.1047 and 0.4 gives 1.1051,
  identical to the earlier `write=first` run. Scripts from before this change that use `memory=online` without `write=`
  (`tier_online.sh`, `tier_full4/7/8/9/10.sh`, `tier_gate_dump.sh`, `tier_time.sh`, `tier_words_gate_dump.sh`) reproduced the
  write-after numbers in this log only at commit 854c38e or earlier; today they run write-first. **`search=` removed too
  (decision: keep the split search; +0.0044 nats for ~1.5x the search time, full run 696 s).** The online memory always
  searches train and in-document keys separately; flat and causal memories keep the single (merged) search, which is
  the only path they have. Online scripts before commit f50fdb5 reproduce the merged-search numbers (control 1.1182) only
  at 854c38e or earlier. Check: `part=0/12`, `knn:256:0.4:15:25:50` = 1.1262, identical to the earlier split run.
  `scripts/tier_search_split.sh` (the A/B) deleted.

- **Richer values: the model's surprise at the stored position (2026-10-08): KILL.** Survey option 3. Idea: store, with
  each key's byte, -ln p(true byte) the model had there, and let it scale the neighbour's weight. Diagnostic (flat
  memory, `part=3/6`, 82 chunks, 6 surprise bins): neighbour hit rate falls steeply with key surprise (rank 0-7:
  95.1% / 73.5% / 41.0% / 15.2% / 5.1% / 1.6% for <0.1 / <0.5 / <1.5 / <3 / <5 / >=5 nats; surprise matters more than
  distance rank), but this is confounded with the byte being hard anywhere. Direct test, weight x exp(-beta * surprise):

  | beta | -0.5 | -0.3 | -0.2 | -0.1 | -0.05 | **0** | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | CE | 1.1440 | 1.1219 | 1.1162 | 1.1135 | 1.1130 | **1.1130** | 1.1141 | 1.1163 | 1.1192 | 1.1259 | 1.1360 |

  The optimum is beta = 0 (-0.05 ties it): the mixture and distance weighting already capture what the surprise says, so a
  stored surprise adds nothing in either direction. Logs `runs/tier_surprise.log`, `tier_surprise_beta.log`,
  `tier_surprise_beta2.log`; the code (a `KnnStore::add_raw` plus a `beta` spec field) was reverted, kept as
  `runs/key_surprise_diag.patch` (git-ignored). Remaining richer-value idea: store the next several bytes (option 2 of
  the survey), not tried.

- **Richer values: the next two bytes (2026-10-08): KILL.** Survey option 2 of the richer-values list. Each train key stores
  its next byte and the byte after it; the neighbours of the previous position vote for the current byte with their
  second byte (an induction-style continuation), mixed in with weight mu inside the kNN term. Flat memory, `part=3/6`,
  `knn:256:0.4:15`:

  | mu | **0** | 0.1 | 0.2 | 0.3 | 0.5 |
  |---|---|---|---|---|---|
  | CE | **1.1130** | 1.1212 | 1.1321 | 1.1443 | 1.1717 |

  Monotonically worse: the previous position's neighbours were chosen for a different context, so their continuation is
  noise next to the current position's own neighbours. Log `runs/tier_next2.log`; code reverted, kept as
  `runs/next2_diag.patch` (git-ignored). With the surprise-weighting result this closes the richer-values options of the
  survey (stored surprise, multi-byte continuation); what is left is outside the value: the key (tap block, a learned
  key) or the model itself.

- **Tap block under the current memory (2026-10-08): block 3 stays.** Online split memory, `part=3/6`, spec
  `knn:256:0.4:15:25:50` (parameters tuned at tap 3 only), model alone 1.1453:

  | tap | memory alone | + lexicon + words |
  |---|---|---|
  | 1 | 1.1557 (+0.0105) | 1.1502 |
  | 2 | 1.1110 (-0.0343) | 1.1063 |
  | **3 (last)** | **1.0929 (-0.0524)** | **1.0881 (-0.0571)** |

  Monotone in depth, so the final block is the best key and no earlier tap is worth retuning for. Logs
  `runs/tier_tap{1,2,3}.log`. The key is still a raw hidden state; a learned key remains untried.

- **Learned key, cheapest probe: a fixed whitening of the keys (2026-10-08): small gain, +0.003.** Keys and queries are
  centred and multiplied by Sigma^(-alpha/2) (symmetric, total variance kept, ridge 1e-3 of the mean eigenvalue), Sigma from
  128k train positions (the top 5% of the eigenvalues hold 66% of the variance); throwaway code (`runs/whiten_diag.patch`,
  `KNN_WHITEN=<alpha>`), online split memory, memory alone, `part=3/6`, `knn:256:0.4:T:5T/3:10T/3` (distance scale shifts
  with alpha, so the temperature is re-picked per alpha; best in **bold**):

  | alpha | T=8 | 15 | 22 | 30 | 45 | 60 | 90 |
  |---|---|---|---|---|---|---|---|
  | 0 (today; reproduces 1.0929) | 1.1038 | **1.0929** | | 1.1036 | | | |
  | 0.25 | 1.1173 | 1.0948 | **1.0905** | 1.0918 | 1.1010 | 1.1140 | 1.1412 |
  | 0.5 | 1.1276 | 1.0997 | 1.0914 | **1.0899** | 1.0948 | 1.1043 | 1.1300 |
  | 1 | 1.1370 | 1.1063 | 1.0950 | **1.0912** | 1.0954 | 1.1087 | 1.1506 |

  (alpha 0 at T=8/15/30 uses shifts 27/50/100; the others tie Td and shift to T as above, T=8 and 15 columns for alpha>0
  are the first grid with Td/shift 13/27 and 25/50.) Best gain **0.0030** (alpha 0.5, T 30: 1.0899 vs 1.0929), a smooth
  optimum between alpha 0.25 and 1. Memory alone, one slice, shift and Td only coarsely tied to T: promising rather than
  proven. A learned (supervised) projection could be worth more, but this probe says the raw metric is only mildly off. The
  fit costs ~25 s once and a 256x256 matrix multiply per key (8 threads); not yet a proper option.

- **Whitened keys, full held-out run (2026-10-08): +0.0020 on the stack, +0.0033 memory alone.** Same throwaway patch
  (`KNN_WHITEN=0.5`, `runs/whiten_diag.patch`), `runs/tier_full_whiten.log`, 656.7 s over 492 chunks (1.1-1.2 s/chunk; the
  machine was 96% busy with other sessions at launch, so timing is not clean), `lexicon:0.3+words:1:0.25+knn:256:0.4:T:Td:S`:

  | T:Td:S | stack CE | vs model |
  |---|---|---|
  | **30:50:100** | **1.1118** | **-0.0646** |
  | 30:40:100 | 1.1124 | -0.0639 |
  | 38:63:127 | 1.1131 | -0.0633 |
  | 30:50:70 | 1.1132 | -0.0631 |
  | 30:50:130 | 1.1132 | -0.0632 |
  | 22:36:73 | 1.1142 | -0.0622 |

  Memory alone 1.1163 (was 1.1196). Control without whitening: 1.1138 / 1.1196, so the slice's -0.0030 shrinks to -0.0020 on
  the stack, still consistent and smooth in the grid (flat within 0.0014 around the optimum, tuned on the same text as the
  score, so a small optimism). Cost: a ~25 s fit once plus a 256x256 multiply per key and query (8 threads); the search itself
  is unchanged (656.7 s vs 696 s is load, not a speedup). Not yet a proper option; the patch is the reference.

- **`whiten=<alpha>` is now an option; alpha 0.25 on the full run (2026-10-08): alpha 0.5 stays.** The patch became a
  proper option (default off; `jacobi_eigen` / `whitening_matrix` unit test; slice check `whiten=0.5`,
  `knn:256:0.4:30:50:100` = 1.0899, as the patch). Full run at alpha 0.25 (`scripts/tier_full_whiten.sh 0.25`,
  `runs/tier_full_whiten_0.25.log`, 648.6 s), stack CE: 22:36:73 **1.1121**, 30:40:100 1.1125, 30:50:100 1.1127, 38:63:127
  1.1165, 15:25:50 1.1173 (memory alone 1.1185). Alpha 0.5 best 1.1118, no whitening 1.1138, so the optimum is flat between
  0.25 and 0.5 (0.0003 apart) and the full-set gain from whitening is ~0.002 either way. Use `whiten=0.5` with
  `knn:256:0.4:30:50:100`. Best deployable stack: **1.1118 (-0.0646)**.

- **Loss by position class (`classes=1`; `scripts/tier_classes.sh`, `runs/tier_classes.log`, 2026-10-08).** Full held-out,
  `warm=32`, online split memory with `whiten=0.5`, 503808 positions, 637 s. Class = was the byte read a letter (letters and
  apostrophes), and what kind of byte is predicted. Gain columns are 1e-3 nats of the overall CE and sum to the overall
  gain (spec0 = lexicon + words, spec1 = memory alone, spec2 = whole stack, overall -0.0646):

  | class | share of positions | model CE there | share of model CE | lexicon+words | memory | stack |
  |---|---|---|---|---|---|---|
  | non-letter -> lower (word start) | 16.4% | 2.524 | 35.2% | 0.00 | 6.64 | 6.64 |
  | non-letter -> UPPER | 1.9% | 3.314 | 5.4% | 0.00 | 2.84 | 2.84 |
  | non-letter -> space | 2.6% | 0.200 | 0.4% | 0.00 | -0.10 | -0.10 |
  | non-letter -> newline | 0.8% | 0.725 | 0.5% | 0.00 | 0.12 | 0.12 |
  | non-letter -> other | 2.3% | 0.689 | 1.3% | 0.00 | 1.51 | 1.51 |
  | letter -> lower (mid-word) | 57.5% | 0.896 | 43.8% | 17.45 | 44.97 | 48.71 |
  | letter -> UPPER | 0.1% | 1.923 | 0.2% | 0.18 | 0.27 | 0.31 |
  | letter -> space (word end) | 14.0% | 0.371 | 4.4% | 1.38 | 2.85 | 3.26 |
  | letter -> newline | 1.1% | 1.839 | 1.7% | 0.04 | **-1.84** | -1.82 |
  | letter -> other (punctuation) | 3.3% | 2.530 | 7.0% | 0.70 | 2.77 | 3.11 |

  Findings: (1) the symbolic tiers only ever act mid-word (17.5 of their 19.7); at word starts they do nothing (the `f`
  flag did not help) and the memory recovers only 6.6 of the 414 (35%) of the CE sitting there, so word starts are the
  biggest untouched mass but predicting the next word is a language-model job, not a lookup. (2) Mid-word the stack
  adds 3.7 over the memory alone. (3) **Line breaks: the memory makes the newline-after-letter class worse (-1.84)**
  and the bytes at line ends carry 1.7% + 0.5% of the CE (~26e-3 nats) plus part of the word-end-space class. The books
  are hard-wrapped (median line 68-69 bytes, p90 71-73, max 71-77 over the six; 77-87% of the non-empty lines fall in
  60-75), longer than the model's 64-byte window, so the model often cannot see the last newline and has no column. A
  rule tier that tracks the column from CONTEXT bytes is the one symbolic tier with a measurable target.

### Line-wrap tier: the column rule (2026-10-08)

Column check (`classes=1` log, model alone): at newline targets with no newline in the 64-byte window the model's CE is
1.25-1.79 nats at columns 40-74 (4.1k of 6.2k such targets sit at >=56), against 0.39 where the window shows the
newline; at space targets the same split is 0.44-0.62 vs 0.28-0.31. The model cannot see the column, so it cannot
tell a line end from a word end.

`wrap:<lambda>` (`Wrap`, KR&R rule tier): where the window has no newline but the 256-byte context does, the model's
mass on newline+space is re-split by the train newline rate at that column (counts by [byte read is letter][column],
`(nl+0.5)/(nl+sp+1)`), mixed with lambda. Nothing else moves. Test
`wrap_resplits_newline_and_space_by_column_only_when_the_window_is_blind`; script `scripts/tier_full_wrap.sh`.

| specs | CE | vs model 1.1764 |
|---|---|---|
| model + wrap:0.2 / 0.4 / 0.6 / 0.8 / 0.95 | 1.1704 / 1.1665 / 1.1637 / 1.1617 / 1.1608 | -0.0060 ... -0.0155 |
| stack (lexicon+words+knn+whiten) control | 1.1118 | -0.0646 |
| stack, wrap:0.8 before knn / after knn | 1.1000 / 1.0964 | -0.0763 / -0.0799 |
| stack, wrap:0.95 before knn / after knn | 1.0988 / **1.0956** | -0.0776 / **-0.0807** |

After knn is better (knn mixes (1-lambda)p + lambda*knn, so wrap placed last corrects the final newline/space split).
Per class (1e-3 nats of overall CE): after-letter->newline goes from -1.82 (the memory made it worse) to +11.36, and
after-letter->space from 3.26 to 5.26; nothing else changes. Applying wrap to every position (not only window-blind
ones, `WRAP_ALL`) was worse and was removed. The wrap tier adds -0.0162 on top of the 1.1118 stack, the largest single
gain since the memory itself. Not tried: lambda 1.0, richer conditioning than the column (previous byte class,
indentation, blank lines), the same rule for the other classes with a visible target.


Tried and reverted (model-only, full held-out; the experiment code was removed, grid: wrap:0.95, 1.0, each with and without :w): wrap lambda 1.0 equals 0.95 (1.1608);
conditioning the rate on the column relative to the longest complete line in the 256-byte context (this book's wrap
width, `wrap:<lambda>:w`) is worse than the plain column, 1.1630 vs 1.1608 (-0.0134 vs -0.0155): a context shows only
2-3 lines, and the longest of them is a noisy width (paragraph ends, verse, indentation), so it blurs a column table that
already pools well. The plain column stays; the code was removed.

### Memory re-tune with wrap, and the punctuation class (2026-10-08)

Re-tune (`scripts/tier_full_wrap_retune.sh`, full held-out, stack + wrap:0.95 after knn, one coordinate at a time from
knn:256:0.4:30:50:100 = 1.0956): lambda 0.3 / 0.5 -> 1.0991 / **1.0950**; T 25 / 40 -> 1.0961 / 1.0983; T_doc 35 / 70 ->
1.0968 / 1.0965; shift 50 / 150 -> 1.0994 / 1.0991. The optimum did not move (only lambda 0.5, -0.0006, which is within
the noise of a tuning step); the specs keep 0.4.

Punctuation (read-only diagnostic, patch in `runs/punct_diag.patch`, not committed): the "after letter -> other" class
(7.0% of the CE) is commas 2.4%, periods 1.8%, the lead byte of UTF-8 punctuation (curly quotes, dashes) 1.35%,
semicolons 0.9%, `!` 0.5%, `-` 0.5%, `?` 0.3%. The model's CE there is 1.8-4.0 nats per target and the stack recovers
0.02-0.19 of it. The books use curly quotes only (no ASCII `"`), and the byte that decides open vs close (`\x9c` vs `\x9d`
after `\xe2\x80`) already costs 0.03-0.17 nats, so quote matching has nothing to win. What is left is where a clause or
sentence ends, which is language modelling. KILL: no rule tier for punctuation.

### Longer training, 16M windows (2026-10-08/09)

Question: is the model undertrained? Held-out CE of the 8M run at 1M-window marks: 1.3349 1.2870 1.2684 1.2578 1.2518 1.2475
1.2430 1.2408 (steps of 0.006, 0.004, 0.0045, 0.002 near the end, with the schedule decaying to the end); the train
probe is 1.1172 against 1.2408 held out, so the gap is large and more steps may mostly overfit (dropout 0.1 helped at
4M and 8M). `scripts/train_16m.sh` = the same recipe at `windows=16384000` (`nov_big_k1_d0.1_16m`; log
`runs/nov_big_k1_d0.1_16m.log`; started 19:16, killed once at 20:12 for a pause and resumed from the checkpoint, 5535 s of
wall time plus the gap; 6.75 ms/step in the training clock, ~9.7 in the log's rows because other sessions loaded the CPU).

Result: held-out CE **1.2319** (8M run 1.2408, -0.0089); train probe 1.0919 (1.1172), so the train/held-out gap rose from
0.124 to 0.140: more steps help, with diminishing returns (a doubling bought what the last 1M windows of the 8M run
bought in 4). Full stack (`scripts/tier_full_16m.sh`, `warm=32`, whiten 0.5, 773.6 s; CPU 44% from other sessions):

| | 8M model | 16M model | change |
|---|---|---|---|
| model alone | 1.1764 | 1.1651 | -0.0113 |
| + lexicon, words, memory | 1.1118 | 1.1041 | -0.0077 |
| + wrap:0.95 | 1.0956 | **1.0876** | -0.0080 |
| gain of all tiers over the model | -0.0807 | -0.0775 | the tiers add 0.003 less |

The better model carries most of its gain through the tiers (0.0080 of 0.0113); the tiers' own contribution shrinks a
little, as expected when the model learns some of what they supplied. Doubling the steps again would cost ~3 h for a
guessed -0.005; width at matched steps gave 0.002 (d = 384), so the next model-side lever is the window (option 3) rather
than more of the same.

### Training step profile: matmul-bound, not launch-bound (2026-10-09)

`fused_models_check ... profile=host|gpu` (new option: 50 steps after 20 of warm-up charged to launch sites; top 40 printed)
on the real config (K = 1, batch 32, d = 256, d_ff = 512, 4 blocks, dropout 0.1; CPU 55% and the eGPU shared with
other sessions, so absolute times are inflated; logs `runs/prof_host.log`, `runs/prof_gpu.log`). **177 launches per
step, of which 92 are matmuls.** GPU time per step under the profiler 10.55 ms, of which **matmuls 8.45 ms (80%)** and
everything else 2.10 ms (85 launches: softmax, layer norm, elementwise, split-k sums, cross entropy). The training run's own
clock was 6.75 ms/step. The largest sites: the FFN weight gradients and input gradients (`[256x2048]T·[2048x512]N`,
`[2048x768]N·[768x256]T` and the like, 0.5-0.85 ms each), then attention `[64x64]` batched matmuls. So fusing
elementwise launches can save at most ~2 ms of 10 (the non-matmul share), and the doc's earlier "dispatch-bound" label
(true at d = 128, batch 8: 1.97 ms) does not hold at this size: matmul throughput is the lever (these kernels are plain
f32; the kNN search already uses f16 matrix-core tiles).

### f16 matrix-core training matmul (2026-10-09)

Spike (`examples/gpu/matmul_cmma_spike.rs`, f16 x f16 -> f32 cmma, f32 operands converted on the way into shared memory,
multi-plane 64x64 to 128x128 tiles, shared eGPU, CPU 97-100% busy): 1.7-2.9x over the f32 `matmul` on the 10 dominant
shapes (2.38 ms -> 1.06 ms summed), max relative error 2.4e-4 to 3.2e-4; the spike's kernels reach 3-8 TFLOP/s on the training
shapes against ~15 on a 4096^3 probe (no vectorised loads, no double buffering, small grids, split-k sum launch).

Integration (commit 003ff86): `k_matmul_cmma` behind `TRAIN_F16=1` (default off; the f32 kernel is unchanged beside it). Used
when the device reports the cmma f16 config, there are no head views, k is a multiple of 32 and m, n divide a tile
(64, 64x128 or 128x128); all epilogues (bias, relu, mask, residual, accumulate) and the deterministic split-k for the weight
gradients. Routed: every Linear, QKV, FFN and projection forward and dX matmul and all weight gradients. Still f32: the
attention matmuls (head views, ~0.9 ms/step) and ragged shapes. Test `matmul_f16_matches_reference` (18 cases at 1e-2; not
vacuous: it fails at 1e-5 with max error ~3e-4 of scale) passes, as do the existing f32 tests. `device_step_matches_cpu_tape`
fails with the flag (block 0 off by 7e-4 against its 1e-4 bound; its 1e-5 ReLU-tie check cannot hold under f16 rounding), so
that test needs a looser f16 variant.

Real recipe (512000 windows, 16000 steps, d = 256, d_ff = 512, dropout 0.1; contended, CPU 50-58%, Defender off, back to
back): held-out CE within 0.0058 nats at all 65 rows (largest gap at step 1750, 1.8158 vs 1.8100), final 1.4056 (f32) vs
1.4022 (f16): the f32 curve is bumpy and f16 crosses it in both directions, so noise, not divergence. Training ms/step
**~10.7 -> ~7.0** (wall 258 s -> 183 s); GPU time per step (`profile=gpu`) 10.62 -> 6.43 ms (1.65x). Biggest sites f32 -> f16: the
FFN dX matmuls 0.55-0.84 -> 0.10-0.23 ms, the weight gradients 0.85 -> 0.50. What is left: `rows.rs` (0.43 ms), the f32
attention matmuls (0.37, 0.22, 0.18, 0.14) and `tokens.rs:51` (0.31).

Open: a longer f16 run (the 16M recipe) before making it the default; the f16 variant of the device-vs-CPU test; attention
matmuls and ragged shapes on cmma; the kernel's own headroom (vectorised loads, double buffering).

### f16 matmul, round 2: attention, double buffering (2026-10-09)

Commits 4ca2751, 7a0e1c7, 72834b1, 3f6944d. `k_matmul_cmma` now takes head views (group/inner addressing), so all four attention
matmuls run on cmma; a 64x32 tile serves n = 32 shapes (used only when no 64-wide tile fits: picking it for the weight
gradients first made them 3x slower, caught in the profile). Shared-memory stages are double-buffered (one barrier per k
stage), which dropped the 128-wide tiles (doubled stages exceed the 32 KiB shared-memory limit: the 128x128 launch failed
at 37,888 bytes); the 64x64 tile now uses 8 planes (4x2). Weight-gradient matmuls ~2x faster from double buffering
(`[256x2048]T.[2048x768]` 0.49 -> 0.22-0.27 ms); 4x2 planes ~5%; 4x4 planes fail to launch; doubling the split-k target
gave nothing. Matmuls now run at ~13-15 TFLOP/s, about the 4096^3 probe's ceiling (~15), so vectorised loads were not tried.

Tests: 4 of 4 matmul tests pass (re-run by the lead), including the new `matmul_f16_head_views_match_reference` (B = 2, H = 4,
t = 64, w = 32, 1e-2). Loss (512000 windows, real recipe, f32 then f16 back to back): max gap 0.0068 (step 14500), final gap
0.0014 (f32 1.4056, f16 1.4070), signed mean f16 - f32 +0.0023; the f16 curve is identical across the double-buffering and
plane-count changes, so those did not change the numerics. Speed (contended: CPU 24-75%, another session's GPU test;
Defender off): GPU time per step f32 10.5-10.7 -> f16 4.59 ms at light load (6.22 vs 14.72 at heavy load), ratio 0.42-0.44;
training ms/step 10.75 -> 6.36 (light), 11.02 -> 7.72 (heavy). Left in the f16 profile (ms/step): `rows.rs:128` 0.43,
`tokens.rs:51` 0.31, `rows.rs:96` 0.25, attention `256x[64x64]T.[64x32]N +=` 0.25 (~1 TFLOP/s, latency-bound; several
matrices per cube might save 0.2-0.3 ms), QKV forward 0.24, then mostly non-matmul rows.rs / tokens.rs / tape.rs work.
Still open: a longer f16 run before making `TRAIN_F16` the default, and a loosened f16 `device_step_matches_cpu_tape`.

## Order

0. `tier_eval.rs` with Gate 0 (needs the GPU for a forward pass only; the d = 384 run is using it, shared lease).
1. Experiment A (smallest, most distinctly a CPU/symbolic tier).
2. Experiment B.
3. Only then: how the tiers interact (a tier gating another, working-memory contents), which needs A and B's
   numbers to be worth designing.

Open questions for the user: is the lexicon a reasonable first "KR&R" content, or should the CPU tier hold
something else (rules the LM must satisfy, e.g. quote/bracket matching)? Is the final residual stream the right
"working memory" to expose to the long-term tier?

## Fused matmul + threshold filter: matrix-core tile into shared memory (2026-10-02)

Spike for the fused-kernel option (survey 3). `cmma::store` into `Shared::<[f32]>::new_slice(256)`, then `sync_cube()`, then
per-lane reads and a `plane_sum` count of entries below a threshold: the tile matches the CPU product (1e-3) and the count
matches (test `matrix_core_tile_goes_through_shared_memory`, f16 x f16 -> f32, 16x16x16, passes on the RX 9060 XT). So the
main unknown is settled: the accumulator tile can be filtered in-kernel without writing `dist` to global memory. Still
unmeasured: the speed of a full fused kernel (threshold lookup, survivor compaction via `Atomic<u32>::fetch_add` or
`plane_exclusive_sum`, a per-tile survivor buffer feeding a smaller top-k). Guess ~2x on search; not started.
