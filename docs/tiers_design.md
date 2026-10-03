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

- **Run in flight (started 2026-10-02 18:40, PID 38072):** `scripts/tier_full4.sh` -> `runs/tier_full4_online.log`.
  Full held-out, `warm=32 stride=1 memory=online`, six specs (lexicon, words, lexicon+words, knn, words+knn,
  lexicon+words+knn). Expected ~30-45 min (30 s build + ~6 x 4-5 min of search); CPU ~60% from other sessions at
  launch, Defender real-time off. Result goes in the next entry.

## Order

0. `tier_eval.rs` with Gate 0 (needs the GPU for a forward pass only; the d = 384 run is using it, shared lease).
1. Experiment A (smallest, most distinctly a CPU/symbolic tier).
2. Experiment B.
3. Only then: how the tiers interact (a tier gating another, working-memory contents), which needs A and B's
   numbers to be worth designing.

Open questions for the user: is the lexicon a reasonable first "KR&R" content, or should the CPU tier hold
something else (rules the LM must satisfy, e.g. quote/bracket matching)? Is the final residual stream the right
"working memory" to expose to the long-term tier?
