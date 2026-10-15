# Tiered AI: first cross-tier experiments

Direction and tier map: README, "Direction: tiered AI". Related work: README, "Tiered and neuro-symbolic
architectures". Status: step 0 and Experiment A built and measured (2026-10-02); Experiments A and B and a second CPU tier (word bigram) measured on the full held-out split.

## Stack history (moved from the README 2026-10-14; restated as a table)

Held-out CE in nats/byte, whole held-out text unless noted; each row adds to the one above and is superseded. The current stack
(0.9754 / 1.0691 per half, IVF memory of 125M 2-bit keys) is in the README, "Current best stack"; each step is detailed below.

| Step | Model alone | Stack |
|---|---|---|
| novels6 model, standard score: lexicon -0.0061, kNN memory over all 4.5M train positions -0.0291 (k = 256), word bigram -0.0161 | 1.2408 | 1.2053 (-0.0356) |
| `warm=32` (every byte gets at least 32 bytes of context); lexicon + bigram + past-only in-document memory | 1.1778 | 1.1311 (-0.0467; memory -0.0425, CPU tiers -0.0201) |
| online memory (written as the text is read, no held-out text in advance), stride 1 | 1.1764 | 1.1313 (-0.0451; memory alone -0.0412) |
| `write=first`, separate in-document weighting and search (top-256 each) | | 1.1138 (-0.0626) |
| whitened keys (`whiten=0.5`) | | 1.1118 (-0.0646) |
| line-wrap rule tier (`wrap:0.95`) | | 1.0956 (-0.0807) |
| 16M-window model | 1.1651 | 1.0876 (held-out 1.2319) |
| 32M windows, f16 training | | 1.0844 (held-out 1.2307) |
| 22-novel training corpus | | 1.0821 (held-out 1.2193) |
| dynamic evaluation (`dyn=<lr> dyn_every=<windows>`: one SGD step per scored chunk, reset per book), tiers re-tuned | | 1.0575 |
| 128-byte window (`seq=128`; the wrap rule becomes redundant and is dropped) | | 1.0392 |
| memory of 100000 windows instead of 50000 | | 1.0352 |

Facts about the memory: its gain is document-specific (unseen text of the same books helps most, -0.036 from a 472k-key memory; text of
other books hurts). Search speed: the first full runs took 2502 s in f32 (2 h 38 min once) because the online memory wrote one tiny
device tile per window; fixed by searching pending keys as one transient tile, then the f16 matrix-core search gave 826 s for the
whole held-out run (~1.7 s per chunk, flat to 5M keys). That search was exact; the current one is approximate (IVF, recall ~0.99).

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

### Bias gradients folded into the dW matmul (2026-10-09)

Commit 35b93db. `Epilogue::col_sum` makes the weight-gradient matmul also sum dY's columns (the bias gradient) in the m-tile-0
cubes over their k-slice: no split-k adds it straight into the bias location, split-k writes per-slice partials that
`k_split_sum` adds in slice order (deterministic, no float atomics); the f16 kernel sums the raw f32 values before the f16
cast. All 17 bias gradients (QKV, out-proj, FFN1, FFN2 in 4 blocks, final projection) are folded, `col_sum` and its kernel
and test are deleted, and an f32 split-scratch size that looked undersized for batch > 1 was corrected (`batch*splits*m*n`).
Tests: 19 of 19 `gpu_step` ignored tests pass with `TRAIN_F16` unset (re-run by the lead); with `TRAIN_F16=1` only the known
`device_step_matches_cpu_tape` fails. Loss (512000 windows, real recipe, fresh baselines from a worktree): new f32 final 1.4028
vs baseline 1.4056 (max gap 0.0041), new f16 1.4057 vs 1.4070 (max gap 0.0043), new f16 vs new f32 max gap 0.0085: float-order
drift, no divergence. Profile (`TRAIN_F16=1`, contended, CPU 40-90%, Defender off): launches 177 -> 160, GPU 4.81 -> 4.24
ms/step, host 9.37 -> 5.86 ms/step (the largest host site, 0.86 ms, is gone); training ms/step at low CPU load 4.64 -> 4.49 and
4.84 -> 4.61 (~4%). Left: `tokens.rs:51` 0.31, layer-norm backward `rows.rs:97` 0.26 and `:94` 0.18, the attention
`256x[64x64]T.[64x32]N` 0.25, then the matmuls (0.16-0.25 each).

### Small step costs: embedding backward, layer-norm params, `+=` attention (2026-10-09)

Commits 6bb0e1b, d59ce9d, 7889d66 (each extends a unit test; 19 of 19 `gpu_step` ignored tests pass, re-run by the lead;
with `TRAIN_F16=1` only the known `device_step_matches_cpu_tape` fails). (1) `tokens.rs:51` is the token-table backward, not a
scatter: each unit scanned all 2048 ids with dependent global loads. Cubes now stage the ids in shared memory in 512-id chunks,
same add order (bit-identical): 0.43 -> 0.16 ms/step. `tape.rs:525` is `k_dropout`, 16 elementwise launches of ~9 us, i.e.
launch floor; left. (2) `k_ln_bwd_params` has only 16 cubes and is load-latency bound: four rows' loads in flight per lane,
same add order: 0.25-0.34 -> 0.148 ms. Fusing the two reductions in `k_ln_bwd_dx` into one barrier tree gave nothing (0.212 vs
0.210) and was reverted; the forward (0.09-0.11 ms) was not touched. (3) Packing several attention matrices per cube would not
help: the 256 matrices already run in one wave, so the cost is per-launch latency (~18 us). The `+=` attention launches cost
~36 us (a dependent read-modify-write per element behind barriers); with no other epilogue, `out` is now loaded into the
accumulator fragments and stored directly (float-order change only): `+=` attention site 0.235 -> 0.215 ms, the N.N one
0.150 -> ~0.107 ms. Loss (512000 windows, builds back to back): curves identical row by row for f32 and f16 (max gap 0); finals
1.4028 / 1.4057 as before. Profile (`TRAIN_F16=1`, contended, CPU 56-100%, Defender off): GPU 4.18 / 4.13 -> 3.86 / 3.82 ms/step;
160 launches unchanged; host and training ms/step within noise (f16 5.60 -> 5.36, f32 11.57 -> 11.15). Floor now: launch count
(160) and per-launch latency (~18-36 us for the small matmuls); the next real gain is fusing (attention QK^T-softmax-PV, layer
norm with the neighbouring matmul or residual).

### f16 training at 8M windows: slower convergence (2026-10-09)

`scripts/train_8m_f16.sh` (`TRAIN_F16=1`, the 8M recipe, log `runs/nov_big_k1_d0.1_8m_f16.log`; machine contended, CPU ~37% at
start, another session's shared lease). Held-out CE at the 1M-window marks, f32 (`nov_big_k1_d0.1_8m`) vs f16: 1.3349 / 1.3361,
1.2870 / 1.2869, 1.2684 / 1.2729, 1.2578 / 1.2665, 1.2518 / 1.2609, 1.2475 / 1.2558, 1.2430 / 1.2512, **1.2408 / 1.2475**. The f16
curve is behind by up to 0.0087 (4M) and ends **+0.0067** worse; the train probe is 1.1430 against 1.1172, so f16 also fits the
training text worse (+0.026): this is slower optimisation, not overfitting or noise. The 512000-window validations (gap
<= 0.0085, final 0.0014) were too short to show it. Cost: training clock 4.33 ms/step against 11.4 (f32, contended; both runs
shared the machine), total 1109 s of training against 2928 s. Reading: f16 at 8M windows is about as good as f32 at ~6M, so
matching f32's 8M result costs roughly 11M windows, still ~2x less training time. **Do not make f16 the default yet.**
Candidates for the loss: f16 rounding of the activations and gradients that feed the weight gradients (dW is Xt.dY with both
operands rounded), the f16 forward, or the attention matmuls (the first validation ran before attention joined; the
512000-window gap stayed small throughout). An ablation by matmul class (forward only, dX only, dW only, attention only in f32)
on a 2M-window run would say which one matters.

### f16 ablation by matmul class, 2M windows (2026-10-09)

`scripts/ablate_f16_classes.sh`: six 2M-window runs (`runs/abl2m_*.log`), `TRAIN_F16=1` with one class kept on f32 through a
throwaway flag (`F16_F32` bits; patch in git-ignored `runs/f16_class_ablation.patch`, source reverted). Machine contended
(CPU ~100% at times), single seed, so timings are rough and CE differences of ~0.001 are near noise.

| run | held-out CE | train probe | train ms/step |
|---|---|---|---|
| all f32 | 1.2849 | 1.1934 | 10.7 |
| all f16 | 1.2869 | 1.1948 | 4.4 |
| keep forward f32 | 1.2858 | 1.1946 | 6.8 |
| keep dX f32 | 1.2866 | 1.1950 | 7.5 |
| keep dW f32 | 1.2859 | 1.1922 | 7.7 |
| keep attention f32 | 1.2859 | 1.1946 | 5.7 |

The gap at 2M windows is only +0.0020 (it is +0.0067 at 8M: it grows with training length). No single class explains it:
forward, dW and attention each recover about half (~0.0010), dX almost none (0.0003). dW is the one that also fixes the
train probe (1.1922, below f32). Attention is the cheapest to keep in f32 (+1.3 ms/step; keeping forward or dW costs
+2.4 to +3.3). Reading: a diffuse rounding effect rather than one bad class; the 8M run with attention in f32
(`scripts/train_8m_f16_keepattn.sh`) tests whether the cheapest fix closes the 8M gap.

**Result (8M windows, attention in f32, `runs/nov_big_k1_d0.1_8m_f16attn.log`):** held-out CE **1.2427** against f32 1.2408 and
all-f16 1.2475, so the gap shrinks from +0.0067 to **+0.0019**; the train probe is 1.1170 (f32 1.1172, all-f16 1.1430), so the
fit to the training text is fully recovered. Training clock 4.93 ms/step (all-f16 4.33, f32 11.4; contended, one session's
jobs in parallel), 1261 s against 1109 s and 2928 s. So attention is the class that mattered at length (it accumulates over
long training; the 2M ablation could not show that), and keeping only it in f32 keeps ~2.3x the f32 speed. The patch is
`runs/f16_class_ablation.patch` (git-ignored); making this the recipe means replacing the flag with "attention matmuls
always f32 under `TRAIN_F16`" (done in 5cb6392, `matmul_f32`).

**Plus dW in f32 (8M, `scripts/train_8m_f16_keepattn_dw.sh`, `runs/nov_big_k1_d0.1_8m_f16attn_dw.log`):** held-out CE **1.2417**
(attention only 1.2427, f32 1.2408), train probe 1.1164 (1.1170), 7.0 ms/step (4.9; contended, CPU ~50%). The 0.0010 gain is
within single-seed noise and costs ~40% more step time, so dW stays f16. Recipe: `TRAIN_F16=1`, attention f32. The residual
+0.0019 is not attributed to a class (forward and dX were not run at 8M); `device_step_matches_cpu_tape` now also runs under
`TRAIN_F16=1` (6e-2 bounds, worst gradient 5.4e-2, ReLU flips allowed).

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

### f16 is the training default (2026-10-09)

`TRAIN_F16` now defaults on (`TRAIN_F16=0` gives the f32 kernels), attention always f32, on devices with f16 matrix cores; the
earlier "default off" and "TRAIN_F16=1" wording in the entries above is history. All 29 `gpu_step` tests pass both ways.
Cost: ~0.0019 held-out CE at 8M windows (single seed) for ~2.3x the step speed. The 16M recipe is rerun in this mode by
`scripts/train_16m_f16.sh` (reference: f32 16M, CE 1.2319).

### 16M model trained in f16: the full stack (2026-10-10)

`scripts/train_16m_f16.sh` (f16 default, attention f32; `runs/nov_big_k1_d0.1_16m_f16.log`): held-out **1.2333** against f32
1.2319 (+0.0014; the 8M gap was +0.0019, so it did not grow with the longer schedule), train probe 1.0919 (1.0906); 2848 s of
training, 5.56 ms/step (contended), against ~2.4 h for f32. Best stack on it (`scripts/tier_full_16m_f16.sh`, same specs as
the f32 16M stack): model alone 1.1684 (f32 1.1651), + lexicon/words/knn 1.1057 (1.1041), + wrap **1.0892** (**1.0876**). So the
tiers give the same gain on the f16 model (-0.0792 against -0.0775) and the stack ends +0.0016 behind, at ~2x less training time.

### Small step costs round 2, and fused f32 attention (KILL) (2026-10-10)

Fresh profile of the f16 default (attention f32, contended, CPU 45-70%): 4.21 ms/step over 38 sites: f16 linear matmuls 1.9
(45%), f32 attention matmuls 0.89 + softmax 0.17 (25%), layer-norm/softmax rows 0.54, embedding/dropout/ReLU-mask/optimiser
~0.6. Done: weight decay folded into the momentum kernel (`DeviceParams::momentum(.., decay)`, the same arithmetic, one
launch fewer) and the dropout backward applying the ReLU mask in the same launch when the dropout is the ReLU node's only
gradient source (guarded: `Node::relu_masked`, asserted in `add_grad`): GPU **4.21 -> 4.15 ms/step** (-1.4%).
**Fused f32 attention: killed.** One cube of 64 units per (sample, head), unit i = query row i, K/V/scores or dS in shared
memory (the exact 32 KiB: dS XOR-swizzled), scores + softmax + weights.V in one launch and a one-launch backward (dQ, dK, dV
and dS together). Correct (matches an f64 reference to 1e-5, all `gpu_step` tests pass, the 1e-4 device-vs-CPU step test
included), but **2.7 ms/step against 1.05 for the matmul + softmax chain it replaces**: first version (private arrays
indexed at run time) 5.6 ms (spilled to scratch); with comptime head width, unrolled channel loops and scores in shared
memory 2.7 (fwd 0.56, bwd 2.14). The structure is the limit: each unit does a serial chain of dot products through
shared memory and the causal triangle idles half the lanes of a wave; a better version needs several units per row or
row-pairing (i, 63-i) plus a split of the channel sum, for a best case ~1.3 ms, still above the existing path. The
attention matmuls' real cost is f32; the f16 matrix-core path is what made the linear layers cheap. Patch and kernel text:
`runs/fused_attention_attempt.patch`, `runs/attention_fused.rs.txt` (git-ignored); source reverted. Parked: attention
sites split by precision (scores vs weights.V vs backward, each an 8M run) to move part of the 0.89 ms to f16; and a
quiet-machine re-time, which the GPU-time column cannot replace for the host side.

### 32M model in f16: the full stack (2026-10-10)

`scripts/train_32m_f16.sh` (the 16M recipe at 32768000 windows, f16 default, attention f32; `runs/nov_big_k1_d0.1_32m_f16.log`):
held-out **1.2307** at 5471 s of training (5.34 ms/step, contended), against 1.2333 at 16M (f16) and 1.2319 at 16M (f32): a
doubling buys 0.0026 and already beats the 16M f32 model, in about 1.5 h. Held-out at 4M/8M/16M/24M: 1.2583 1.2447 1.2339 1.2323;
the train probe falls faster (1.0756 against 1.0919 at 16M), so the held-out curve is flattening while the fit to the training
text keeps improving: the novels6 corpus, not the training time, is becoming the limit. Best stack (`scripts/tier_full_32m_f16.sh`):
model alone 1.1633, + lexicon/words/memory 1.1008, + wrap **1.0844** (16M f16 1.0892, 16M f32 1.0876; best so far). The tiers add
-0.0789 on this stronger model (-0.0792 at 16M f16, -0.0775 at 16M f32), so their gain is stable as the model improves.

### More training text: novels22 at 8M windows (2026-10-10)

`scripts/fetch_gutenberg_extra.sh` fetches sixteen more Gutenberg novels (10 MB, Austen, Dickens, Doyle, Twain, Wells, Hardy,
Conrad, the Brontes, Wilde, Stevenson) into git-ignored `data/gutenberg_extra/`; corpus `novels22` = novels6 (same
held-out text, step-0 CE 6.2606 identical) plus those books as training only (14.5 MB; 8M windows is ~35 passes instead of
~112). `scripts/train_8m_novels22.sh`, f16 default, 8M windows: held-out **1.2454** against 1.2427 on novels6 (+0.0027,
worse); at 4M 1.2679 against 1.2580. The extra text dilutes the style the held-out books share, and at 8M the novels6 model
has not yet started to overfit enough for more text to pay (train probe 1.1170 against held-out 1.2427 at 8M; the 32M
novels6 model has 1.0756 against 1.2307). 4.56 ms/step. So at this training length more text does not help; whether it helps
at 32M windows (where novels6 flattens) is untested (~2.7 h).

### novels22 at 32M windows: the full stack (2026-10-10)

`scripts/train_32m_novels22.sh` (32768000 windows, f16 default, 4.75 ms/step, 2h09m wall, contended): held-out **1.2193**
against 1.2307 on novels6 (-0.0114); train probe 1.1329 against 1.0756 (less memorisation). Held-out by marker, novels22 vs
novels6: 4M 1.2679/1.2583, 8M 1.2454/1.2447, 16M 1.2300/1.2339, 24M 1.2252/1.2323: the extra text loses early and wins once
novels6 starts to overfit (crossover between 8M and 16M). Stack (`scripts/tier_full_32m_novels22.sh`): model alone 1.1478
(1.1633), + lexicon/words/memory 1.0981 (1.1008), + wrap **1.0821** (1.0844; new best, -0.0023). The model gained 0.0155 but
the stack only 0.0023: the tiers add -0.0657 here against -0.0789 on novels6. That run used the default `corpus=novels6` for
the lexicon, words and memory (the checkpoint only fixes the model), so they knew none of the extra books. Rerun with
`corpus=novels22` for them (`scripts/tier_full_32m_novels22_corpus22.sh`): lexicon misses fall from 1.40% to 0.77% of held-out
words, but the stack is **1.0827**, not better (model 1.1478; + lexicon/words/memory 1.0986; + wrap 1.0827; tiers -0.0651).
The memory is capped at `store=100000` evenly spaced windows, so a larger train pool thins the sampling rather than adding
keys, and a bigger lexicon did not help: more text reaches the stack only through the model. Best stack stays 1.0821.

### Where the stack's remaining CE sits: survey (2026-10-10)

`tier_eval ... survey=1` (last spec; `survey_report`, test `survey_splits_letters_by_lexicon_and_recency`) splits the CE of the
1.0827 stack by target kind. Of 1.0827 nats: letters of words **in the train lexicon 83.4%** (first letter 42.4%, 2nd-3rd 31.1%,
4th+ 9.9%); punctuation `. , ; : ! ? -` 6.5%; space 4.7%; other bytes 1.4%; newline 1.2%; quote/bracket/marks 0.3%; digits 0.1%;
words not in the lexicon 2.1% (new to the book 1.9%, seen earlier in the book 0.3%). Consequences:
- An in-document word cache has a ceiling of **0.0011 nats** (every seen-earlier out-of-lexicon letter brought down to the
  in-lexicon mean CE at the same word position): dead end with a 22-novel lexicon (0.77% of held-out words are out of it).
- Structural rules (quotes, brackets, newline, digits, other) hold 3.0% of the CE (0.033 nats) in total, so even a perfect rule
  tier gains under 0.03 and a realistic one a few thousandths.
- What is left is choosing the next word among words the lexicon knows: 18% of positions (word-initial letters) carry 42% of the
  CE at 2.5 nats each. That is syntax/semantics, not vocabulary or layout. A KR&R tier that matters has to predict word
  classes or topics (the grammar or residual-mined-rule options), not store more words.

Hand reading of 61 word-initial targets (`survey=1` prints a fixed-stride sample with the stack's top 3 bytes; log
`runs/tier_full_32m_novels22_corpus22_samples.log`, git-ignored). The stack is 2-3.5 nats at most of them and its top 3 hold
only 0.25-0.5 of the mass: open choices ("my imagination was vivid, yet my | powers", "the grimness", "in the | site"),
where no short-context rule or word-class prior would move the target far; no recurring pattern a rule could name (the
sentence-initial capital and the opening of a quotation are uncertain by nature). The one visible lead is rare in-lexicon
names that recur in a book: Hertfordshire 5.5 nats, Greenland 6.2, Mast(-head) 3.7. They are in the train lexicon, so the
"seen earlier" ceiling above (out-of-lexicon words only) did not count them; whether an in-document recency boost for in-lexicon
words has room is unmeasured (split first-letter CE by "word occurred in the last N bytes" and by train frequency).

### Recency of in-lexicon words at word starts (2026-10-10)

`survey=1` also splits the stack's CE at word-initial targets in the lexicon (91641 positions) by the word's train count and
the distance to its previous occurrence in the same book (`runs/tier_full_32m_novels22_corpus22_recency.log`; stack 1.0827):

| train count | prev <=500 B | prev <=5000 B | prev farther | none earlier |
|---|---|---|---|---|
| <20 | 3.56 (233) | 3.66 (306) | 3.27 (514) | 3.42 (3719) |
| 20-199 | 3.18 (1293) | 3.18 (1814) | 3.16 (3314) | 3.22 (7396) |
| >=200 | 1.97 (34178) | 2.48 (23102) | 2.90 (10625) | 3.00 (5147) |

(mean stack CE in nats, position count in brackets.) For rare and mid-frequency words the CE does not fall when the word
occurred a few hundred bytes earlier (3.56 against 3.42 for rare; 3.18 against 3.22 for mid), so the stack does not use a
repeat as a cue for the first letter, and neither does the online memory at these positions. The common words do fall with
recency (1.97 near, 3.00 none), but that mixes in predictability. Room: 7474 rare and mid-frequency word starts that repeat a
word from the book, at 1.5% of the positions; each nat saved on all of them is 0.0148 nats of overall CE, so a cache that
saves a third of a nat on them is worth about 0.005: the same size as the whole novels22 effect on the stack (0.0023) or the
wrap rule's remaining gain, and the only measured KR&R lead. Untested; the cache is a recency prior over the first letters of
book words not frequent in the train text, mixed in like `words`.

## Direction: more capability at the current scale (2026-10-10)

Set by the user: stay at about 2M parameters (4 blocks, d=256, ff=512, 64-byte windows) and look for new ideas, not larger
models. Ideas, ranked by how directly they meet the survey above (the stack's CE sits on word-initial letters of known
words, 42%; it does not use repeats of rare words in the book):
1. **Dynamic evaluation** (fast weights; the short-term-memory tier): score a chunk, take a gradient step on the bytes just
   scored, carry the weights through the book, reset between books. Extra cost ~80 s per held-out pass. Open: learning rate
   (tune on a disjoint split), reset points, interaction with the kNN keys (the frozen model's train keys vs adapted queries).
   **Started 2026-10-10.**
2. In-model memory: kNN attention over earlier hidden states of the book (Memorizing Transformers), trained end to end.
3. Auxiliary heads aimed at word starts (next 2-4 bytes, next word's first letter or class); free at inference.
4. Hashed previous-word / prefix embeddings as extra inputs (capacity as lookups); adds parameters. **Closed without a run** (2026-10-14, see "Hashed previous-word embeddings").
5. Train with the tiers in mind: distil the stack into the model, or train on the residual the tiers leave (the 32M novels22 model
   gained 0.0155 alone but 0.0023 in the stack, so the two overlap). Earlier mutual distillation (same-size teacher) hurt.
6. Context-dependent tier gating: **killed** (2026-10-07/08: kNN weight gate +0.001, words weight gate +0.0004 held-out; weights are within ~0.001 of optimal; see "Verdict" under `gate_fit`).
Also open: the recency prior for rare in-lexicon words (~0.005 est.; see "Recency of in-lexicon words").

### Dynamic evaluation, model alone (2026-10-10)

`tier_eval ... dyn=<lr>` (`adapt_step`; `scripts/dyn_eval_sweep.sh`, `dyn_eval_sweep2.sh`; logs `runs/dyn_sweep/`, git-ignored): after each
32-window chunk is scored, one plain SGD step (no dropout, f16 default) on that chunk, which is text already scored; the
parameters are re-uploaded from the checkpoint at each book boundary. The 32M novels22 model, warm=32, no tiers. `dyn=0`
reproduces the baseline (halves 1.0969 / 1.1987, mean 1.1478 = the full-pass value). Held-out CE by half (part=0/2 chooses,
part=1/2 confirms) and the control (each step taken from a chunk half the held-out text away, i.e. another book):

| dyn lr | half 0 | half 1 | mean | control mean |
|---|---|---|---|---|
| 0 | 1.0969 | 1.1987 | 1.1478 | |
| 0.01 | 1.0914 | 1.1893 | 1.1404 | 1.1459 |
| 0.05 | 1.0810 | 1.1769 | 1.1290 | 1.1468 |
| 0.1 | 1.0741 | 1.1692 | 1.1217 | |
| **0.2** | **1.0672** | **1.1624** | **1.1148** | 1.1557 |
| 0.4 | 1.0683 | 1.1638 | 1.1161 | |

Both halves pick lr 0.2: **-0.0330** (-2.9%) on the model alone, 491 steps per pass, a few seconds of extra time. The control
(adapting on another book) helps a little at 0.01 (-0.0019), is neutral at 0.05 (-0.0010) and hurts at 0.2 (+0.0079), so the gain is from the book being read, not from re-tuning
the checkpoint. Model alone with dynamic evaluation (1.1148) is already within 0.033 of the whole tier stack (1.0821).
Not yet tried: the stack on top (the memory keys come from the frozen model), RMS-normalised steps, more or smaller steps,
a pull back toward the checkpoint.

### Dynamic evaluation: finer steps, and under the tier stack (2026-10-10)

Model alone, mean of the two held-out halves (`dyn_every=<windows per chunk>`, `dyn_steps=<k per chunk>`;
`scripts/dyn_eval_sweep3.sh`, `dyn_eval_sweep4.sh`; baseline 1.1478, 32-window chunks at lr 0.2 gave 1.1148):

| windows per step | lr 0.005 | 0.01 | 0.02 | 0.03 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|---|
| 16 | | | | | 1.1204 | 1.1126 | 1.1103 |
| 8 | | | | | 1.1114 | 1.1077 | 1.1208 |
| 4 | | | 1.1132 | 1.1086 | **1.1061** | 1.1191 | 1.1740 |
| 2 | | 1.1128 | 1.1057 | **1.1055** | | | |
| 1 | 1.1127 | **1.1055** | 1.1094 | | | | |

The best learning rate falls with the chunk size (0.2 at 32 windows, 0.05 at 4, 0.03 at 2, 0.01 at 1; collapse above it) and
the optimum plateaus at **1.1055-1.1061 (-0.042, -3.6%)**: finer steps beyond 4 windows add nothing at plain SGD. Several steps
per chunk are no better than one finer step (2 steps at 0.1: 1.1152; 4 steps at 0.05: 1.1159). Halves agree on the choice.

The stack with dynamic evaluation (`scripts/tier_full_32m_novels22_dyn.sh`; tiers built from the default novels6 train split as the
1.0821 run, memory train keys from the frozen model, queries and in-document keys from the adapted one; 32-window chunks, lr 0.2):
model alone 1.1138, + lexicon/words/memory 1.0876, + wrap **1.0719** (new best; 1.0821 without). The tiers now add -0.0419
against -0.0657 (-0.0789 on the novels6 model): adaptation takes over part of what the memory and lexicon gave, as the
in-document memory and fast weights both learn the book. The stack could not use the finer chunks: the tier machinery works
in 32-window chunks (online memory writes), so the finer optimum (1.1055 model alone) is not yet reachable under it.

### Dynamic evaluation under the stack at finer steps; RMS steps (2026-10-11)

The tiers had to become chunk-size independent first: `dyn_every=<windows>` now sets the scoring / online-write / adapt chunk for
the tiers too. Gate: the stack without dynamic evaluation on held-out half 0 gives 1.0366 at both 32 and 8 windows per chunk
(`scripts/tier_dyn_chunks.sh`, `runs/dyn_tier/`). Stack on half 0 with dynamic evaluation: 32 windows at lr 0.2 **1.0271**; 8 windows
at 0.1 **1.0245**; 4 windows at 0.05 **1.0235** (no dynamic evaluation 1.0366).

Optimiser, model alone (`Adapter`: `dyn_rms=<beta>` divides each gradient by a bias-corrected running RMS, `dyn_pull=<f>` pulls
each parameter back towards the checkpoint; a host-side prototype that reads gradients and parameters back every step, ~2-3x slower
than the device SGD; gate: `dyn_pull=1e-12` reproduces the device SGD value 1.0577 exactly). Half 0 at 8-window steps (SGD best
1.0595): RMS beta 0.99 lr 1e-4 **1.0568** (beta 0.9: 1.0572; lr 3e-4 1.0749, 5e-5 1.0640, 3e-5 1.0711, 1e-3 diverges); a pull of
0.001 is neutral (1.0567), 0.01 hurts (1.0601). At 4-window steps lr 1e-4 gives half 0 **1.0550** and half 1 **1.1511**, mean
**1.1031** against 1.1055 for the best SGD: the optimiser is worth ~0.002.

Full-pass stack with dynamic evaluation at 4 windows per step (`scripts/tier_full_32m_novels22_dyn4.sh`; tiers from the novels6 train
split; memory train keys from the frozen model):

| dynamic evaluation | model alone | + lexicon/words/memory | + wrap |
|---|---|---|---|
| none | 1.1478 | 1.0981 | 1.0821 |
| 32 windows, SGD 0.2 | 1.1138 | 1.0876 | 1.0719 |
| 4 windows, SGD 0.05 | 1.1054 | 1.0848 | **1.0689** |
| 4 windows, RMS 1e-4 | 1.1022 | 1.0846 | **1.0688** |

Dynamic evaluation now buys -0.0132 on the stack (-0.0426 on the model alone); a model gain of 0.0084 from finer steps and RMS reaches
the stack as 0.0030, and RMS's extra 0.0032 on the model alone as 0.0001: the memory and the adapted weights learn the same book. The
tiers add -0.0364 on top of the adapted model (-0.0657 on the frozen one). Plain device SGD at 4-window steps is the setting to keep
(RMS ties it and is a slow host prototype). Not tried: rebuilding the memory's train keys from the adapted model (~70 s per
rebuild), adapting only some layers, and a different tier mix now that the model alone is close to the stack.

### Memory keys from the frozen model, and re-tuning the tiers for the adapted model (2026-10-11)

**Keys.** `dyn_keys=frozen` takes the hidden states the memory uses (queries, in-document keys) from an extra forward of the
unadapted checkpoint, so they live in the same space as the train keys; probabilities still come from the adapted model.
Half 0, 4-window SGD 0.05: 1.0232 against 1.0235 with adapted hidden states: no effect (`scripts/tier_dyn_keys.sh`). The key-space
mismatch does not matter; adapted hidden states are kept (one forward).

**Tier mix.** `scripts/tier_dyn_retune.sh` (one parameter at a time on half 0, base 1.0235): the in-document temperature 30
instead of 50 gives 1.0170 (-0.0065; 80 hurts: 1.0443); knn lambda 0.3 1.0222; words lambda 0.15 1.0218; k 128 1.0219; knn temp
20/45, lexicon eps 0.2/0.5 and words 0.4: no gain. Round 2 (`tier_dyn_retune2.sh`, from lexicon 0.3 + words 0.15 + knn lambda 0.3):
in-document temperature 30/20/15/10 -> 1.0151/1.0141/1.0141/1.0144; shift 50/100/150/200 (at 20) -> 1.0172/1.0141/1.0183/1.0269;
**k 128 1.0130**; words 0.1 1.0137; lambda 0.25 1.0149. Half 1 confirms (`tier_dyn_confirm.sh`): old tuning 1.1145, new
(words 0.15, `knn:128:0.3:30:20:100`) **1.1030** (-0.0115; half 0: 1.0235 -> 1.0130, -0.0105). Full pass
(`scripts/tier_full_32m_novels22_dyn4_tuned.sh`): model alone 1.1054, + lexicon/words/memory 1.0731, + wrap **1.0575** (new best; 1.0689
with the old tuning, 1.0821 without dynamic evaluation). The memory now gets -0.0323 and wrap -0.0156 on the adapted model; the
lower in-document temperature makes the in-book neighbours count for more, the same direction as the adaptation itself. The
tuning is one-at-a-time and two rounds deep; the other parameters (whiten, tap, store) and the wrap lambda are not re-tuned.

**Round 3 and which parameters to adapt.** `scripts/tier_dyn_retune3.sh` (half 0, base 1.0130): wrap lambda 0.9/0.98 1.0131/1.0130; query
temperature 25/35 1.0133/1.0141; shift 75/125 1.0131/1.0150; knn lambda 0.35 1.0129; k 96 1.0127; whiten 0.75 1.0125, whiten 0.25 1.0150;
tap=2 1.0386. Every gain is at most 0.0005: the tier mix has converged, so no full pass. `dyn_layers` masks the adapted parameters
(model alone, half 0, 8-window plain SGD lr 0.1, `scripts/dyn_eval_sweep7.sh`; the all-parameter mask reproduces device SGD, 1.0597
against 1.0595): all 1.0597; **blocks only 1.0590**; b2+b3+head 1.0643; b3+head 1.0714; head only 1.0862 (lr 0.3: 1.0848); embedding
only 1.0942. The blocks carry the adaptation, the more of them the better, and embedding/head add nothing; restricting to a few layers
is not a shortcut. Low-rank adapters are not tried; with blocks-only already as good as everything, they would at best match it.

**Context-length probe.** `scripts/ctx_probe.sh`: the same recipe (novels6, d=256, dropout 0.1, f16, 8M windows of 64 bytes = 524M
bytes, equal bytes per step and equal step count; batch and warmup scale down with seq), seq 64/128/256 via the new `seq` option of
`fused_models_check`. Held-out CE scores the same target bytes for every seq (`score_tail=32`: the last 32 positions of windows
ending every 32 bytes, windows start at 256 - seq). Held-out **1.1784 / 1.1524 / 1.1418** (-0.0260, then -0.0106; -2.2%, -0.9%);
train-probe 1.0377 / 1.0084 / 1.0008. Cost per step 4.7 / 5.1 / 6.9 ms (same bytes per step;
wall time 45 / 39 / 49 min). Longer context helps the model itself, with diminishing returns
(each doubling buys about 40% of the previous gain). The tiers read 256 bytes of context already, so part of this gain may
overlap theirs; not yet measured (tier_eval has `SEQ_LEN` 64 as a constant, and its windows/`warm` assume it).

**Context length under the tiers.** `tier_eval` takes `seq=` (default 64; window length, the launch chunk shrinks as 32*64/seq).
Gate: the probe models alone on the same protocol (`warm = seq - 32`) give 1.1785 (seq 64) and 1.1418 (seq 256), matching the
trainer's 1.1784 / 1.1418. `scripts/ctx_probe_tiers.sh`, novels6, held-out half 0, static stack (no dynamic evaluation),
store fixed in bytes:

| seq | model | + lexicon/words/memory | + wrap |
|---|---|---|---|
| 64 | 1.1315 | 1.0693 | 1.0532 |
| 128 | 1.1039 | 1.0381 | 1.0381 |
| 256 | 1.0953 | 1.0360 | 1.0360 |

The model-alone gain (-0.0276, then -0.0086) shrinks under the tiers: the memory's gain falls from -0.0622 to -0.0592 at seq 256,
and the stack gains -0.0151 at 128 and only -0.0021 more at 256. Wrap rule is now redundant: with a 128-byte window the window
shows a newline whenever the context does ("applied at 0 positions"), so it is dropped from the stack at seq >= 128. Seq 128
captures nearly all of the stack gain at 1.07x the step cost, so the 32M novels22 model is retrained at seq 128
(`scripts/train_32m_novels22_seq128.sh`).

**Seq-128 novels22 model, full pass.** `scripts/train_32m_novels22_seq128.sh` (same step count and bytes per step as the seq-64
model; 1.0M steps, 5.3 ms/step, 1h30 training): held-out 1.1189 on the shared protocol (tail 32 bytes; the seq-64 model, same
protocol, 1.1478). `scripts/tier_full_32m_novels22_s128.sh`, `warm=96`, no wrap (redundant), full held-out:

| | model | + lexicon/words/memory |
|---|---|---|
| static, old tiers (words 0.25, knn:256:0.4:30:50:100) | 1.1189 | 1.0647 |
| static, tuned tiers (words 0.15, knn:128:0.3:30:20:100) | 1.1189 | 1.0599 |
| dynamic evaluation, SGD 0.05, 2-window steps (256 bytes) | 1.0848 | 1.0468 |
| dynamic evaluation, SGD 0.05, 4-window steps (512 bytes) | 1.0722 | **1.0392** |

New best **1.0392** (1.0575 at seq 64: -0.0183, -1.7%). Context length and dynamic evaluation compose: dynamic evaluation takes
-0.0467 off the model and -0.0207 off the stack (seq 64, 256-byte steps: -0.0424 off the model, 1.1478 -> 1.1054).
Caveat: the memory used `store=50000` windows of 128 bytes (6.4M keys) against 4.5M keys at seq 64, a small advantage not
separated here; the tiers were tuned at seq 64 and not re-tuned for seq 128 (one-at-a-time tuning, as above, is next).
Larger steps again won at the longer window (4 windows beat 2), so 8 is untried.

**In-model memory: parked, with reasons.** Three variants were considered. (1) Cached previous-window K/V (Transformer-XL): it
extends the reach to 2 windows, which the seq 128 -> 256 probe already measures (model -0.0086, stack -0.0021), so the headroom
is about 0.002 and it needs new attention kernels (scores [T, 2T], offset causal mask, stop-gradient cache) in `gpu_step`.
(3) Learning when to trust the memory: the gate fit on entropy and neighbour distance (above) gains 0.001 over the best
fixed lambda, so there is little to learn at that level. (2) A kNN-attention layer trained end to end (Memorizing Transformers,
Wu et al. 2022) is the only variant that changes what the model can represent, but the survey puts the remaining CE in
word-initial letters of known words and in new words, which a book-level memory does not contain; it is the largest build for
the least evidenced gain. None is started. Revisit if the context or the tier gains stop shrinking.

**Re-tune at seq 128 (half 0, dynamic evaluation 4 windows SGD 0.05, model alone 1.0246, stack 0.9941).**
`scripts/tier_s128_retune.sh`: the tiers are flat. One at a time: lexicon 0.2/0.45 0.9939/0.9944; words 0.1/0.2 0.9941/0.9945; knn
lambda 0.25/0.35 0.9945/0.9943; k 96/192 0.9939/0.9947; temperature 25/35 0.9948/0.9949; in-document temperature 15/30
0.9942/0.9949; shift 75/150 0.9939/1.0005. No gain above 0.0002: the seq-64 tuning carries over unchanged. Adaptation step 8
windows: SGD 0.05 0.9968, 0.1 0.9949 (both worse than 4 windows, 0.9941; the model alone 1.0304/1.0261 against 1.0246).
**Memory size is a lever**: store 35000 windows (4.5M keys) 0.9965, 50000 (6.4M) 0.9941, 100000 (12.8M) **0.9902**
(memory gain -0.0281, -0.0306, -0.0344), model alone unchanged. So the 0.0183 improvement over the seq-64 stack included about 0.0024
from a bigger memory (6.4M against 4.5M keys); the seq-128 gain itself is a little smaller than the headline. 100000 windows
is 12.8M f16 keys = 6.5 GB of the 16 GB eGPU, near the practical limit for a second resident job.

Full pass with store 100000 (`scripts/tier_full_32m_novels22_s128_store100k.sh`): model alone 1.0722, stack **1.0352** (-0.0370
from the memory; 1.0392 at store 50000).

The memory is capped by the training text: novels22 holds 116,733 windows of 128 bytes (14.9M keys, 7.6 GB f16). The whole of it
on half 0 gives 0.9893 (store 100000: 0.9902; `s128_storemax`), so the size lever is spent (-0.0009 more, no full pass run;
the full-pass number stays 1.0352). More memory now means more text, not a larger `store`.

**Step size at the 100k memory.** SGD lr 0.03 / 0.05 / 0.07 at 4-window steps, half 0: 0.9919 / 0.9902 / 0.9909. 0.05 stays.

**Memory-only extra text.** `scripts/fetch_gutenberg_memory.sh` (18 more Gutenberg novels, 23 MB, `data/gutenberg_memory/`, gitignored;
corpus `novels40` = novels22 + them, training only, same held-out) and `tier_eval memory_corpus=novels40` (memory keys from that
text; model, lexicon, words and whitener stay on novels22). At the same key budget as the whole of novels22 (116,733 windows =
14.9M keys; the windows are sampled evenly, so each book is covered at a third) half 0 gives **0.9898** against 0.9893 for
the contiguous novels22 memory: no gain. At a fixed budget, 2.2x more text (sampled) buys nothing over full coverage of the
smaller text. The key budget is VRAM (16 GB eGPU; 14.9M f16 keys of 256 floats = 7.6 GB, 38M keys for all of novels40 would be
19 GB), so more text can only help through smaller keys (fewer dims by PCA of the whitened keys, or int8), not tried.

**Dropout 0.2 at seq 128 (stopped).** Held-out at the same steps, dropout 0.1 / 0.2: 300k 1.1505 / 1.1677; 500k 1.1317 / 1.1606;
744k 1.1229 / 1.1459. The gap (0.017, 0.029, 0.023) was not closing, so the run was stopped at 73% (no checkpoint kept). The
train/held-out gap is not what limits the model at this schedule; more dropout only slows learning.

**Key compression (`whiten_rank`).** Keeping only the top-r eigen-directions of the whitened keys (half 0, store 50000, base 0.9941):
rank 128 **0.9990** (memory gain -0.0306 -> -0.0257; r 64/32 not run, they only get worse). Halving the key width would allow twice
the keys in the same VRAM, which is worth about -0.0039 (store 50000 -> 100000), so compression loses about 0.001 net. The keys
carry their information across the whole spectrum; PCA truncation is not the way to a bigger memory. (int8 keys, half the size of f16, are untried.)

**Eval priority (A/B).** `scripts/prio_ab.sh`: the same eval (seq 128, half 0 / 4, store 50000, dynamic evaluation) alternating
Normal and BelowNormal, with another session's `cargo` on the CPU (~35% load): Normal 366 s / 375 s, BelowNormal 431 s / 384 s
(identical CE 0.9566 in all). BelowNormal costs 3-18%, most when the machine is busy, which is when the priority matters; a
disk cache of the memory keys was judged not worth it (the build is 70-110 s of 400-600 s, and 100k windows of f16 keys is 6.5 GB
per file). Training alone runs at 4.7-5.3 ms/step; with an eval on the GPU 9.6-12.4 ms/step: do not overlap GPU jobs.

**Where an eval spends its time.** `timing:` on the half-0 / 4 run: tier prepare (the kNN searches) is 95% of the scored loop (268 of
283 s); model forward and dynamic-evaluation steps 3%, row scoring 1%. `KNN_TIMING=1` (synced stages), per chunk of 128 queries over
6.4M keys: the train search costs ~285 ms matmul + ~390 ms top-k; the in-document tail search 12 ms. Per query row the cost is the
same for 16-window and 4-window chunks (1.8 vs 2.1 ms), so it is not scan-bound; 3.3 GB of distances is written and read per
chunk at ~10 GB/s against 320 GB/s of VRAM bandwidth. `KMAX` 256 -> 128 (the tiers use k = 128; same CE 0.9566) saves 9% of the
search (243 vs 268 s). A real speedup (2-5x on every eval) would fuse the matmul with the thresholded top-k so the distance matrix
is never materialised; not started. `KMAX` is left at 256 since `knn:256` specs need it.

**kNN search knobs (negative result).** `scripts/knn_knobs.sh`, `knn_tile.sh` (seq 128, half 0 / 32, 123 chunks of 128 queries over 6.4M
keys; tier-prepare seconds, same CE 1.0745 throughout): `KNN_PARTS` 16 / 32 (default) / 64 / 128: 45.3 / **30.5** / 35.6 / 55.6;
`KNN_SLICE` 8192 / 16384 / 32768: 33.1 / 30.5 / 29.9; tile 4096 / 16384 / 65536 / 262144: 34.6 / 30.5 / 33.3 / 34.6. The defaults are
at the optimum and the search is not launch-bound (4x fewer matmul launches changes nothing), so the time is in the kernels
themselves: the cmma matmul (one plane per 16x16 output tile, no shared-memory tiling, ~1.5 TFLOPS) and the per-thread top-k
insertion (58% of the synced search time). A better matmul would save at most the ~42% it takes; a fused kernel is a research
project for perhaps 1.5x. Parked; the cheaper levers are not overlapping GPU jobs and `KMAX` 128 (-9%).

**Approximate kNN search (IVF), adopted for evals.** Dynamic evaluation never lets the tiers influence adaptation, so `tier_eval`
now collects hidden states first and runs the searches in batches (`batch=<chunks>`, default 1 = the old per-chunk path, unchanged).
`ivf=<cells>:<probes>` clusters the train keys (k-means on the key sample, chain-ordered cells, cell-ordered storage; ~29 s once) and
searches each query's `probes` nearest cells (`src/gpu_step/knn/ivf.rs`); queries are sorted by nearest cell and grouped so cells
share matmuls, and a sampled threshold prunes list insertions as in the exact search. The in-document keys are still searched exactly.
`ivf_check=<n>` prints the recall of n queries against the exact search. Gates (seq 128, store 50000, 1024 cells x 8 probes,
`batch=128`): recall of the exact top-128 0.972 / 0.986 / 0.993 at 4 / 8 / 16 probes (0.9972 on half 1);
CE identical to exact on half 0 (**0.9941**) and half 1 (**1.0860**) and on `part=0/32` (1.0745); tier prepare 30.5 -> ~18 s on
`part=0/32` (the batching alone gives 1.3x) and 526 -> 131 s on half 1 (whole eval 555 -> 160 s, 3.5x). Remaining cost: the exact
in-document tail search, which was slow because of a bug: the kNN sample that gives each query its threshold was taken when a tile
was uploaded, so an online memory's in-document keys (still in the pending tile) were never sampled and every tail query ran with an
infinite threshold (found with a debug print of the thresholds: finite for 0 of 2048 queries; a denser tail sample "changed nothing"
for the same reason). The sample is now taken when keys are added. Tail top-k per block 2100 -> ~100 ms; `part=0/32` tier prepare
`ivf=1024:8 batch=128` 19 -> **11.1 s** (CE 1.0744), exact `batch=128` ~23 -> **15.3 s** (CE 1.0745), exact `batch=1` unchanged
at ~30 s (a chunk's window is too small to have samples). The IVF matmul operands are gathered on the device (matmul stage 1.8 -> 1.0 s).
Use `ivf=1024:8 batch=128` for long evals; keep the exact path for final numbers.

**int8 keys are not needed; a bigger store is, and IVF makes it cheap.** Simulated key quantisation (per-key symmetric scale, rounded
then kept as f16; `part=0/32`, IVF `1024:8`, store 50000): CE 1.0745 at f16 / **1.0746 at 8 bits** / 1.0745 at 6 / 1.0749 at 4, so
quantised keys cost nothing, but the point of halving the key size (twice the keys in the same VRAM) is moot here: the corpus
holds only ~117k windows (14.9M keys). With IVF (2048 cells x 8 probes, f16) the store size no longer drives the search time:
store 50000 / 100000 / 200000 (= all 117k windows): CE 1.0744 / **1.0714** / **1.0705**, tier prepare 11.1 / 8.8 / 8.0 s (the
build grows: k-means + reorder 30 / 65 / 78 s). So `store=0`-style "everything" is -0.0040 vs 50000 for free; int8 only matters
for a larger corpus. The simulation code was removed; `scripts/ivf_eval.sh` takes `STORE=`.

**Full store on the held-out halves.** `store=200000` (all ~117k windows, 14.9M keys), IVF 2048 cells x 8 probes, `batch=128`, seq-128 model,
dynamic evaluation: half 0 CE **0.9893** (store 50000 exact: 0.9941, -0.0048), half 1 CE **1.0810** (1.0860, -0.0050); recall of
the exact top-128 0.990 / 0.997; whole eval 158 s / 159 s (tier prepare 126 / 127 s) against 555 s for the exact store-50000 run.
IVF knob scan at the full store (`part=0/32`, tier prepare, CE, recall): 1024x8 10.6 s, 1.0707, 0.991; **2048x4 9.6 s, 1.0704, 0.973**;
2048x8 ~8-11 s, 1.0705, 0.990; 2048x16 11.2 s, 1.0705, 0.995; 4096x8 10.4 s, 1.0708, 0.989. CE is flat across all of them (recall
0.97 is enough), so the knobs are not worth tuning further; 2048x8 stays the default. Long evals: `store=200000 ivf=2048:8 batch=128`.

**int8 keys, implemented, and the whole of novels40 in the memory (correction to the two entries above).** The int8 note above called
int8 moot because it only considered novels22; the earlier `memory_corpus=novels40` entry, however, was stuck on a VRAM budget
(38M f16 keys = 19 GB). `KnnStore::set_int8` (`tier_eval int8=1`) now stores each tile as int8 with one scale per key (half the
memory; `k_dequant` expands the rows a matmul needs into an f16 scratch; norms are those of the dequantised f16 values; the
IVF reorder moves bytes). `part=0/32`, store 200000, IVF 2048x8: CE **1.0705**, identical to f16, 10.0 s tier prepare, recall 0.990.
Unit test `int8_keys_search_like_f16_keys`. Then `memory_corpus=novels40 store=1000000 ivf=4096:8 batch=128 int8=1`, seq-128 model,
half 0, dynamic evaluation: 36.7M keys (9.4 GB on the device), CE **0.9841**, against 0.9893 for novels22's 14.9M keys and
0.9941 for store 50000 exact, so the extra 18 books now give -0.0052 (the fixed-budget test found nothing, because it
sampled the same number of windows). Recall of the exact top-128 0.992; the eval took 189 s (tier prepare 160 s); the build took
140 s of k-means and assignment + 59 s of reorder, with ~14 GB of host RAM at peak. More text keeps helping at about -0.005 per 2.5x.

**seq-128 retrain at 64M windows (done 2026-10-12 23:05; `scripts/train_64m_novels22_seq128.sh`, 21,823 s, `runs/nov22_s128_k1_d0.1_64m_f16_m0.ckpt`).**
Evals (`scripts/`-style: seq 128, dynamic evaluation 4 x 0.05, `batch=128`, full stack; half 0 / half 1): model alone **1.0165 / 1.1146**
(32M model, half 0: 1.0246); stack with the novels22 memory (`store=200000 ivf=2048:8`) **0.9840 / 1.0765** (32M: 0.9893 / 1.0810, i.e. -0.0053 /
-0.0045); stack with the novels40 memory (`int8=1 ivf=4096:8`) **0.9792 / 1.0728** (32M half 0: 0.9841, -0.0049). Recall of the exact
top-128 0.987-0.999. Against the store-50000 exact stack this work started from (0.9941 / 1.0860) the best stack is now -0.0149 /
-0.0132, with an eval of ~3 min instead of ~9. Doubling the training windows is worth about -0.005 on the stack (-0.008 on the model alone).

**4-bit keys.** `set_key_bits(bits)` generalises the int8 storage to 8 or 4 bits (two values per byte, one scale per key; `tier_eval
keybits=8|4` replaces `int8=1`; test `quantised_keys_search_like_f16_keys`). The 64M-window model, half 0, `novels40` memory (36.7M keys),
IVF 4096x8: **CE 0.9793 at 4 bits against 0.9792 at 8 bits** (f16 would be 18.8 GB; 8 bits 9.4 GB; 4 bits **4.7 GB**), recall 0.991, eval
195 s. Keys cost nothing down to 4 bits, so the same VRAM now holds ~4x the f16 key count (~150M keys, ~1.2M windows, ~150 MB of
text); the corpus, not the device, is again the limit: more text means more books (a download, not yet done), and the k-means
build and its ~7 GB of host RAM per 37M keys would need checking at that size.

**Memory cache and device-side whitening (2026-10-13).** `KnnStore::save/load` (`knn/cache.rs`; test `saved_store_reloads_identically`) write the built
store (tiles, scales, norms, values, key sample, IVF index) to `memcache=<dir>/<hash>.mem` under a key string naming the checkpoint (path, length,
mtime) and every build option; a mismatch rebuilds. novels40, 4-bit, 36.7M keys: build 605 s -> **14 s load** (5.76 GB file, 7.6 s to save), same CE
(0.9793 half 0, 1.0729 half 1), recall 0.991 / 0.997. The cold build's forward + readback phase (309 s) was mostly the host whitener (2.4 TFLOP of
scalar code) plus the raw f32 readback: `forward()` now whitens on the device (`whiten_device`, f32 matmul with bias -W mu; the host
`apply_whitener` is deleted) and reads back whitened rows only. Cold build 370 s -> **129 s** (forward + readback 309 -> 74 s; host copy + norms 49-54 s
unchanged); short-eval CE identical (1.0745 both). Quantising on the device before readback (to save the 4x readback) is not done: the
remaining 74 s is mostly forward compute, and the host copy + norms (49 s) is the next largest phase.

**Host-side build costs (2026-10-13).** Quantising and norming keys is per-row independent, so `tile_of` now spreads it over 8 threads, and the IVF reorder carries
each key's norm with it instead of recomputing from the packed bytes (`tile_of_quant` takes norms; `unpack` deleted). novels40 4-bit, half 0, cold:
host copy + norms 49 -> 23 s, IVF reorder 56 -> 20 s, memory build 605 -> **277 s** with whitening on the device (forward + readback 75 s, k-means +
assignment 151 s); CE 0.9793 and recall 0.9913 unchanged. What is left of a cold build is k-means + assignment (151 s, matmul-bound: ~77 TFLOP at ~1.5 TFLOPS)
and forward + readback (75 s).

**Survey result: the matmul was not the bottleneck; `k_argmin` was (2026-10-13).** Timing the pieces in isolation (16384 keys x 256 x 1024 queries): the current `k_dots_cmma`
runs at 11-12 TFLOPS (not the ~1.5 inferred end to end), and the shared-memory tiled training `matmul` at 17-19 TFLOPS (1.5-1.7x). The assignment step per 16384-key tile
against 4096 centroids cost dots 3.1 ms, but `k_argmin` **43 ms** (one thread per row, neighbours `ld` floats apart: uncoalesced), i.e. ~100 s of the 151 s
"k-means + assignment". `k_argmin` is now one 256-thread cube per row reducing through shared memory (0.9 ms; test `nearest_centroid_matches_cpu_argmin`).
novels40 4-bit half 0, cold: k-means + assignment 151 -> **28.7 s**, memory build 277 -> **145 s** (605 s before the memcache/whitening/host-copy work), CE 0.9793 and
recall 0.9913 unchanged. A faster matmul kernel for `dots` would now save only a few seconds of the remaining 29 s (parked: not worth a new f16-input tiled kernel).
Remaining cold build: forward + readback 69 s, IVF 55 s (k-means + assignment 29 s, reorder 19 s), host copy + norms 21 s.

**Eval-side timing (2026-10-13; novels40 4-bit half 0 from the memory cache, scored eval 185 s -> 171 s, CE 0.9793 / recall unchanged).** `KNN_TIMING=1` syncs between
stages, so it overstates launch-bound work: with it, the IVF matmul stage looked 100x off the FLOP count (5.5 s per 16384 queries; one dequant + one tiny matmul per
probed cell and piece), and a 4x larger workspace (groups of 2990 queries) cut that stage to 2.3 s, yet the unsynced eval did not move (85.6 -> 83.5 s in IVF calls);
the constants stay at `WS_FLOATS` 64M / `GROUP_MAX` 2048. `KNN_WALL=1` prints unsynced wall time per `search_ivf` / `search_from` call (use that, not the stage timers, to judge a
change). What did help: (1) the tail (in-document) search ran the whole 600k-key threshold sample matmul for every 2048-query block (210 ms of 360) although only sample
keys inside the query ranges count; chunks outside the window `[min lo, max hi)` are now skipped like the tiles (block 360 -> 145 ms, ~16 s of the eval);
(2) the per-query top-`nprobe` cell selection runs on 8 threads (~4 s). Of the 171 s eval, ~82 s is `search_ivf` (5 s per 16384 queries, GPU side: per-cell dequant
+ small matmuls + scan; not yet split without syncs), ~65 s other `search` calls (23 s of that is the `ivf_check` exact searches, a recall measurement; the rest the tail search).

**Parked: quantise on the device before the readback (2026-10-13).** Device->host reads run at ~2.5 GB/s, so the 37.6 GB of f32 keys of a 37M-key build cost ~15 s of the 69 s
"forward + hidden readback"; quantising first would save ~13 s of a cold build that the memory cache now makes a one-off. The rest is forward compute. Not worth a device-side
`add` API; revisit only if cold builds of much larger memories become routine.

**Where the IVF eval time goes, and the device-side list merge (2026-10-13).** Skipping the dequant, the matmuls and the scan (temporary flags, short eval, unsynced wall) cut a `search_ivf`
call only from ~5.3 to ~4.5 s: the GPU stages are not the bottleneck, the host side is (reading back `PARTS` top-256 lists per query, 2.1 GB per 16384 queries at 2.5 GB/s, then a
`select_nth` merge per query). `PARTS` trades the two: the scan is thread-bound (2 / 4 / 8 / 16 / 32 / 64 parts: 51 / 27 / 16 / 10.7 / 12.2 / 21.9 s per two calls). `k_merge_parts` now merges
the lists on the device (one thread per query, shared by `search_block` and `scan_group`), so only 256 entries per query come back: IVF calls 10.7 -> 9.4 s, other searches 8.1 -> 6.7 s per
two calls; 32 parts is still not better, so `PARTS` stays 16. Together with the earlier changes, the scored half-0 eval (novels40 4-bit, memory from the cache, no `ivf_check`) is **136 s**
(185 s at the start of this round), CE 0.9793. `ivf_check` (recall against two exact searches over 300 queries, ~23 s) is opt-in: leave it out of routine runs.
Remaining per 16384-query call (~4.5 s): probe selection (270 MB read + select), per-group host index building and uploads, the sample-threshold pass; not yet split.

**Split of a `search_ivf` call, unsynced (2026-10-13; ~5.9 s per 16384 queries).** Probes 0.5 s; per group the host enqueues the dequant + matmul launches (~1.5 s per call in total), then waits for
the GPU: the sample pass ~0.5 s, the scan kernel and list merge ~3 s per call (about 54% of the call). Groups are small: a nominal 747 queries exceed the workspace after
padding each cell's queries to 16 and are split twice to ~187, so the scan launches only ~3000 threads and a call makes ~85 groups. A 4x workspace (`WS_FLOATS` 256M floats, `GROUP_MAX` 8192; groups of ~700, 25 groups per two calls)
cut a short-eval IVF call 5.9 -> 4.1 s (eval 20.7 -> 19.1 s) but the full half-0 eval moved only 136.0 -> 136.7 s (noise), so the 64M / 2048 constants stay (512M floats = 2 GB fails: over the device buffer limit).
Left for later if the eval needs to be faster: a scan kernel with more threads per query (the parts count is capped by the host merge no longer, `k_merge_parts` handles up to 64), or fusing the per-cell dequant into the matmul.

**Scan tuning: bigger groups, fewer parts (2026-10-13; half 0, novels40 4-bit, from the cache, no `ivf_check`).** With the device-side list merge, the scan's group size and part count were re-swept
together. `WS_FLOATS` 256M floats / `GROUP_MAX` 8192 (groups of ~700 queries instead of ~187 after the workspace splits; 1 GB of device memory; 512M floats = 2 GB fails the device buffer limit)
and `PARTS` 8 (lists per query; fewer parts mean a cheaper merge, less readback and less list initialisation, while the larger groups already supply the threads). Scored eval
(`PARTS`: eval s / `search_ivf` s over 16 calls): 16 with the old 64M workspace 136 / 82; 16 with 256M 128.5 / 62.9; 32 159.5 / 95.5; **8 117.1 / 53.9**; 4 125.9 / 64.1; 2 166.5 / 100.9. CE 0.9793 throughout.
The earlier "larger workspace is within noise" note (one full run, 136.0 -> 136.7 s) was the old 16 parts with a host-side merge still in the way; with the current code it is worth ~8 s.

### Tail search split (2026-10-13)
Half-0 eval from cache, `KNN_WALL=1` plus a temporary queue flush before each tail `search_from`: the queue is empty on entry (1-2 ms), so the ~2.6-2.9 s per 16384-query tail call (16 calls, ~43 s) is the tail search's own work, not queued adaptation. Unsynced laps inside `search_block` put most of it (28 s of ~48 s) at the pass-1 threshold `read(&thr_d)`, 12.7 s in the final read, 7 s enqueuing pass 1; the synced figure (~145 ms/block) is lower than the ~228 ms/block unsynced wait, cause not found. Untested suspects: the per-block INF/zero fills uploaded from the host (`best_d`, `best_v`, `thr_d`: parts*q*KMAX floats each) and `k_thresh` latency. Laps were temporary and removed.

### Device-side fills (2026-10-13)
The suspect in the entry above was right: `best_d`, `best_v`, `thr_d` (parts*q*KMAX floats each, per block) were built on the host and uploaded over USB4. `knn::filled(len, v)` (empty buffer + `k_fill`) replaces all five such uploads in `search_block` and `search_ivf`. Tail search per 16384-query call 2.6-2.9 s -> 0.9-1.2 s; scored half-0 eval 137.3 s -> 108.6 s (tier prepare 110 -> 77 s), CE 0.9793 unchanged (CPU ~18% at start; shared GPU not checked by counter, no leases). The ivf-path fills shrank too but were not timed separately.

Split after the fills (same run, `KNN_WALL=1`, wall per call): 16 `search_ivf` calls 56.8 s (3.2-3.5 s each), 16 tail `search_from` calls 18.3 s, together 75 s of the 77 s tier prepare. The IVF scan is now ~76% of it; the tail is 24%.

### Inside `search_ivf` (2026-10-13)
Unsynced host laps (temporary, removed), half-0 eval from cache, 16384-query calls, early calls (calls 6+ drifted 20-30% slower, likely contention): of ~2.7 s per call, probe selection (centroid matmul + readback + CPU top-8) 0.5 s (18%), matmul enqueue loop 1.1 s (40%; it blocks on a full queue, so it is mostly GPU dequant + cmma time), scan + list merge wait 0.9 s (32%), sample pass 0.1 s, gather prep 0.08 s, host vector build and fills ~0. Candidates: overlap the CPU probe selection of the next call with the GPU work of this one (up to ~0.5 s x 16 = 8 s), shave the dequant (parked fused kernel, <= 10-15 s total).

### Probe selection overlapped (2026-10-13)
In `search_ivf`, the CPU top-`nprobe` selection of centroid block b now runs on its 8 threads while the main thread fetches block b+1's centroid distances (matmul + readback). `search_ivf` calls (16): 56.8 s -> 44.8 s (early calls 2.35-2.55 s vs 2.7-3.0 s before); scored half-0 eval 108.6 s -> 87.1 s, CE 0.9793 unchanged. The eval gain is larger than the ivf gain (forward was also 4.5 s faster that run), so about 12 s is this change and the rest is run-to-run variation; CPU/GPU contention was not checked for either run (no leases).

### Faster stack on half 1 and the 32M model (2026-10-13)
Same options as the half-0 run (`runs/t11.sh`, `runs/t12.sh`; novels40 4-bit memory, ivf 4096:8, `store=1000000`, dyn 0.05/4). Half 1, 64M model, memory from cache: scored eval **73.4 s** (forward 12.3, tier prepare 52.9), CE **1.0729** (model alone 1.1146), same as before. 32M-window model (`nov22_s128_k1_d0.1_32m_f16_m0.ckpt`), half 0: cold memory build 157.6 s (forward + readback 70.7, host copy + norms 19.9, other 67.1), then scored eval **74.1 s**, CE **0.9843** (model alone 1.0246): the 64M model is better by 0.005 on this half with the stack. Timings single runs, contention not checked (no leases).
Overlapping the tail search with the IVF scan (step 3) was not built: per half the 16 `search_ivf` calls (44.8 s) plus the 16 tail calls (~17 s) already add up to the 63 s tier prepare, so there is no idle gap between them for an overlap to fill; both are GPU-bound on one queue.

### Fused dequant+matmul: measured upper bound, not built (2026-10-13)
Throwaway bench (ignored test, removed): one 4-bit tile, a cell-sized region of 9008 rows x 256, 64 query columns, 400 back-to-back launches then one sync. `k_dequant` alone 0.049 ms per cell, `k_dots_cmma` alone 0.040 ms (~7 TFLOP/s, far below the matrix cores' peak), dequant + dots 0.068 ms (the two overlap on the queue). Fusing can save at most the 0.028 ms difference: x ~8200 cell-launches per 16384-query call (2 groups x 4096 cells) = ~0.23 s per call, ~3.7 s per half, and the fused kernel would dequantise each A tile once per query column tile, giving part of that back. Not worth a new kernel. The modelled 0.56 s per call for dequant + dots is about half the measured 1.1 s enqueue loop, so the rest is probably per-launch and host overhead (2 launches per cell piece, ~16k per call), a better target than the arithmetic: fewer, larger launches (one dequant per group instead of per piece, or batched cmma over cells) is the untested idea.

### Launch overhead in the cell loop: measured, not worth restructuring (2026-10-13)
Throwaway bench (removed), same cell-sized region as above, 2000 back-to-back repetitions, host enqueue time vs time to a final sync: dequant + dots with a fresh f16 buffer per cell 0.058 ms enqueue / 0.072 ms total; with one reused scratch buffer 0.052 / 0.063; dequant alone 0.020 / 0.028; dots alone 0.023 / 0.031. So a launch costs ~20 us of host time and ~28-31 us on the queue even for 1-2 MB of work, i.e. the loop is at the per-launch floor and the arithmetic is a minor part. Batching the dequants (a 128-cell chunk per launch, the 2 GB buffer limit forbids a whole group) and reusing a scratch buffer would save about 0.01 + 0.02 ms per cell = ~0.25 s per call, ~4 s per half; batching the matmuls too is a redesign (a cmma kernel taking a list of cells). The bench re-reads one cached tile, so it models ~0.57 s per call against the measured 1.1 s enqueue loop; the other half (real data in DRAM, ragged cells, more columns per cell) is not explained.

### Profile of a real search: allocation was the cost (2026-10-13)
`samply` (Samply task, debug-info build in a separate target dir, `part=0/4`, main thread, 1 ms samples, symbols via llvm-symbolizer; profile kept in `runs/prof/p.json.gz`): `search_ivf` 11.2 s inclusive, of which `scan_group` 7.2 s and `rows_f16` 5.8 s, almost all in `Client::empty` / `do_empty` (10.9 s inclusive over the run; ~6.3 s self time in the channel `submit_inner`, the main thread waiting for the device thread to hand back a fresh buffer). The bench above missed this because it allocated one size repeatedly (pool hit). Fix: `rows_f16` dequantises into one scratch buffer shared by all calls (`deq_scratch`, grown on demand; the queue is in order and every caller uses the result immediately). Half-0 eval from cache: scored 87.1 s -> **74.1 s**, `search_ivf` 16 calls 44.8 -> 37.0 s, tail calls 17 -> 14 s (some of that is run-to-run), CE 0.9793 unchanged, 13 knn tests pass. Other `client().empty` calls per block/group remain (dist, ws, qall); untested whether they matter now.

### Scratch slots, and a second profile (2026-10-13)
All per-block/per-group `client().empty` buffers that are pure scratch (dequantised rows, tile distances, best lists, thresholds, centroid distances, the 1 GB workspace) now come from `KnnStore::scratch(slot, bytes)` (one mechanism, replacing `deq_scratch` and the free `filled`; slots that are read back whole keep their exact size). Half-0 eval 74.1 s -> 71.0 s, within noise (`search_ivf` 37.0 -> 36.8 s, tail 14.1 -> 13.7 s), CE 0.9793.
Re-profile (`runs/prof/p2.json.gz`, `part=0/4`, main thread): `search_ivf` 9.3 s inclusive (was 11.2); `dots` 6.9 s of it, and of that 4.9 s is the main thread blocked inside `ChannelDeviceHandle::submit_inner` plus 1.6 s in a `sleep` back-off, with the kernel-launcher work itself ~0.3 s. So the cell loop is waiting for the device queue: it is GPU-bound (cmma at ~7 TFLOP/s, dequant), not host-bound, and further host-side tidying will not pay. Remaining levers are GPU time: a faster cmma kernel (the earlier survey bounded it at the ~40% the matmuls take), or fewer bytes moved (dequant writes 19 GB of f16 per call).

### Register-blocked cmma kernel: faster alone, no change end to end (2026-10-13)
Tried `k_dots_cmma` computing a 2x2, 4x2, 4x4 or 2x4 block of 16x16 tiles per plane (A fragment reused across columns, B across rows), bit-identical to the one-tile kernel on every shape tried (9008 or 16384 rows x 256 x qp 32..128, plus odd tile counts via clamping). Alone, back to back: one tile 4.7-9.4 TFLOP/s, 2x4 7.7-21.8 (best at qp >= 64; 2x2 best at qp 32-48); 1.3-2.3x. In the half-0 eval it changed nothing: 71.0 s -> 71.1 s (`search_ivf` 36.8 -> 37.5 s, tail 13.7 -> 13.2 s), CE 0.9793. Reverted (not worth 90 lines for no gain). So the matmuls are not what the full loop waits on: with the matmul 1.5-2x faster the call time did not move, which means the queue time is something else (dequant writes, the scan kernels, or the scheduling between thousands of small dependent launches); the profile only shows that the main thread waits for the queue, not which kernel holds it. Next: time the GPU side per kernel (timestamp queries, or an unsynced run with one kernel type removed while keeping the others valid).

### What holds the queue in the cell loop: launch count (2026-10-13)
> **Superseded in part:** every synced "matmul" time in this entry includes a 147 MB host readback per group (see "Timing-mode artefact" below) and the memory-bound / launch-bound conclusions drawn from it do not hold; wall-time results (eval seconds, CE) are unaffected.
Synced `KNN_TIMING` runs (`part=0/16`, temporary `KNN_SKIP` switch, results of the skipped variants ignored): the matmul stage of a 16384-query `search_ivf` call takes **1.78 s** with dequant + dots, **1.49 s** with dots only, **1.62 s** with dequant only. Neither kernel dominates and removing either saves little, so the stage is a per-launch floor, not arithmetic or bandwidth. Groups are **2990 queries** (the 1 GB workspace holds 256M floats at ~90k floats per query), not 8192 as assumed earlier, so a call is ~6 groups x ~4096 active cells = ~25k cell pieces = ~50k launches at ~30 us each = ~1.5 s, matching. (This also explains why a 1.5-2x faster cmma kernel changed nothing.)
Levers, all about launch count: (a) dequant consecutive cells in one launch (cell-ordered storage makes active cells adjacent in the tile; saves ~0.3 s per call, ~5 s per half); (b) one grouped-GEMM launch for many cells (a kernel whose cubes find their piece by binary search over per-piece descriptors: A row range, B offset, qp, output offset), with the dequant scratch holding a chunk of ~50 cells at a time: ~450 chunk launches per call instead of ~50k; bound on the saving ~1.2 s per call, ~19 s per half; (c) f16 workspace entries would double the group size but cost distance precision. Not built.

### Grouped matmul for the cell loop (2026-10-13)
> **Superseded in part:** every synced "matmul" time in this entry includes a 147 MB host readback per group (see "Timing-mode artefact" below) and the memory-bound / launch-bound conclusions drawn from it do not hold; wall-time results (eval seconds, CE) are unaffected.
`k_dots_group` (ivf.rs): one launch per chunk of cell pieces (storage order, row span <= `SCRATCH_ROWS` = 262144 rows, one `rows_f16` dequant of the span per chunk) instead of one dequant + one matmul launch per piece; cubes find their piece by binary search over per-piece descriptors. Chunks broke at tile boundaries, so the launches per call became ~2k per group (one per tile touched), not a few hundred. Output identical (CE 0.9793, 13 knn tests pass). Half-0 eval 71.0 s -> **68.5 s** (`search_ivf` 36.8 -> 33.7 s). Synced matmul stage per call 1.78 s -> 1.67 s.
That is far below the ~19 s per half the launch-count argument allowed, so the stage was **not launch-bound**: the earlier skip experiments (each kernel alone ~1.5 s) were consistent with a memory-bound loop too. Revised model: every group (6 per call, 2990 queries each) re-streams the whole 37M-key store, dequantising 19 GB of f16 and reading it back: ~230 GB of DRAM traffic per call (~0.75 s at ~300 GB/s), plus ~1 GB of workspace writes per group and the matmul arithmetic (~3.6 TFLOP, ~0.4 s). The dequant-to-f16 round trip is therefore the lever: a matmul that reads the 4-bit keys directly (dequantising a 16x16 tile into shared memory per cube) would cut the traffic ~8x. My earlier fused-kernel bound (<= ~4 s per half) came from a bench that re-read one cached tile and so hid the DRAM cost; it should be redone on the real shape. Alternative lever: fewer passes (bigger groups), limited by the 1 GB workspace.

### Fused dequant in the grouped matmul: no speed change, simpler (2026-10-13)
> **Superseded in part:** every synced "matmul" time in this entry includes a 147 MB host readback per group (see "Timing-mode artefact" below) and the memory-bound / launch-bound conclusions drawn from it do not hold; wall-time results (eval seconds, CE) are unaffected.
`k_dots_group` now takes the tile's packed keys and scales and each cube decodes its 16x16 key block into shared memory before the cmma (f16 stores read the keys directly); one launch per tile, no dequant scratch, `SCRATCH_ROWS`/`chunk_rows` and the multi-chunk test line deleted. Output identical (CE 0.9793, 13 knn tests pass). Half-0 eval **68.5 s -> 68.5 s**; `search_ivf` 2.05 s per call (16 calls, ~33 s) vs 33.7 s before; idle machine (CPU 5%, no GPU leases).
So the f16 round trip (~19 GB written + read back per group) was **not** the bottleneck either, and the "memory-bound on the dequant traffic" model above is wrong or at least not the dominant term. What remains per call: the gathered-query operand reads (each cube reads a 16 x k strip of B from global per k-step, B reused only across a piece's row tiles), the 1 GB workspace writes per group, the CSR top-k scan (`k_scan`), and the ~2k per-tile launches per group. Next measurement should be per-kernel synced time on the real shape (`KNN_TIMING=1`) with the matmul stage split from `k_scan`, before any further lever. Kept because it is the same speed with less code and 128 MB less scratch.

### Synced split of `search_ivf` after the fused matmul (2026-10-13)
> **Superseded in part:** every synced "matmul" time in this entry includes a 147 MB host readback per group (see "Timing-mode artefact" below) and the memory-bound / launch-bound conclusions drawn from it do not hold; wall-time results (eval seconds, CE) are unaffected.
`KNN_TIMING=1`, half 0 (eval 83.7 s because every stage is synced; unsynced it is 68.5 s). Summed over 16 calls: probes 5.1 s, **matmuls 23.0 s (49%)**, **scan launch + readback 18.1 s (39%)**, host merge 0.4 s; 46.9 s in all vs ~33 s unsynced, so overlap hides ~14 s. Per call ~1.4 s matmul, ~1.1 s scan. The scan (`k_scan` over the 1 GB workspace per group, ~6 groups) is as big a lever as the matmul; both stream ~1 GB per group of workspace (matmul writes it, scan reads it). Candidate: fuse the top-k selection into the matmul cube (keep each 16x16 tile's best in registers/shared and never write the workspace), which removes the workspace traffic and the group-size limit it forces.

### `k_scan` is the scan stage, and it is not bandwidth-bound (2026-10-13)
Per-group synced split from the same `KNN_TIMING=1` run (192 group lines): sample pass ~2 s, **`k_scan` kernel 15.2 s** (~80 ms per group), readback 0.9 s per half. Reading the 1 GB workspace once is ~3 ms at ~300 GB/s, so the kernel runs ~25x over the bandwidth floor. Likely cause: thread t = (part, query) walks one workspace column with stride `qp` floats, and neighbouring threads are different queries in different cells, so every 4-byte read pulls a 32-64 B line (8-16x amplification) plus divergent top-list inserts. Fusing top-k into the matmul cube (previous entry's candidate) does not fit cheaply: a cube holds 16 rows x 16 queries, so per-query top-256 lists would need cross-cube merging (atomics or a second pass), i.e. a scan anyway. Cheaper candidate: remap scan threads so a warp covers adjacent queries of one cell (adjacent workspace columns), making reads coalesced.

### Negative: scan threads per (query, probe) entry, sorted by cell and column (2026-10-13)
Hypothesis: `k_scan`'s 25x gap to the bandwidth floor is uncoalesced workspace reads. Tried: thread = one (query, probe) entry, entries ordered by cell then workspace column (neighbouring threads read neighbouring columns), 8 lists per query (parts = nprobe), `k_sample` the same. Output identical (CE 0.9793, 13 tests), but `search_ivf` went **2.05 s -> ~4.8 s per call** (half-0 eval 68.5 -> 110.7 s; machine idle, CPU 14%). Reasons I can see, not separately measured: a group has ~2990 x 8 / 4096 = ~6 queries per cell, so a warp spans several cells and coalescing gains little (a cell's workspace row is only qp = 16 floats), while a thread now owns a whole cell, so the load imbalance between cell sizes is large (before, every thread did 1/8 of every probed cell). Reverted. So the scan's cost is not mainly read coalescing; the remaining suspects are the insert path (divergent shifting of 256-entry lists in global memory) and the thresholds (how many keys pass `dist < worst`).

### Transposed workspace: `k_scan` threads stream contiguous rows (2026-10-13)
Two diagnostics first. With inserts disabled (thresholds forced to -1e30, results garbage, timing only) the scan kernel still took ~61 ms per group vs ~80 ms, so the insert path is only ~25% of it; the rest is reading the 1 GB workspace at ~16 GB/s. Cause: the workspace was [rows, qp] row-major, so one scan thread (one query column) stepped `qp` floats between consecutive keys and used 4 B of every 32 B sector it pulled, never getting a second hit. Fix: `k_dots_group` now computes the transposed tile (queries as the A operand and keys as the B operand, both `ColMajor` loads, the f32 tile stored at `out_off + query * rows16 + key`), so each query's distances to a piece are contiguous and `v_stride` is 1. Output identical (CE 0.9793, 13 knn tests pass). `search_ivf` 32.9 s -> **30.0 s** per half (16 calls: 2.05 -> 1.88 s), half-0 eval 68.5 -> **65.0 s**. Modest (the synced split was not repeated). A gain, not the 25x: the kernel remains far from the bandwidth floor, so the insert path and per-thread serial latency (24k threads, 8 chained loads) are the next suspects.

### After the transposed workspace: synced split, repeat run, tail search (2026-10-13)
> **Superseded in part:** every synced "matmul" time in this entry includes a 147 MB host readback per group (see "Timing-mode artefact" below) and the memory-bound / launch-bound conclusions drawn from it do not hold; wall-time results (eval seconds, CE) are unaffected.
Second uncontended half-0 run: **66.3 s** (first 65.0 s; `search_ivf` 30.06 s over 16 calls). Synced (`KNN_TIMING=1`, eval 80.8 s) per half: probes 5.0 s, matmuls 22.9 s, scan stage 16.1 s (sample pass 2.1 s, `k_scan` kernel 13.1 s, readback 0.8 s); `k_scan` was 15.2 s before the transpose. The grouped matmul is now the largest stage (about 1.4 s per call synced).
Tail search (14.0 s unsynced over 16 calls, ~0.9 s per call, 8 blocks of 2048 queries): `k_topk` already has neighbouring threads on neighbouring queries over `d[j * ld + qi]`, so it is coalesced and the transpose idea does not apply. Per block, synced, the matmul is 11-18 ms and the top-k 50-110 ms, so the in-document top-k (insert path; 50% of neighbours are in-document keys) dominates; the lever there is the insert/threshold logic, not memory layout.

### Timing-mode artefact: the "matmul" stage was a 147 MB readback (2026-10-13)
`KNN_TIMING=1` synced the grouped-matmul stage with `read(&ivf.vals)`, which copies the whole 37M-float value array to the host (147 MB over USB4) once per group, so every synced "matmuls" figure above (23 s per half; the "matmul is the largest stage / 1.4 s per call" conclusion; the skip-experiment that left 19.5 s with the kernel body disabled and 18.5 s with no launches at all) measured that copy, not the kernels. The sync is now a read of the small `cube_start_dev` buffer. Honest synced split, half 0 (eval 65.5 s, equal to unsynced): **probes 5.0 s, matmuls 11.1 s (0.69 s per call: gather, uploads, ~13k launches, kernels), scan stage 13.6 s**. Conclusion: the scan stage is the largest again, then matmuls, then probes. The 2240-launches-per-group count is still the unresolved part of the matmul stage (to test: a run with the launches skipped, now that the sync is cheap).

### Tail search: ranges and thread count (2026-10-13)
`KNN_TIMING` now prints the mean per-query range. In-document ranges are 3k-40k keys (windows 4k-40k keys wide), so each 2048-query block does ~6M-80M key evaluations in a 45-63 ms top-k (matmul 11-17 ms); thresholds from a sample of every 61st key within so small a window are loose, which is why the insert path dominates. Thread count is already at its knee: default `parts` = THREADS / 2048 = 16 (32768 threads); `KNN_PARTS=32` fails (dispatch of 65536 groups > 65535 limit), so more threads would need a 2-D dispatch. Not tried: threshold from the query's own recent in-document keys (they are the nearest candidates) instead of the global sample; that is the lever for the insert cost.

### The qall upload was the matmul stage's cost: filled on the device (2026-10-13)
Per-group breakdown of the (honest) 11.1 s matmul stage: host prep ~0, `qall` creation 5 ms to enqueue but ~22 ms to land (a 33 MB zero vector uploaded from the host over USB4, ~1.5 GB/s), gather kernel ~1 ms, grouped-matmul launches + kernels ~30 ms. `qall` is now zero-filled on the device (`KnnStore::filled`, new scratch slot `S_QALL`, f16 pairs as f32 zeros), and `k_gather_q` uses one thread per (pair, component) instead of a 256-step loop per pair (the second part changed nothing measurable). Matmul stage 11.1 -> **5.9 s** per half, `search_ivf` 30.0 -> **23.4 s**, half-0 eval 65-66 -> **58.8 s** (CE 0.9793, 13 tests pass). Lesson, third time: any host-built buffer that crosses the link per group costs ~1.5 GB/s; fill or build on the device. Remaining host-to-device traffic per group to check: `up(&pq/pdst/pstride)` (24k ints each) and the descriptor arrays, all small.

### Tail threshold rank 20 with a fallback chain; probes selected on the device (2026-10-13)
1. **Tail threshold.** The masked (in-document) search took its threshold from the 32nd-nearest sample key in the query's range. A query's 256 nearest hold ~256/61 = 4.2 sample keys on average, so rank 20 still leaves >= 256 keys under it for all but ~2e-8 of queries in theory, and passes ~1200 keys instead of ~2000. Rank 20: tail search 14.2 -> 10.3 s per half (half 0), half-1 CE 1.0729 and half-0 CE 0.9793 unchanged. Rank 14 lost the device (the miss fell straight to the unthresholded kernel, which runs for seconds on 40k-key ranges), and so did the first rank-20 half-1 run. Misses are now retried down a chain: rank 20, then `THRESH_RANK` = 32, then none (`search_from`); half 1 had 8 rank-20 misses (of ~250k queries), all recovered at rank 32. The measured miss rate is far above the Poisson estimate, so neighbouring keys are not independent samples; rank 20 is a speed choice, not a guarantee.
2. **Probe selection on the device.** `search_ivf` read back the full [4096, 2048] centroid distance matrix per 2048-query block (33 MB, ~268 MB per call over USB4 at ~1.5 GB/s) and selected the 8 nearest cells on 8 host threads. `k_probe` now picks the `nprobe` nearest cells per query on the device (insertion into a register list, `nprobe` <= `PROBE_MAX` = 16) and only the ids come back. Probes stage 5.0 -> **1.5 s** per half; half-0 eval 58.0 -> **51.6 s** (CE 0.9793, 13 tests pass). Ties between exactly equal centroid distances now go to the lower cell id (before: unspecified).
Synced split of half 1 before change 2 (eval 58.7 s): probes 5.0, matmuls 5.6, scan 13.7 (of which `k_scan` ~11-13), host merge 0.4; tail search 11.4 s (top-k 5.0, matmul+thresholds 5.3). Half 0 after both changes (synced 53.1 s): probes 1.5, matmuls 5.4, scan 13.4.

### Negative: tighter IVF threshold rank; refreshed half-1 and 32M timings (2026-10-13)
`scan_group` thresholds (SAMP = 32, rank 32 = ~the 1024th nearest row, 4x the 256 needed) tried at ranks 20, 12 and 8 with the same rank-32 / none retry chain as the tail. Half 0, `KNN_TIMING=1`: rank 32: eval 53.4 s, scan 13.3 s, no misses; rank 20: 52.8 s, scan 11.7 s but 43 of 192 groups missed (a miss redoes the whole ~3000-query group: matmul stage +1.1 s), net ~0; ranks 12 and 8: every group missed, 64-65 s. A group of 3000 queries hits some query below the 8-sample expectation too often for a tighter rank to pay. Reverted; rank stays 32 with no chain.
Refreshed unsynced-equivalent timings after today's changes (CE unchanged): 64M model half 0 **51.6 s** (0.9793), half 1 **52.4 s** (1.0729, was 73.4 s); 32M model half 0 **51.9 s** (0.9843, was 74.1 s).
Tail top-k insert cost (step 3 of the last list) not pursued: it shares the insert-shifting logic with `k_scan`, and the only cheap lever (threshold rank) was just measured to give ~nothing for the IVF scan; a shared-memory candidate batch is a kernel redesign, parked.

### Whole-run accounting and the memory-cache load (2026-10-13)
A half-0 run is ~70-78 s wall, of which the timed chunk loop is ~52 s (forward 12, tier prepare 32 = `search_ivf` 20.4 + tail 10.3, scoring 2.7, writes 0.3); the rest is startup (checkpoint, lexicon, whitener) and the memory load. The `.mem` cache (5.76 GB) loaded in 13.7 s: file reads 2.6 s (the file itself reads in 1.0 s from the page cache), tile uploads 7.1 s (~0.8 GB/s over USB4, 4 buffers x 2240 tiles), and 3.7 s reading the per-key norms and values back from the device for the IVF arrays although the loader already had those bytes. The loader now keeps the norm/value bytes it reads (3.7 -> 0.5 s) and a reader thread runs a few tiles ahead of the uploads (file reads overlap; small gain, uploads dominate): load **13.7 -> 10.1 s**, same results (13 tests incl. the save/reload identity test). Remaining: ~7 s of tile uploads at below link speed; the model forward pass (12 s) and the 5-11 s of startup before the memory load were not examined (stdout buffering made a line-timestamp poll unreliable; a `std::time::Instant` print per setup step is the next measurement).

### Setup timeline and host-device link rate (2026-10-13)

`tier_eval` now prints `setup: <step> at <t>s`. Half 0, 64M: corpus + checkpoint read 1.3 s, whitener fit 1.8 -> 5.7 s (3.9 s), memory cache ready 15.9 s (10.1 s), tiers built 18.7 s (2.8 s); then the 52 s loop (wall ~73 s).
A throwaway test (removed) measured the eGPU link: uploads of 1-128 MB buffers, with 1, 2 and 4 threads, all ran at 0.29-0.39 GB/s; a 256 MB readback ran at 0.30 GB/s. Rates do not depend on buffer size or thread count, so the ~7 s of tile uploads in the cache load are link-bound and merging them into fewer, larger uploads cannot help. Only shrinking the bytes (the 4-bit keys already do) or keeping data resident would.

Startup trims (2026-10-13): the lexicon build inserted every prefix of every word *token* into the prefix set; it now does so only for words not yet seen, and the word-count loop clones a word only when it is new. Tier build 2.8 -> 1.8 s, CE unchanged (0.9793). The whitener's 2.6 s is forward 0.8-1.1 s, host covariance ~1.2 s, Jacobi eigendecomposition 0.4 s; a threaded covariance gained only 0.1 s (reverted), so it is bound by streaming the 256k x 256 sample, not arithmetic.

### Result lists of 128 instead of 256 (2026-10-13)
`KMAX` 256 -> 128: the best stack uses k = 128 and the list-insert shifting in `k_scan`/`k_topk` costs in proportion to the list length. Half-0 eval **52.5 -> 50.5 s** (tier prepare 32.0 -> 28.4 s), CE identical (0.9793), 13 knn tests pass. Costs the option of k = 256 (+0.001 CE in the older stack, `knn:256` rows above). A shared-memory candidate batch (kernel redesign) is still parked; this was the cheap form of the same lever.

### Forward readbacks: whitener uploaded once, one read per forward (2026-10-13)
Instrumented `forward` (1968 scored chunks + frozen-key forwards, ~2000 calls per half): launch ~1.4 ms, logits read ~3.2 ms (includes waiting for the forward), hidden ~1.8 ms per call. `whiten_device` re-uploaded the 256 KB whitening matrix and bias on every call (~0.9 ms at the 0.35 GB/s link); they are now uploaded once. Logits, row losses and hidden now come back in one `read_many`. Forward stage 12.1 -> 8.9 s, CE identical (0.9793); the half-0 loop is ~46 s (was 50.5). What is left per call is the logits (0.5 MB) and hidden (0.5 MB) crossing the link, plus launch; keeping hidden on the device through the kNN search is the next lever (large plumbing: the search takes host slices).

Refreshed timings after the 2026-10-13 trims (loop only; KMAX 128, whitener uploaded once, single read): half 0 ~46 s (CE 0.9793), half 1 **48.2 s** (CE 1.0729, was 52.4 s), 32M half 0 **44.6 s** (CE 0.9843, was 51.9 s); CE identical everywhere.
Device-resident hidden states (parked): per half the host round trip is ~1.5 s of hidden readback plus the queries going up ~3 times (f16: once for the probe matmul input, once transposed per 2048-query block for centroids, once for the tail search), ~1 s in all, so at most ~3 s of ~46. Getting it needs the in-document ("pending") keys quantised on the device (they are quantised on the host from the host copy today), a device append for the online store, and handle-taking search entry points: a deep rewrite for ~6%. Not done.

### Top lists as max-heaps: inserts were 97% of `k_scan` (2026-10-13)
Control experiment (CE meaningless): with the insert body disabled (`dist < worst && cnt > 1000000`), `k_scan` summed over a half fell from **9.5 s to 0.7 s**, so scanning the workspace is nearly free and the cost was the sorted-list insertion (binary search plus shifting ~64 entries of two global arrays per insertion). That settles the parked "shared-memory candidate batch" idea: the lever was the insertion itself, not the access pattern.
Change: `k_scan` and `k_topk` keep each list as a max-heap (`heap_replace`: replace the root, sift down 7 levels; `worst` is the root), and `merged_lists` first heapsorts every list in place (`k_sort_lists`, one thread per list) before `k_merge_parts`. Results identical (13 knn tests incl. brute-force comparisons; CE 0.9793). Half 0: `k_scan` **9.5 -> 3.5 s** synced, tail top-k ~1.4 s (was ~5), tier prepare 27.5 -> 24 s, loop **46 -> 44.4 s** (machine shared with nothing else at the time; forward stage varied 8.8-10.1 s run to run).

### Where the half-0 loop stands after the heap change (2026-10-13)
Refreshed (CE identical in all): half 0 ~44 s, half 1 **43.4 s** (52.4 at the start of the day), 32M half 0 **41.4 s** (51.9).
Synced split (`KNN_TIMING=1`, 43.8 s): forward ~9; `search_ivf` ~14 (probes 1.5, matmuls 5.3, scan stage 6.4 = sample pass ~2 + `k_scan` 3.5 + readback ~0.9, host merge 0.3); tail search ~8.7 (matmul + thresholds 4.6, top-k 1.4, readback 2.7); scoring ~3. Also visible: the first search pays a **1.6 s upload of the key sample** (every 61st key, ~600 MB f32 over the link; ~2.2 s over all `knn pass` lines), and the cache file carries the same 600 MB. Candidates: derive the sample on the device from the stored 4-bit tiles (no upload, a smaller cache, faster load); pack the tail's returned (distance, value) pairs (the readback is 2.7 s); the IVF sample pass (2 s).

### The key sample is gathered on the device (2026-10-13)
The sample (every 61st key) was kept on the host and uploaded at the first search (1.6 s, ~600 MB f16 over the link) and stored in the cache file (600 MB f32). It is now gathered on the device from the uploaded tiles (`k_gather_rows`, typeless; quantised tiles are dequantised first), full chunks kept between searches, the pending keys' sample added from the host as one small chunk; `build_ivf` reads it back once for its k-means; `KnnStore::save` writes an empty sample section and `load` skips whatever an older file carries, so existing caches work unchanged (new ones are 600 MB smaller). Sample keys are now the stored (4-bit rounded) keys, which only moves the thresholds slightly (results stay exact: the rank fallback chain still guarantees >= KMAX candidates). Half 0 loop **44.4 -> 39.4 s**, CE identical (0.9793); 13 knn tests pass (the sample-count assertion now checks `sample_chunks`).

### Threshold samples as heaps; packed value readback (2026-10-13)
`k_sample` (IVF) and `k_thresh` (tail) kept their `TR`-entry lists sorted (a linear rank scan plus a shift per insertion); they are max-heaps now (`heap_replace_key`), which the host already tolerated (`rank_of_parts` selects a rank from all parts' values). `k_merge_parts` packs the returned values two u16 to a word, so each query reads back 3/4 of the bytes. Together: half 0 **39.4 -> 38.6 s**, half 1 **38.7 s**, 32M **37.9 s**, CE identical (0.9793 / 1.0729 / 0.9843); 13 knn tests pass. Day total on half 0: 68.5 s at the start of the session's timed-loop work -> 38.6 s (52.5 s at the start of today's trims).

### Synced stage split, forward launch cost, whitener on the device (2026-10-13)
Synced split of the 38.8 s loop (`KNN_TIMING=1`): forward 8.6; `search_ivf` ~13.4 (probes 1.5, matmuls 5.3, scan stage 6.3 = `k_scan` 3.5 + the rest [sample pass, readback incl. the list sort and merge]); tail ~6.6 (matmul + thresholds 2.4, top-k 1.4, readback incl. sort + merge 2.75); scoring 2.3.
Forward launch cost: a forward is 41 kernel launches built in ~0.6 ms (15 us each), so the host-side tape is not worth caching; the rest of the 1.4 ms "launch" is id upload and the whiten matmul, and the stage is dominated by the logits/hidden readbacks (~1 MB per chunk over the link).
`fit_whitener` now accumulates the hidden states' Gram matrix and sums on the device (two f32 matmuls per launch, d x d read back, summed in f64) instead of reading 260 MB of hidden states and forming the covariance on the host: whitener 3.2 -> 1.4 s, eigenvalue line unchanged to 4 digits, CE identical (0.9793). Startup before the loop is now 13.7 s (memory load 9.7 s of it).

Negative (2026-10-13): logits read back as f16 (rounded on the device, widened on the host; half the bytes) gave no measurable forward-stage change: interleaved quarter-half runs 5.4/5.6/7.2 s (f32) vs 5.5/7.2/7.7 s (f16) under ~30% background CPU load from other sessions, CE identical. So the logits readback is not bandwidth-bound; reverted. The cache's old 600 MB key-sample section costs nothing to load (`load` seeks over it), so re-saving the caches would only free disk.

### Memory at 2.3x the text: novels90 (2026-10-14)
`scripts/fetch_gutenberg_memory2.sh` adds 51 more Project Gutenberg novels (45 MB, `data/gutenberg_memory2/`, gitignored); the `novels90` corpus is `novels40` plus those, same held-out. Memory: 83.4M 4-bit keys (651,591 train windows), IVF 4096 cells x 8 probes (cell sizes min 20 / mean 20,362 / max 136,448). Options as `runs/t7.sh` / `t11.sh` (64M model, `store=1000000`, dyn 0.05/4, `batch=128`), only `memory_corpus` changed.
- **Cold build 372.5 s** (forward + readback 183.8, host copy + norms 49.3, k-means + assignment 62.6, reorder 41.2, rest 35.6); cache 11.68 GB, saved in 17 s; reload + setup 22.6 s.
- **CE: half 0 0.9751 (was 0.9793, -0.0042), half 1 1.0687 (was 1.0729, -0.0042).** Model alone unchanged (1.0165 / 1.1146). 2.27x the memory text for -0.0042: on the log-linear trend (-0.005 per 2.5x), a little under it.
- **Cost:** timed loop half 0 63.2 s (tier prepare 40.8, forward 10.7), half 1 70.2 s (tier prepare 42.6) against ~38 s at 36.7M keys; the IVF scan is roughly linear in the store (mean cell 20k rows vs 9k), so the loop costs +25-30 s for the -0.004. Single runs, not checked for contention (shared lease only).
- **VRAM is now the limit.** After the cache loads, the device holds 12.38 GB in use / 14.10 GB reserved (16 GB card), before the 1 GB search workspace and other scratch. **A cold build followed by the eval in the same process died with `out of device memory allocating 1073741824 bytes`** (the build's transient allocations stay in the pool); rerunning from the cache worked. Open: release the build's pooled memory (`client().memory_cleanup()`) before the eval, and drop the tile copies of norms/values that `ivf.norms/vals` duplicate (~0.67 GB). 83M keys is about the ceiling for 4-bit on this card; more text needs fewer bytes per key (the key is 256 floats at 4 bits = 128 B plus 8 B norm/value) or a host-resident tier.
- `tier_eval` now prints `device memory: X GB in use, Y GB reserved` before the chunk loop.

### novels90 follow-ups: build-then-eval fit, 2-bit keys (negative), cache pruning (2026-10-14)
- **Cold build + eval in one process now fits.** `tier_eval` calls `client().memory_cleanup()` right after the memory is ready (the build's transient buffers otherwise stay pooled). A cold novels90 run (build 371 s) then went through: device memory 12.39 GB in use / 13.97 GB reserved, CE **0.9751** (same as the cache-loaded run), loop 66.1 s. Evidence is one failing run without the call and one passing run with it; the pool state at the failure was not measured.
- **Negative: 2-bit keys.** Prototyped by generalising the host packer (`tile_of` packs any `bits` dividing 8 at bit offset `i * bits`, replacing the 8/4 branches; kept) and allowing `keybits=2` (levels -1, 0, 1 per key; the device kernels were already generic in `bits`; a `bits.max(4)` in the `k_dots_group` launch had to become `.max(2)` for it). novels90, half 0: device memory 7.06 GB (vs 12.39), but memory build 341 s and **CE 1.0258, worse than the model alone (1.0165)**, and the loop 90.6 s (tier prepare 67 s; the IVF cells became very uneven, max 348,700 rows vs 136,448). Ternary keys lose the neighbours; 2 bits uniform is out. The 2-bit permission is reverted (4-bit stays the floor); a non-uniform or product quantiser would be the next try, not more uniform bits.
- Pruned `runs/memcache`: the novels40 caches (64M and 32M models) and the 2-bit cache are deleted; only the novels90 4-bit cache (10.9 GB) remains. The novels40 A/B timing scripts (`t7`, `t11`-`t13`) point at a deleted exe and cache; novels90 scripts are `t40.sh` (half 0) and `t41.sh` (half 1).

### Rotated, rms-scaled keys: 2-bit memory works; 8192 cells (2026-10-14)
Ternary max-abs 2-bit keys failed (above). Following RaBitQ (Gao & Long 2024, arXiv 2405.12497) two changes, both general rather than per-bit-width:
1. **Random rotation folded into the whitener** (`tier_eval rotate=<seed>`: W' = R W with R a Gram-Schmidt orthogonal matrix; keys and queries are rotated alike, distances unchanged, no query-time cost; test `random_rotation_is_orthogonal`). The cache key is now `v2` and includes `rotate`.
2. **Quantiser**: a row's scale is `quant_step(bits) x rms(row)` (the Gaussian-optimal uniform step: 0.9957 for 2 bits, 0.3352 for 4, 0.031 for 8) and the codes are mid-rise, `u in 0..2^bits` decoding to `(u - (2^bits-1)/2) x scale`, clamped. This replaces the max-abs scale and the signed-integer decode (one decode for every width, the sign branch deleted from `k_dequant` and `k_dots_group`; the host packer handles any width; the 13 knn tests pass for 8, 4 and 2 bits).

novels90, 64M model, `runs/t44`-`t47` (half 0 / half 1; loop time half 0, single runs, contention not checked):

| keys | cells | device memory | CE half 0 | CE half 1 | loop |
|---|---|---|---|---|---|
| 4-bit max-abs (before) | 4096 | 12.39 GB | 0.9751 | 1.0687 | 66 s |
| 4-bit rotated, rms | 4096 | 12.39 GB | **0.9750** | - | 66 s |
| 2-bit rotated, rms | 4096 | **7.06 GB** | 0.9772 | - | 65 s |
| 2-bit rotated, rms | 8192 | 7.06 GB | 0.9772 | 1.0711 | **48 s** (half 1: 51 s) |
| novels40, 4-bit (36.7M keys) | 4096 | - | 0.9793 | 1.0729 | 38 s |

- 4-bit is unchanged by the new quantiser (0.9750 vs 0.9751). **2-bit costs +0.0022 / +0.0024 CE against 4-bit and still beats the 36.7M-key memory** (0.9793 / 1.0729) while using 57% of the device memory it needed.
- **8192 cells x 8 probes** (mean cell 10k rows, as at 36.7M keys) cuts tier prepare from 43 s to 28 s with identical CE: the loop is 48 s instead of 65 s. With 4-bit keys the same setting ran out of device memory in the search scratch (12.4 GB in use before it), so the 2-bit memory is what makes the cheaper schedule fit.
- The 2-bit memory frees ~9 GB: room for roughly 2x more keys (about 170M, ~200 MB of text) before the card is full again.

### novels150: 125M 2-bit keys (2026-10-14)
`scripts/fetch_gutenberg_memory3.sh` adds 58 more Project Gutenberg novels (41 MB, `data/gutenberg_memory3/`, gitignored); `novels150` = `novels90` + those (149 books; same held-out). 2-bit rotated keys, IVF 8192 x 8, 64M model, `runs/t49.sh` (half 1, cold build) / `t50.sh` (half 0, from cache):
- **125,357,056 keys** (979,352 train windows). Cold build 563 s (forward + readback 263, host copy 69, k-means + assignment 143, reorder 44, rest ~44); minimum free host RAM during the build 26.8 GB of 64 GB (so ~37 GB peak; the 14 GB per 37M keys estimate was pessimistic). Device memory **10.58 GB** in use / 10.77 reserved.
- **CE half 0 0.9754 (was 0.9772 at 83M keys), half 1 1.0691 (was 1.0711)**: -0.0018 / -0.0020 for 1.5x the keys, i.e. the 2-bit memory now equals the 4-bit 83M one (0.9751 / 1.0687) in 6 GB less device memory. Rotated 4-bit on half 1 (novels90): 1.0688 (the max-abs figure was 1.0687), so the new quantiser is neutral at 4 bits.
- Loop: half 0 72.6 s, half 1 70.8 s (tier prepare 40 s, mean cell 15k rows) against 48 s at 83M keys: the search cost is linear in the store at a fixed cell size. Single runs, not checked for contention. `ivf=12288:8` would restore ~10k-row cells (untested; k-means grows too).
- The text-vs-CE trend continues log-linear (~-0.002 per 1.5x), now at the point where each further step costs more search time than it earns CE.

**12288 cells (2026-10-14, `runs/t51.sh`, novels150 half 0).** IVF 12288 x 8 (mean cell 10.2k rows, as at 36.7M keys): loop **58.7 s** (tier prepare 32.8 s) against 72.6 s at 8192 cells, CE identical (0.9754). The cold build is 641 s (k-means + assignment 199 s of it, 143 s at 8192). Recall was not measured. Cell count is the knob that keeps the search cost flat as the store grows; 16384 cells would be the next step (k-means ~260 s). The memory cache in `runs/memcache` is now the 12288-cell one; `t50.sh`/`t49.sh` (8192) would rebuild.

### Recency prior for rare in-lexicon words: negative (2026-10-14)

Prototype tier `recent:<lambda>:<max train count>:<span>` (not kept): at a word start, the first letters of the train words with
1..max occurrences that the book used in its last `span` bytes (same book only) get a share `lambda` of the letter mass, appended
to the full stack (novels150 2-bit memory, 12288 cells, half 0, one pass, 7 specs). Stack alone 0.9754 (model 1.0165).

| lambda | max count | span (B) | CE |
|---|---|---|---|
| 0.05 | 200 | 5000 | 0.9770 |
| 0.10 | 200 | 5000 | 0.9798 |
| 0.05 | 20 | 5000 | 0.9777 |
| 0.10 | 20 | 5000 | 0.9813 |
| 0.05 | 200 | 20000 | 0.9770 |
| 0.10 | 200 | 50000 | 0.9796 |

Every setting is worse, monotonically in lambda (extrapolates to zero gain as lambda -> 0), and the span barely matters. The
estimate of 0.005 assumed the cache saves a third of a nat on the 1.5% repeating rare-word starts; a flat mix over all recently
used rare words dilutes: most word starts (60300 of 251904 had a candidate) do not repeat a rare word, and the candidates
carry no information about which one comes next. The online kNN memory (50% in-document neighbours) already captures what
repeats are cheap to capture. Not tried: gating by the model's entropy or weighting candidates by distance to their last use;
expected small given the monotone trend. The "only measured KR&R lead" above is closed.

### 16384 cells: no gain over 12288 (2026-10-14)

`ivf=16384:8` on the novels150 2-bit memory (half 0, single run, GPU and CPU idle at launch, contention not rechecked): loop 66.7 s
against 58.7 s at 12288 cells, CE 0.9755 against 0.9754, build 750 s against 641 s, 10.59 GB in use. The search cost has stopped
falling (the larger centroid pass and smaller cells cost more than they save), so 12288 stays. Cell count is closed as a lever
at this store size.

### Auxiliary word-start heads: negative at 4M windows (2026-10-14)

Spike of ranked idea 3 above (code reverted; this is the record). Three extra linear heads on the final LayerNorm (bytes +2 and +3
ahead, and the first letter of the next word), each with a cross-entropy of weight 0.2 added to the loss and dropped at inference.
Paired A/B on novels22, seq 128, the 64M recipe otherwise (dropout 0.1, wd 2e-4, momentum 0.9), 4096000 windows, K=2 fused models
(seeds 1, 2) per condition, main-head held-out CE (score_tail 32):

| windows seen | baseline | with aux heads |
|---|---|---|
| 1.02M | 1.2262 | 1.2240 |
| 4.10M (end) | 1.1502 | 1.1623 |

The aux models led by 0.002 at 1M windows (inside the noise) and ended 0.012 worse; the two seeds of one condition agree within
0.005. They also cost 17% more per step (10.3 vs 9.0 ms) and their train-probe CE was higher (1.077 vs 1.064, mean of the two seeds, at the end), so
at 2M parameters the extra targets compete for the trunk's capacity rather than sharing it. Not tried: a smaller weight (0.05),
the word-start head alone, or a later start (the heads switched on after most of training); a monotone loss in capacity makes
those unlikely to reach the +0.005 bar. Idea 3 is closed at this size.

### Where the stack gains, by position class (2026-10-14)

`classes=1` on the current stack (novels150 2-bit memory, half 0; gain columns in 1e-3 nats of overall CE, total 41.1):

| class | share of positions | model CE | gain |
|---|---|---|---|
| word start, lower (after non-letter -> lower) | 16.2% | 2.444 | 7.58 |
| word start, upper | 2.0% | 2.922 | 1.32 |
| inside a word (after letter -> lower) | 57.8% | 0.734 | 27.77 |
| space after a letter | 14.1% | 0.299 | 2.38 |
| punctuation after a letter | 3.1% | 2.148 | 1.88 |

The tiers (lexicon, words, memory) remove 6.5% of the CE inside words but only 1.9% at word starts, which hold 45% of the model's CE.
Dynamic evaluation (in the stack since 2026-10-11) and tier gating (killed) are therefore settled; what is left for training is
whether capacity moved towards word starts helps the *stack*.

### Word-start loss weighting: negative (2026-10-14)

Spike of "train with the tiers in mind" (idea 5), in its simplest form; code reverted (per-row loss weights in the tape's cross-entropy,
`ws_weight=<a>` in `fused_models_check`: weight a where the input byte is not a letter and the target is, rescaled to mean 1).
Models of 4.1M windows (novels22, seq 128, 64M recipe otherwise; seed 1 of K=2), then the tier stack (lexicon 0.3, words 0.15, knn
128, dynamic evaluation 4 windows SGD 0.05, memory from the model's own keys on novels22 train, 4-bit, 2048 cells, half 0):

| word-start weight | model alone | stack | tiers' gain | word-start CE (model) |
|---|---|---|---|---|
| 1 | 1.0552 | **1.0110** | -0.0442 | 2.488 |
| 2 | 1.0575 | 1.0121 | -0.0453 | 2.461 |
| 4 | 1.0687 | 1.0160 | -0.0528 | |

Weighting does move capacity (word-start CE falls 0.027 at weight 2) and the tiers recover part of the loss elsewhere (their gain
rises with the weight), but the stack gets monotonically worse. Held-out CE is not improved by redistributing a 2M model's capacity
towards the positions the tiers help least. One seed per cell on half 0; seed spread of the model alone is ~0.005 at the end of
training, so weight 2 is a tie and weight 4 a loss. Ideas 3 and 5 (as tried) are closed; what remains of the list is 2 (in-model
memory, large) and 4 (hashed previous-word inputs, adds parameters).

### In-model kNN attention: design pass, recommendation not to build (2026-10-14)

What it would be here: one attention layer (say block 2) also attends over the top-k (key, value) states retrieved from the earlier text
of the same book (Memorizing Transformers, Wu et al. 2022), with a learned per-head gate; trained end to end so the model learns how
to use retrieved states instead of the fixed output-level mixture.

Build list: (1) the trainer samples random windows (`sample_start`), so there is no in-book history; it would need document-ordered
batches and a per-document cache of block-2 K/V, written as training proceeds (stop-gradient); (2) a top-k attention kernel over
that cache in `gpu_step` (scores against up to ~64k stored states, softmax over k plus the local window, backward through the retrieved
values only); (3) a few parameters (gate, maybe a key projection), well inside 2M; (4) `tier_eval` must carry the cache through the
chunked, adapted forward, and the dynamic-evaluation step must differentiate through it. A multi-day change across `tape.rs`, the
trainer and `tier_eval`.

Evidence on the headroom, all from this log: the in-book memory is already half of the kNN neighbours and the tuned stack prefers it
(in-document temperature 30 vs 50); dynamic evaluation and the memory "learn the same book" (a 0.0084 model gain reached the stack
as 0.0030, RMS's extra 0.0032 as 0.0001); a longer window is the nearest proxy for a longer in-model reach and doubled the window for
-0.0086 on the model and -0.0021 on the stack; and the tiers barely touch word starts (1.9%), the only large CE mass left, which a
retrieval over this book's earlier states does not contain unless the word recurs (the recency prior tested exactly that: negative).
The expected gain on the stack is at most a few thousandths, with the same overlap that cost the longer window most of its gain.

Recommendation: do not build. The cheapest falsifier, if wanted anyway, is the Transformer-XL variant (one cached previous window, no
retrieval): it shares the trainer change (1) and the attention change (2) in simplified form and measures the same overlap; if its
stack gain at 4M windows is under 0.003, the retrieval version is closed too. Revisit only if a cheaper variant shows more than that.

### First-letter probe tier (residual mining): marginal, reverted (2026-10-14)

Word-initial letters hold ~45% of the CE and the stack removes only 1.9% of it there, so a count tier `fl:<feature>:<lambda>:<c>` was tried: at a word start, mix in the TRAIN-text first-letter distribution for a context feature (weight `lambda*n/(n+c)`), after the stack. Single runs on the cached 12288-cell memory; stack alone 0.9754, model alone 1.0165.

| feature (previous...) | lambda:c | CE |
|---|---|---|
| separator + capitalised? (`sep`) | 0.1:3 | 0.9772 |
| last 3 letters (`suf3`) | 0.1:3 / 0.3:3 | 0.9765 / 0.9817 |
| word (`w1`) | 0.1:3 | 0.9757 |
| two words (`w2`) | 0.1:3 / 0.1:10 / 0.2:3 / 0.05:3 | 0.9747 / 0.9748 / 0.9750 / 0.9749 |
| three words (`w3`) | 0.1:3 | 0.9748 |

Coarse classes (`sep`, `suf3`) and one word hurt, and the damage grows with lambda: the model's hidden state already carries that information. Only the sparse, specific contexts (`w2`, `w3`) help, by at most 0.0007 (applied at 34k of ~250k positions for `w2`). That is below the 0.002 bar and close to run noise, so the tier is not kept; the code is reverted. Closes first-letter count tiers: the remaining word-start residual is not recoverable from train-text context counts.

**Transformer-XL falsifier: already answered, not run (2026-10-14).** The "Context length under the tiers" table above is this test with a stronger variant: a 256-byte window with gradients through all of it reaches further back than one cached stop-gradient previous window at seq 128. It gained -0.0086 on the model and only -0.0021 on the stack (novels6, static stack), under the 0.003 bar, so in-model retrieval through longer reach is closed on that evidence. Caveat: measured on the older static stack, not the current dynamic-evaluation one; a re-measure would be a 64M-window retrain at seq 256 (~2x step cost) and is not planned.

### Hashed previous-word embeddings: closed without a run (2026-10-14)

Idea 4 of the list (hashed previous-word / prefix embeddings as extra inputs) is a trained lookup of "what follows this word". The first-letter probe measured the same lookup with exact train-text counts (`fl:w1`, `fl:w2`, `fl:w3`): at most -0.0007 on the stack, and `w1` hurt, because the model already sees the previous words in its 128-byte window. A hash table holds no more than those counts. The word bigram tier only acts inside words ("between words at 0 positions"), so word starts were the case that mattered, and the counts show nothing there. Budget also bites: 8192 buckets x 256 = 2.1M parameters doubles the model against the ~2M constraint; a 1024-bucket table fits but holds little. Not run. Revisit only if a model-alone probe shows the hidden state lacks previous-word information at word starts.

### Retrieval-driven test-time training (`dyn_nn`): first result, a lead (2026-10-14)

Hardt & Sun (README, Related work) fine-tune on the nearest training sequences of each test sequence. Here: `dyn_nn=<m>` adds, after each chunk's own dynamic-evaluation step (SGD 0.05, 4 windows), one step of the same optimiser on m train windows of the memory corpus (novels150; every 8th window embedded once by the frozen model, 122k windows, ~40-50 s). A window's embedding is the mean frozen hidden state over its scored rows (positions >= warm); each of the chunk's 4 windows contributes its m/4 nearest (brute force on the host, L2). `dyn_nn_mode=random` takes m random train windows instead: the control for "extra train-text steps help whatever they are". Half 0, current stack (`runs/t57.sh`; single runs; CPU about 70% busy from another job at launch, which affects wall time only):

| extra steps | model alone | stack |
|---|---|---|
| none (dyn only) | 1.0165 | 0.9754 |
| m=16 random train windows | 1.0119 | 0.9751 |
| m=16 nearest windows | **1.0102** | **0.9731** |

Random train windows alone give -0.0046 on the model but only -0.0003 on the stack (a general train-text effect that the memory already supplies). Retrieval adds -0.0017 on the model and **-0.0020 on the stack beyond the control**, -0.0023 against dynamic evaluation alone: just over the 0.002 bar and the 0.001 margin over the control, single run. Not yet done: m, neighbour-step learning rate, `only` mode, half 1, and whether the gain survives re-tuning the tier weights for the adapted model.

Second grid (`runs/t58.sh`, half 0, single runs, same stack; baseline 1.0165 / 0.9754):

| setting | model alone | stack |
|---|---|---|
| add, m=16, lr 0.05 (first grid) | 1.0102 | 0.9731 |
| add, m=64 | 1.0099 | 0.9732 |
| add, m=16, neighbour lr 0.02 | 1.0121 | 0.9739 |
| add, m=16, neighbour lr 0.1 | 1.0101 | 0.9731 |
| random, m=16 (first grid) | 1.0119 | 0.9751 |
| random, m=64 | 1.0112 | 0.9747 |
| only, m=16 (no own-chunk step) | 1.0406 | 0.9880 |

The gain saturates at m=16 (64 neighbours add nothing; the margin over the random control shrinks from 0.0020 to 0.0015) and is flat in the neighbour learning rate between 0.05 and 0.1 (0.02 is worse). Neighbours cannot replace the chunk's own step: `only` is 0.015 worse than dynamic evaluation alone, so the text just scored is the stronger signal and retrieval is a supplement. Next: half 1, a repeat for run noise, and re-tuned tier weights.

### Retuning the tiers for the 125M-key memory; `dyn_nn` confirmed (2026-10-14)

Controls on `dyn_nn` showed the stack itself was mistuned: the kNN weight 0.3 was set for a far smaller memory. Half 0 (`runs/t60.sh`, `t61.sh`, `t62.sh`; single runs, deterministic: a repeat reproduced 0.9731 exactly; `knn:128:<lambda>:30:20:100`, lexicon 0.3, words 0.15 unless noted):

| kNN lambda | dyn only | + `dyn_nn=16` |
|---|---|---|
| 0.3 | 0.9754 | 0.9731 |
| 0.4 | 0.9721 | 0.9699 |
| 0.45 | 0.9712 | 0.9691 |
| 0.5 | **0.9708** | **0.9687** |
| 0.55 | 0.9709 | 0.9689 |
| 0.6 | 0.9715 | 0.9695 |
| 0.7 / 0.8 | 0.9745 / 0.9802 | 0.9724 / 0.9780 |

Retuning the weight alone gains 0.0046; neighbour adaptation adds a steady 0.0021 at every weight. Other settings at lambda 0.5: lexicon 0.2 + words 0.1 -0.0005 (0.9703 / 0.9682); in-document temperature 15 -0.0006 (0.9702 / 0.9684; not combined with the others); train temperature 25 or 40, in-document temperature 30, shift 80 or 120 or (at 0.6) 50 or 150 all worse or equal. Combined best (lexicon 0.2, words 0.1, lambda 0.5, temperatures 30:20:100):

| | half 0 | half 1 |
|---|---|---|
| old stack (lambda 0.3) | 0.9754 | 1.0691 |
| retuned, dyn only | 0.9703 | 1.0654 |
| retuned + `dyn_nn=16` | **0.9682** | **1.0630** |

Half 1 (not used for tuning) confirms: lambda 0.5 is its optimum too (0.3: 1.0691, 0.4: 1.0663, 0.5: 1.0658, 0.6: 1.0676); `dyn_nn` gains -0.0024 there (1.0654 -> 1.0630), against -0.0010 for 16 random windows. Other first-grid facts: the last row as the window embedding (instead of the mean over scored rows) is worse (0.9741 vs 0.9731; code removed); 64 neighbours = 16; neighbours cannot replace the chunk's own step. New best: **0.9682 / 1.0630** (was 0.9754 / 1.0691).

Final tier tuning and dynamic-evaluation retune with `dyn_nn=16` on (`runs/t63.sh`, lexicon 0.2, words 0.1, kNN 128:0.5, temperatures `30:<doc>:100`; single deterministic runs):

| in-document temperature | half 0 | half 1 |
|---|---|---|
| 20 | 0.9682 | 1.0630 |
| **15** | **0.9678** | **1.0626** |
| 10 | 0.9682 | 1.0636 |

New best **0.9678 / 1.0626** (doc temperature 15, confirmed on both halves). Dynamic evaluation retuned on half 0 at doc temperature 15 (neighbour step fixed at lr 0.05, 16 windows per chunk):

| dyn lr, chunk (windows) | model alone | stack |
|---|---|---|
| 0.05, 4 (current) | 1.0102 | 0.9678 |
| 0.03, 4 | 1.0157 | 0.9702 |
| 0.08, 4 | 1.0092 | 0.9673 |
| 0.05, 2 | 1.0063 | **0.9665** |
| 0.05, 8 | 1.0193 | 0.9719 |

Both directions point the same way: more and larger steps help (smaller chunks -0.0013, lr 0.08 -0.0005), so the optimum is not yet bracketed; the model alone gains more than the stack (-0.0039 against -0.0013 at 2 windows). Not yet confirmed on half 1; smaller chunks also cost time (twice the steps, and twice the neighbour steps at 16 windows per chunk).

Bracketing dynamic evaluation with `dyn_nn=16` (`runs/t64.sh`, `t65.sh`; lexicon 0.2, words 0.1, kNN 128:0.5:30:15:100; stack CE, single deterministic runs; the same pass also scored kNN 0.4/0.6 and doc temperature 10/20 and 0.5/15 was best in every run: the tier weights do not move):

| chunk (windows) | lr 0.04 | 0.05 | 0.06 | 0.08 | 0.12 |
|---|---|---|---|---|---|
| 4 | | 0.9678 (half 1 1.0626) | | 0.9673 | |
| 2 | **0.9659** | 0.9665 (half 1 1.0631) | 0.9677 | 0.9722 | 0.9867 |
| 1 | | 0.9740 | | 1.0031 | |

The optimum is a narrow ridge in step size x steps per byte, not an open slope: past chunk 2 / lr 0.05 the CE rises and large steps blow up (chunk 1 at lr 0.08, chunk 2 at lr 0.12). Half 0 favours chunk 2 (-0.0013 at lr 0.05, -0.0019 at lr 0.04), but half 1 does not confirm it (1.0631 against 1.0626 at chunk 4), so the README keeps chunk 4, lr 0.05 (0.9678 / 1.0626). Chunk 2 also costs about 25% more loop time (~0.088 s per chunk at twice the chunks, from the per-chunk speed lines, not a wall-clock total). Unchecked: chunk 2 at lr 0.04 on half 1.

Half 1 check of chunk 2 / lr 0.04 (`runs/t66.sh`): 1.0620 (against 1.0626 at chunk 4 / lr 0.05, 1.0631 at chunk 2 / lr 0.05). It improves on both halves (0.9659 / 1.0620), so it is adopted as the best stack, at about 25% more loop time; the half-1 gain (-0.0006) is small, and lr 0.04 was chosen on half 0 only.

**Third held-out split (2026-10-15, `runs/t67.sh`, logs in `runs/fresh/`).** The tuning so far used only the two halves of one held-out set, so the final stack was re-scored on a set it never saw: the last 10% of seven memory-only books (far_from_the_madding, evelina, madame_bovary, allan_quatermain, house_of_mirth, jungle, lost_world; 0.49 MB, same size as the old set) that the model never trained on. Corpora `fresh22` / `fresh150` (`FRESH` in `examples/common/mod.rs`): the memory (125.6M keys, cold rebuild) and lexicon/words hold the first 90% of those books and, unlike before, the novels6 tails. Same checkpoint, nothing retuned. Half 0 / half 1 (model alone 1.0796 / 1.0906):

| setting | half 0 | half 1 |
|---|---|---|
| current best: chunk 2, lr 0.04, `dyn_nn=16` | **1.0045** | **1.0140** |
| chunk 4, lr 0.05, `dyn_nn=16` | 1.0078 | 1.0171 |
| chunk 2, lr 0.04, no `dyn_nn` | 1.0138 | 1.0203 |

The ordering and size of both gains carry over: the chunk-2 retune gives -0.0033 / -0.0031 (it was -0.0019 / -0.0006 on the tuning set), `dyn_nn` gives -0.0093 / -0.0063. Model alone is worse here (1.0796 vs 1.0165), as the model never saw the first 90% of these books, yet the tiers recover more (-0.075 vs -0.05), so the stack gain is not an artefact of the tuning halves. Contention: CPU was ~30% busy from another session at launch (timings not used).

**Does the `dyn_nn` gain grow with the memory? No (2026-10-15, `runs/t68.sh`, logs `runs/fresh/small_*`).** Same fresh split, same settings (kNN weight 0.5 untuned for the small memory), memory built from `fresh22` alone (20.1M keys, 22 books) against `fresh150` (125.6M keys). Half 0 / half 1:

| memory | with `dyn_nn=16` | without | `dyn_nn` gain |
|---|---|---|---|
| 20.1M keys | 1.0224 / 1.0352 | 1.0335 / 1.0427 | -0.0111 / -0.0075 |
| 125.6M keys | 1.0045 / 1.0140 | 1.0138 / 1.0203 | -0.0093 / -0.0063 |

Six times the keys is worth -0.0179 / -0.0212 on the full stack, while the `dyn_nn` gain is slightly smaller with the larger pool (its neighbours come from the same train windows either way, so a bigger kNN memory takes over part of what it supplied). So `dyn_nn` is not a reason to add memory text; the memory itself is.

**Cost of the dynamic-evaluation settings (same logs, half 0, 244,736 positions, another session using ~30% CPU).** Chunk loop: 97 s without dynamic neighbour steps (chunk 2), 234 s at chunk 4 / lr 0.05 with `dyn_nn`, 353 s at chunk 2 / lr 0.04 with `dyn_nn`; the neighbour index adds 42 s of setup. Per 0.001 of CE (half 0 / half 1 average): `dyn_nn` costs ~28 s of loop (-0.0093 / -0.0063 for 256 s), and chunk 2 over chunk 4 costs ~36 s (+119 s, +51%, for -0.0033 / -0.0031). The "about 25% more" quoted earlier was measured on a quiet machine; this ratio is contended, so treat it as an upper bound.

**kNN retune at the small memory (2026-10-15, `runs/t69.sh`, `runs/fresh/grid_small*`).** Fresh split, 20.1M-key memory (`memory_corpus=fresh22`), best dynamic evaluation + `dyn_nn=16`, spec `knn:128:<λ>:<T>:<Td>:100` after lexicon 0.2 + words 0.1. Half 0 / half 1: λ 0.3 1.0218 / 1.0350, **λ 0.4 1.0209 / 1.0339**, λ 0.5 1.0224 / 1.0352, λ 0.7 1.0333 / 1.0453. At λ 0.4: doc temperature 8 / 15 / 25 = 1.0225 / 1.0209 / 1.0224 (half 0), temperature 20 / 30 / 45 = 1.0301 / 1.0209 / 1.0237 (half 0; 1.0400 / 1.0339 / 1.0393 on half 1). So the optimum weight moves with the memory (0.4 at 20M keys, 0.5 at 125M; 0.3 was best at the earlier, smaller memory) but only by -0.0015 / -0.0013 at the small end, and the temperatures stay at 30 / 15. A default of 0.5 loses nothing material at any size tried; no setting changes.

**Adapting a parameter subset: already answered, closed (2026-10-15).** Proposed again as a cheaper route to more steps, but the `dyn_layers` sweep in "Round 3 and which parameters to adapt" covers it: restricting to b2+b3+head, b3+head, head or embedding only is worse than adapting all blocks (1.0643 / 1.0714 / 1.0862 / 1.0942 against 1.0590 for blocks only), so fewer parameters narrows the gain instead of widening the ridge. No run.

### Meta-trained adaptability: design, not started (2026-10-15)

**Idea.** The checkpoint was trained without adaptation (64M windows, 21,823 s), yet at test time every chunk gets an SGD step (`dyn`, lr 0.04, 2 windows) and the optimum is a narrow ridge in step size x steps per byte. MAML (Finn et al., arXiv:1703.03400) trains parameters so that a few gradient steps on a little data give good performance; the same move here is to train the model so that the per-chunk step it will receive at test time helps more. Krause et al. (arXiv:1904.08378) also evaluate models trained without adaptation. For the TTT-layers paper (arXiv:2407.04620) only the abstract page was read: training the network through the inner updates is from memory, unverified. Related work paragraph: README, "Training for test-time adaptation".

**Mechanism (first-order, FOMAML-style; no second derivatives).** Per outer step, take a run of K consecutive chunks from one book. Copy the weights to fast weights theta_0; for t = 1..K: score chunk t with theta_{t-1} (loss L_t, dropout off as at test time), accumulate the gradient g_t = grad L_t(theta_{t-1}), take the inner SGD step theta_t = theta_{t-1} - eta g_t with eta = the test-time lr. The outer update applies sum_t g_t to the slow weights (Adam, the usual recipe). With K = 1 this is ordinary training, so the control is the same code with K = 1. The existing pieces cover it: `DeviceTape` forward/backward, `dev.sgd`, `dev.zero_grads`; what is new is a device-side copy of the parameter and gradient buffers (today only `read` and `upload` exist, a host round trip) and a gradient accumulator. The exact second-order term (backprop through the inner step, a Hessian-vector product) is out of reach for this tape and probably not needed (first-order MAML is the standard cheap form).

**Test.** Fine-tune the existing checkpoint rather than retrain (a full run is 6 h): about 1-2% of the original steps (1M windows) with K = 4, against a control that fine-tunes the same number of steps with K = 1. Then score model alone, half 0 of the original held-out set, with `dyn=0.04 dyn_every=2`, and compare the dynamic-evaluation gain (CE without minus CE with `dyn`) for the two. The control matters: extra training alone moves CE. Then, only if the model-alone gain is more than about 0.01, the full stack on both halves and the fresh split. Overlap rule: only 15-30% of a model-alone gain reaches the stack, so 0.01 alone is about 0.002-0.003 there.

**Cost and risks.** Each outer step costs about K x a plain step plus two buffer copies; the fine-tune is therefore of the order of tens of minutes at K = 4 (to be measured with a 100-step probe before the run). Risks: (1) the inner lr is tied to the evaluation tuning, so retuning `dyn` afterwards could undo it; (2) the model may learn to depend on adaptation and lose its unadapted quality (log CE with `dyn=0` too); (3) the chunk size the model meets in training (2 windows) is the one tuned at test time, so the ridge may move; (4) tiers already capture repeats, so the stack may take little. A negative result is cheap to record; code is reverted if it fails.

### Meta-trained adaptability: negative, but a plain fine-tune helps a lot (2026-10-15, `runs/t70.sh`, `runs/meta/`)

Built as designed (code in commit `db75b17`, then removed: `examples/tiny_lm/meta_finetune.rs`, first-order, K = 4 chunks of 2 windows, inner SGD lr 0.04 with full rows, outer gradient on the scored rows only via ignored-target CE rows, which needed a two-line kernel change and a test; outer momentum SGD lr 0.01, 4 runs per update, 3000 updates = 96k windows, 0.15% of the original 64M; 330-390 ms per update, transfer-bound). Control: the same code with `inner_lr=0` (every gradient at the slow weights). Model alone on the original held-out halves, `dyn=0.04 dyn_every=2` (no tiers), half 0 / half 1:

| checkpoint | no dyn | dyn |
|---|---|---|
| original (64M windows) | 1.0562 / 1.1586 | 1.0199 / 1.1210 |
| fine-tuned, meta (K = 4) | 1.0313 / 1.1316 | 1.0068 / 1.1087 |
| fine-tuned, control (K = 1) | 1.0314 / 1.1311 | 1.0069 / 1.1089 |

**Meta-training adds nothing**: meta and control agree to 0.0003 everywhere, so the model gets nothing from having seen its own inner steps; within-run loss by chunk position is the same for both. **The control itself is the finding**: 96k windows of fine-tuning without dropout at constant lr 0.01 (momentum 0.9, wd 0.0002) lower the model alone by -0.025 without dynamic evaluation and by -0.013 / -0.012 with it. The 64M-window recipe ended at constant lr 0.05 with dropout 0.1 and no final decay (`lr_decay_frac` exists in `fused_models_check`, but it was tried only on the aesop runs, docs/fusion_results.md round 6), so the checkpoint is un-annealed. The dynamic-evaluation gain shrinks as the model improves (-0.0363 to -0.0245 on half 0). Overlap rule: expect -0.002 to -0.004 in the stack, to be measured; the memory is built from the model's hidden states, so a new checkpoint needs a new cache (cold build ~11 min). Not yet run: the stack with the fine-tuned checkpoint; a proper anneal (linear decay to 0, with and without dropout, `fused_models_check lr_decay_frac=`) as the cleaner way to get the same effect.
