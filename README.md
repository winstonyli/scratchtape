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

## Scope

This is a personal research and learning project, not a library or a
framework. It trains small byte-level language models (d_model 32 to 256) on a few
hundred kilobytes of public-domain text, and its
results are about mechanisms (attention variants, local-SGD averaging,
memory tiers), not benchmark scores. It makes no stability promises, and
nothing here is packaged for reuse.

## Requirements

- **Rust:** edition 2024. The dependency tree declares a minimum of Rust
  1.95; the project is developed on a recent nightly (rustc 1.100) and has
  not been tried on other toolchains.
- **CPU:** x86-64 with AVX2 and FMA. `.cargo/config.toml` builds with
  `+avx2,+fma`, and older CPUs fail with an illegal-instruction error.
- **OS:** Windows 11. The CPU engine (`tensor`, `tape`, `nn`, `optim`) is
  portable Rust, but `gpu_lease.rs` needs `%LOCALAPPDATA%` and the
  `scripts/` drivers assume Git Bash; nothing has been run on Linux or macOS.
- **GPU (optional):** everything under `src/gpu*` and `examples/gpu/` needs
  a Vulkan-capable discrete GPU. It was developed on an AMD Radeon RX 9060
  XT in a USB4 eGPU enclosure (host: Ryzen 7 7840U laptop) and has not been
  run on any other card. Without a discrete GPU the CPU examples still run,
  and the GPU tests are `#[ignore]`d.
  - `gpu.rs` (the raw wgpu matmul) picks DX12 on Windows and falls back to
    any backend; `WGPU_BACKEND=dx12|vulkan` overrides it.
  - `gpu_step/` (cubecl) uses Vulkan on the first discrete GPU, the route to
    the card's matrix cores. `WGPU_BACKEND=dx12` selects DX12 instead, and
    needs the Windows SDK's `dxcompiler.dll` on `PATH`; without it wgpu
    silently falls back to FXC and the step runs ~170x slower
    (`docs/gpu_step_design.md`).
- **Disk and time:** the largest runs (`fused_models_check`, `tier_eval`)
  take minutes to hours and write checkpoints under `runs/` (gitignored).

## AI-assisted development

Most of this code was written with Claude Code (Anthropic's coding agent):
254 of the 255 commits carry a `Co-Authored-By: Claude` trailer (Sonnet 5,
Sonnet 5.5 and Opus 5.5 versions). The direction, the experiments to run,
and the decision to keep or retract a result are the author's. The
trailers are the only record of the split; `git log` is the audit trail.

## Direction: tiered AI

The overarching idea (the user's, 2026-10-02): each tier of the system
should roughly correspond to a hardware level and/or a neuroscience
role.

| Tier | Hardware | Role (as stated) |
|---|---|---|
| Gradient-trained models | GPU | the learned, statistical tier (the transformer LM; `gpu_step/`) |
| Knowledge representation and reasoning (KR&R) | CPU | symbolic structure and rule-based reasoning (the KR&R line in `tiny_lm_corpus.rs`, `differentiable_reasoning/`) |
| Working memory | registers / cache | what is held and manipulated right now |
| Long-term memory and notes | RAM / disk | what is stored and retrieved later |

The mapping is a design principle, not a finished design: how the tiers
talk to each other, and which neuroscience correspondences to commit to,
are open. Stack history and first cross-tier results: `docs/tiers_design.md`, "Stack history". Existing work by tier: the transformer and its GPU-resident step (tier 1); the symbolic-structure and reasoning examples (tier 2); replay buffers, checkpoints and consolidation in `examples/memory_tiers/` (long-term side, toy scale). Experiments are chosen by what they say about the tiers and how they connect.


Current best stack (2026-10-14, `docs/tiers_design.md`; seq-128 byte model trained on 64M windows, dynamic
evaluation, lexicon + word bigram + online kNN memory; held-out CE, nats/byte, halves 0 / 1): model alone
1.0165 / 1.1146, full stack **0.9659 / 1.0620** (lexicon 0.2, words 0.1, kNN weight 0.5, in-document temperature 15, dynamic evaluation in chunks of 2 windows at lr 0.04 (about 25% more loop time than 4 windows at 0.05, which gives 0.9678 / 1.0626), and `dyn_nn=16`, retrieval-driven test-time training: each chunk also takes a step on the 16 train windows nearest to it, -0.0021; before retuning the kNN weight for the 125M-key memory and adding it: 0.9754 / 1.0691; on a third held-out split of seven unseen books, with nothing retuned, 1.0045 / 1.0140 against 1.0078 / 1.0171 for the previous dynamic-evaluation setting and 1.0138 / 1.0203 without `dyn_nn`). The memory holds 125.4M keys of 149 novels as 2-bit integers (one scale per key;
keys randomly rotated, `rotate=1`; 10.6 GB of the 16 GB card) and is searched approximately (IVF, 8 probes of 12288 cells; recall of the
exact top-128 ~0.99). With 4-bit keys the 83M-key memory of 91 novels gives 0.9751 / 1.0687 in 12.4 GB. Building the 125M-key memory takes
~11 min (whitening on the device) and is cached on disk (`memcache=<dir>`, ~30 s to reload); a scored held-out half takes ~59 s
(48 s at 83M keys). The tiers never alter the adaptation, so their searches are batched.

Where the rest sits: word-initial letters carry 45% of the model's CE and the tiers remove only 1.9% of it there (6.5% inside words).
Tried and closed on that front (all negative, `docs/tiers_design.md`): a recency prior for rare words, tier-weight gating, auxiliary
word-start heads, word-start loss weighting, 16384 cells, first-letter count tiers, and (by the evidence, without a run) hashed
previous-word inputs and in-model kNN attention. The one open line is more memory text (about -0.005 per 2.5x), VRAM-bound at 125M keys.

## Quick start

```bash
cargo test --release              # 35 gradient-check / correctness tests (+18 GPU-only: -- --ignored)
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

Full descriptions and every measured result: [`docs/examples.md`](docs/examples.md).

- **Fundamentals**: `xor_mlp`, `xor_es`, `linear_regression`, `byte_tokenizer`.
- **The transformer LM family**: `tiny_lm` (original), `tiny_lm_corpus` (Aesop, 90/10 split; also the symbolic/KR&R investigation), `training_recipe_check` (GPU recipe sweeps), `fused_models_check` (K fused models, the 64M-window recipe), `ngram_baseline` (the count-model bar).
- **Alternatives to attention**: `attention_recall`, `ssm_recall`, `softmax1_comparison`, `attention_null_baseline`.
- **Structure and memory**: `gnn_byte_classification`, `predictive_coding`, `catastrophic_forgetting` and the `memory_tiers` family, differentiable reasoning.
- **Tiers**: `tier_eval` (lexicon, word bigram, online kNN memory, `docs/tiers_design.md`); `gpu_*_check` probes for dispatch overhead, priority and contention.

## Related work

Literature (surveyed 2026-09-23) and a sibling project, each with what
it means here.

**Low-bit key quantisation for the kNN memory**
- Gao & Long 2024, *RaBitQ* ([arXiv 2405.12497](https://arxiv.org/abs/2405.12497)): rotate vectors with a random orthogonal
  matrix, then quantise each coordinate, for nearest-neighbour search with an error bound. Here the rotation is folded into the
  whitener (`rotate=<seed>`, free at query time) and the keys are per-row-scaled uniform integers (2 or 4 bits) with the
  Gaussian-optimal step; a plain max-abs scale without the rotation made 2-bit keys useless (CE worse than the model alone).

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

**Tiered and neuro-symbolic architectures** (surveyed 2026-10-02; abstracts or
paper sections re-fetched, the Nature paper via the DeepMind post)
- Sumers, Yao, Narasimhan, Griffiths 2023, *Cognitive Architectures for
  Language Agents* ([arXiv 2309.02427](https://arxiv.org/abs/2309.02427)):
  CoALA organizes language agents with cognitive-science and symbolic-AI
  ideas. Its section 4.1 splits memory into working memory (the active
  state of the current decision cycle) and long-term memory (episodic,
  semantic, procedural). The closest published map to our working-memory
  and long-term tiers.
- Packer et al. 2023, *MemGPT: Towards LLMs as Operating Systems*
  ([arXiv 2310.08560](https://arxiv.org/abs/2310.08560)): virtual context
  management moves data between a fast context and slower external storage
  like RAM and disk. The same hardware analogy as our cache vs RAM/disk
  tiers, applied at the prompt level rather than inside the model.
- Trinh et al. 2024, *Solving olympiad geometry without human
  demonstrations* (Nature; [DeepMind post](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/)):
  AlphaGeometry pairs a language model that proposes constructions with a
  symbolic deduction engine that does the rigorous steps; 25 of 30 olympiad
  problems. A working GPU-neural plus CPU-symbolic split, which is our
  tier 1 and tier 2.
- Manhaeve et al. 2018, *DeepProbLog*
  ([arXiv 1805.10872](https://arxiv.org/abs/1805.10872), NeurIPS 2018):
  neural predicates inside a probabilistic logic program, trained end to
  end. The tightly coupled alternative to AlphaGeometry's propose-and-check
  loop.
- Khandelwal et al. 2020, *Generalization through Memorization: Nearest
  Neighbor Language Models* ([arXiv 1911.00172](https://arxiv.org/abs/1911.00172),
  ICLR 2020): interpolates a language model with a nearest-neighbor
  datastore of past contexts, no extra training; swapping the datastore
  adapts to a new domain. An external long-term store behind the model.
- Wu et al. 2022, *Memorizing Transformers*
  ([arXiv 2203.08913](https://arxiv.org/abs/2203.08913), ICLR 2022): kNN
  lookup into a non-differentiable memory of past (key, value) pairs;
  gains continue up to 262K tokens of memory, and new functions or theorems
  are usable at test time without retraining. Memory the model reads but
  does not train through.
- He, Neubig & Berg-Kirkpatrick 2021, *Efficient Nearest Neighbor Language
  Models* ([arXiv 2109.04212](https://arxiv.org/abs/2109.04212), EMNLP 2021):
  makes kNN-LM cheaper three ways: skip retrieval at steps a small predictor
  marks as not needing it, prune or merge datastore entries, and PCA the keys
  (256-512 dims matched the full 1024 at 3-4x speed). Speed, not cross-entropy;
  relevant because our run time is ~97% kNN search.
- Grave, Joulin & Usunier 2017, *Improving Neural Language Models with a
  Continuous Cache* ([arXiv 1612.04426](https://arxiv.org/abs/1612.04426),
  ICLR 2017): a cache of
  recent hidden states used as a pointer-style distribution. Adding it to
  kNN-LM gave about 2% perplexity on WikiText-103 (Khandelwal et al. 2020).
  Our online in-document memory plays this role.
- Drozdov et al. 2022, *You can't pick your neighbors, or can you? When and
  how to rely on retrieval in the kNN-LM*
  ([arXiv 2210.15859](https://arxiv.org/abs/2210.15859), Findings of EMNLP
  2022): large n-gram overlap between datastore and test text drives the
  gain, and the interpolation weight is set from retrieval quality (the
  3.8% WikiText-103 perplexity figure is from a review page, unchecked). Our
  uncertainty gate (best fixed weight 0.4; a gate adds ~0.001 nats) is the
  same idea and found little.

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

**Horizontal fusion and codistillation**
- Wang et al. 2021, *Horizontally Fused Training Array* (HFTA, MLSys;
  [arXiv 2102.02344](https://arxiv.org/abs/2102.02344)): fuse K
  same-shape models into the same kernel launches. `DeviceParams::upload_models`
  does this here. Measured on the dispatch-bound step (batch 8): K = 4 costs
  3.95 ms/step against 1.97 for one model, 0.99 per model; at batch 32,
  K = 4 is 13.1 ms (3.28 per model) against 4.67.
- Zhang et al. 2018, *Deep Mutual Learning*
  ([arXiv 1706.00384](https://arxiv.org/abs/1706.00384)) and Anil et al.
  2018, *Large scale distributed neural network training through online
  distillation* ([arXiv 1804.03235](https://arxiv.org/abs/1804.03235)):
  each model also matches its peers' predictions
  (`DeviceTape::with_distill`). **Here it hurt** (2026-09-30, batch 32,
  lr 0.48, dropout 0.3, decay 2e-4, 1M windows, K = 4, held-out CE mean /
  ensemble): α = 0 1.686 / 1.548, α = 0.5 1.783 / 1.727, α = 1 1.940 /
  1.896. Independent seeds averaged beat every other arm, and one model
  with 4× the steps reached 1.7175. Follow-ups: α = 0.1 is the one
  setting that helps per model (1.650 vs 1.686) at a small ensemble cost
  (1.558), and ramping α changes nothing. Details and caveats in
  `docs/fusion_results.md`.
- Lin, Stich, Patel, Jaggi 2020, *Don't Use Large Mini-Batches, Use Local SGD*
  ([arXiv 1808.07217](https://arxiv.org/abs/1808.07217), ICLR 2020):
  post-local SGD (local SGD after a warm phase of large-batch SGD)
  generalizes much better than large-batch training. Our batch-128 vs
  averaged gap (1.755 vs 1.651 on aesop) points the same way; we did not
  try their warm-start variant.
- Xue et al. 2023, *To Repeat or Not To Repeat: Insights from Scaling LLM
  under Token-Crisis* ([arXiv 2305.13230](https://arxiv.org/abs/2305.13230)):
  repeating data across epochs degrades language models, and dropout is
  the one regularizer that clearly helps, with careful tuning as the
  model grows. Our novels6 dropout sweep (0.1 best at 4M windows) is a
  small-scale case of this.
- Liu et al. 2023, *Dropout Reduces Underfitting*
  ([arXiv 2303.01500](https://arxiv.org/abs/2303.01500)): dropout early in
  training reduces gradient-direction variance and lowers training loss
  ("early dropout"); for models that overfit, switching it on only late
  ("late dropout") helps. Untested here: our gap widens late in long
  runs (0.105 at 4M windows, 0.124 at 8M), so a dropout schedule is a
  cheap follow-up.
- Douillard et al. 2023, *DiLoCo* ([arXiv 2311.08105](https://arxiv.org/abs/2311.08105)):
  replicas from one init train locally and periodically average weights.
  Tried in its simplest form (plain mean of the K models every H steps,
  shared init, no outer optimizer): one model at held-out 1.651 for
  H = 100 and 1000, against 1.680–1.696 for single independently trained
  models. A single model with the same 4× data per step (batch 128,
  same steps) is worse, 1.753 held-out, so it isn't a large-batch effect.
  Three seeds each: averaged 1.651 (1.646–1.656) against batch-128 1.755
  (1.752–1.760). DiLoCo's outer Nesterov step (the paper's lr 0.7, mu
  0.9, untuned) did worse than the plain mean (1.662–1.672). Averaging
  plus α = 0.1 reached 1.610 over three seeds (1.605–1.615), the best
  held-out CE here. K = 2 gave 1.625 and K = 8 1.613, so K = 4 is enough;
  α is flat from 0.05 to 0.1 and worse at 0.2 (1.637). A sweep of the
  outer step found nothing reliably better than the plain mean (best
  1.640 at lr 0.5 / mu 0.5, within ~2× the 0.005 rounding noise).

**Retrieval-driven test-time training** (read 2026-10-14, arXiv HTML of v3). Hardt & Sun, "Test-Time Training on Nearest Neighbors for Large
Language Models" (ICLR 2024, arXiv:2305.18466; reproduced in arXiv:2511.16691): for each test sequence, embed it, retrieve its nearest
Pile training sequences (RoBERTa-based 1024-d embeddings, exact L2 index of ~210M sequences), take one Adam step per neighbour chunk
(20 neighbours already give most of the gain; the main setting uses 50), closest first (farthest first is reported significantly worse),
then reset to the pretrained weights; it does not train on the test input itself. Dynamic evaluation (Krause et al. 2018, arXiv:1709.07432;
2019 for Transformers, arXiv:1904.08378) is the same idea with the recent history of the same text as the training signal. This project's
dynamic evaluation (SGD on the chunk just scored, reset per book) is the second; the first is untried here, where the kNN memory could supply
the neighbours (`docs/tiers_design.md`).

**Sibling project: `humble-cortex`** (`../humble-cortex`, predictive
coding in Rust). Its 95 checks overlap here in three places. First, every
model is benchmarked against the simplest baseline for the task
(persistence, OLS, backoff n-gram, kNN). scratchtape's tiny LMs had none
until `ngram_baseline.rs`, which the early models didn't clearly beat
(averaged fused models now do on aesop, 1.610 vs 1.655). Second, partial replay + consolidation
compounded (37.6% / 41.2% alone, 54.0% combined, over 3 repeats), which is
untested here. Third, retention is normalized by how much was learned,
because raw forgetting deltas hid a floor effect.

## Open threads

Full text: [`docs/open_threads.md`](docs/open_threads.md). Live research lines are in `docs/tiers_design.md`.

- **Averaged-model recipe** (2026-10-01): a small-data, overfitting-regime result; the big model's win over the 7-gram comes from data and training.
- **Replay + Fisher consolidation together**; **CLS-ER-style two-speed memory**: untried combinations from the memory-tier work.
- **Parked: fold the copy-and-tweak example families** into one runner; not worth it unless more variants are planned.
- **GPU training backend: done** (`src/gpu_step/`, design and background decisions in `docs/gpu_step_design.md`); `docs/open_threads.md` keeps the contention and CPU-priority findings (GPU-feeding processes need Normal CPU priority).
- **Parked: report the tracel-llvm space-in-path bug upstream**: waits on the owner's go-ahead (posts publicly).

## A note on the name

The crate is `scratchtape`: from-scratch, and literally built around a
`Tape` struct at its center — not a metaphor.

## License

MIT, see [`LICENSE`](LICENSE). Third-party dependencies keep their own licenses
(permissive, plus two MPL-2.0 crates pulled in by cubecl). The four texts in
`data/` are public-domain works, with Project Gutenberg headers removed.
