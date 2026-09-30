use crate::optim::{Optimizer, Sgd};
use crate::tape::{Tape, Var};
use crate::tensor::NdArray;
use std::cell::{Cell, RefCell};

/// xorshift64 PRNG. Hand-rolled, not from a crate - deterministic given a
/// seed, which matters for reproducing a training run exactly.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The whole generator state: `Rng::new(r.state())` continues `r`'s
    /// stream exactly (for resumable checkpoints).
    pub fn state(&self) -> u64 {
        self.0
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64 / u64::MAX as f64) as f32
    }

    /// Box-Muller: two independent uniforms -> one standard normal sample.
    /// Simplest transform to get right from scratch (vs Marsaglia polar /
    /// ziggurat) - not a hot path, so the discarded-sin-half inefficiency
    /// doesn't matter. Guards u1 away from exactly 0 (ln(0) = -inf).
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-9);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Output of a Linear layer's forward pass. Carries the leaf `Var`s
/// (not just the result `y`) because the training loop needs them to read
/// gradients back off the tape and step the optimizer - no hidden state,
/// same explicit style as the tape itself. w/b are pub - useful directly
/// for raw NdArray parameters outside any Linear (e.g. predictive_coding.rs,
/// ssm_recall.rs's weight-tied embed), though apply_grad_with now covers
/// the common Linear-plus-any-Optimizer case without needing them exposed.
pub struct LinearOut {
    pub y: Var,
    pub w: Var,
    pub b: Var,
}

#[derive(Clone)]
pub struct Linear {
    pub w: NdArray, // [in_dim, out_dim]
    pub b: NdArray, // [out_dim]
    // Tracks the most recent leaf pair forward() produced, so apply_grad
    // can catch the weight-tying footgun instead of silently computing a
    // partial gradient - see forward's and apply_grad's doc comments.
    // Interior mutability (not &mut self on forward) for the same reason
    // TransformerBlock's mask_cache already uses it: forward() computing a
    // value is conceptually a read, not a mutation, everywhere else in
    // this file.
    last_leaf: Cell<Option<(Var, Var)>>,
}

impl Linear {
    /// He/Kaiming uniform init: bound = sqrt(6/fan_in). Zero-init would fail -
    /// every hidden unit would receive an identical gradient and never
    /// differentiate (symmetry never breaks). Bias stays zero - asymmetry
    /// already comes from W, no symmetry problem on the bias term.
    /// He (not Xavier) because Xavier assumes a symmetric activation
    /// (tanh/sigmoid); ReLU kills the negative half of the distribution, so
    /// He doubles the variance target to keep activation variance stable
    /// across layers.
    pub fn new(rng: &mut Rng, in_dim: usize, out_dim: usize) -> Self {
        let limit = (6.0 / in_dim as f32).sqrt();
        let w_data = (0..in_dim * out_dim)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * limit)
            .collect();
        let b_data = vec![0.0; out_dim];
        Self {
            w: NdArray::new(w_data, vec![in_dim, out_dim]),
            b: NdArray::new(b_data, vec![out_dim]),
            last_leaf: Cell::new(None),
        }
    }

    /// Builds a Linear directly from already-computed w/b - for callers
    /// constructing weights by some means other than `new`'s random init
    /// (e.g. a hand-built identity/slicing matrix, or an ES-perturbed
    /// copy of an existing layer's weights). Exists because `last_leaf`
    /// being private blocks the struct-literal syntax these callers used
    /// before it was added, not because this needs to be a common path.
    pub fn from_parts(w: NdArray, b: NdArray) -> Self {
        Self { w, b, last_leaf: Cell::new(None) }
    }

    /// Leafs a FRESH copy of w/b into the tape on every call - correct and
    /// the intended usage for the normal one-call-per-training-step
    /// pattern every current example uses, but NOT safe for weight-tied
    /// reuse: calling forward() more than once on the same Linear within
    /// one tape (e.g. weight-tying across an unrolled RNN/SSM loop)
    /// produces a SEPARATE leaf and gradient slot per call, not one shared
    /// gradient. apply_grad now catches this (panics instead of silently
    /// dropping the earlier call's gradient) - for weight-tied reuse, leaf
    /// once yourself and call forward_shared with the same Var every time,
    /// the same pattern ssm_recall.rs already used manually before this
    /// method existed.
    pub fn forward(&self, tape: &mut Tape, x: Var) -> LinearOut {
        let w = tape.leaf(self.w.clone());
        let b = tape.leaf(self.b.clone());
        self.last_leaf.set(Some((w, b)));
        self.forward_shared(tape, x, w, b)
    }

    /// The general form forward() delegates to, for callers that leaf w/b
    /// themselves and want to reuse the same Var across multiple calls
    /// (weight-tying) instead of getting a fresh leaf each time. Doesn't
    /// touch last_leaf - the staleness check is specifically about
    /// protecting forward()'s own fresh-leaf-per-call default from
    /// accidental misuse; a caller using this method has already opted
    /// into managing the Vars themselves.
    pub fn forward_shared(&self, tape: &mut Tape, x: Var, w: Var, b: Var) -> LinearOut {
        let mm = tape.matmul(x, w);
        let y = tape.add(mm, b);
        LinearOut { y, w, b }
    }

    /// Panics if `out` isn't the most recent LinearOut forward() produced -
    /// i.e. forward() was called again on this same Linear (within the
    /// tape's lifetime or a later one) since `out` was created, meaning
    /// out.w/out.b's gradient reflects only part of what should have
    /// updated this layer. Silently proceeding would compute a partial,
    /// wrong weight update instead of erroring - the actual danger this
    /// whole footgun poses. Only checks LinearOuts from forward() (which
    /// records into last_leaf); forward_shared's callers manage their own
    /// Vars and are exempt by design.
    /// Shared by apply_grad and apply_grad_with - both need the same
    /// staleness check before touching the tape, only the actual step
    /// call differs (Sgd-specific vs generic Optimizer).
    fn check_not_stale(&self, out: &LinearOut) {
        assert_eq!(
            self.last_leaf.get(),
            Some((out.w, out.b)),
            "Linear::apply_grad called with a stale LinearOut - forward() was called again on \
             this Linear since this LinearOut was created, so its gradient reflects only that \
             later call, not this one. This is the weight-tying footgun forward()'s doc comment \
             describes: for weight-tied reuse across multiple calls, leaf w/b once yourself and \
             call forward_shared with the same Var each time instead of forward()."
        );
    }

    pub fn apply_grad(&mut self, tape: &Tape, out: &LinearOut, opt: &Sgd) {
        self.check_not_stale(out);
        opt.step(&mut self.w, tape.grad(out.w).unwrap());
        opt.step(&mut self.b, tape.grad(out.b).unwrap());
    }

    /// Generic over Optimizer, so Adam (or anything else implementing it)
    /// can update this layer without the caller hand-rolling a per-
    /// parameter step loop the way every current Adam consumer does today.
    /// State stays caller-owned (create once via O::new_state, thread it
    /// every step) - same explicit, no-hidden-state placement AdamState
    /// already uses, this just removes the duplicated step-invocation code
    /// around it. apply_grad (Sgd-specific, no state to thread) is
    /// unchanged and still the simplest path for the common case.
    pub fn apply_grad_with<O: Optimizer>(&mut self, tape: &Tape, out: &LinearOut, opt: &O, w_state: &mut O::State, b_state: &mut O::State) {
        self.check_not_stale(out);
        opt.step(&mut self.w, tape.grad(out.w).unwrap(), w_state);
        opt.step(&mut self.b, tape.grad(out.b).unwrap(), b_state);
    }

    /// Flattens to raw floats for checkpointing - no headers or shape
    /// metadata, since the same architecture code that saves also loads,
    /// so shapes are already known statically rather than needing to be
    /// self-described in the file.
    pub fn to_flat(&self) -> Vec<f32> {
        let mut out = self.w.data.clone();
        out.extend_from_slice(&self.b.data);
        out
    }

    /// Inverse of to_flat. `offset` is threaded through by the caller
    /// (e.g. TransformerBlock::from_flat) so composite structs can just
    /// concatenate calls to their children's from_flat in order.
    pub fn from_flat(data: &[f32], offset: &mut usize, in_dim: usize, out_dim: usize) -> Self {
        let w_len = in_dim * out_dim;
        let w = NdArray::new(data[*offset..*offset + w_len].to_vec(), vec![in_dim, out_dim]);
        *offset += w_len;
        let b = NdArray::new(data[*offset..*offset + out_dim].to_vec(), vec![out_dim]);
        *offset += out_dim;
        Self { w, b, last_leaf: Cell::new(None) }
    }
}

pub struct EmbeddingOut {
    pub y: Var,
    pub table: Var,
}

/// Token/positional embedding table - a thin wrapper over Tape::gather,
/// same "persisted NdArray + forward + apply_grad" shape as Linear. No new
/// backward math here at all: correctness is already covered by Gather's
/// own gradient-check test (including the repeated-index case), so this
/// gets a lighter sanity test rather than a redundant finite-difference one.
/// In the library, not example-local like the tokenizer - both token and
/// positional embeddings need this, clearing the "2+ consumers" bar before
/// any code was written, same as Gather itself.
#[derive(Clone)]
pub struct Embedding {
    pub table: NdArray, // [vocab_size, d_model]
    last_leaf: Cell<Option<Var>>, // see Linear's identical field for why
}

impl Embedding {
    /// Small-scale uniform init (+/- 0.02), not He/Kaiming like Linear.
    /// He-init's fan-in variance-preservation reasoning is about signal
    /// passing through a matmul into a nonlinearity - it doesn't apply to a
    /// lookup table, where each row is an independently learned vector with
    /// no fan-in at all. Small init keeps initial embedding norms modest,
    /// standard practice (e.g. GPT-2 uses a similar small-scale init).
    pub fn new(rng: &mut Rng, vocab_size: usize, d_model: usize) -> Self {
        let scale = 0.02;
        let data = (0..vocab_size * d_model).map(|_| (rng.next_f32() * 2.0 - 1.0) * scale).collect();
        Self { table: NdArray::new(data, vec![vocab_size, d_model]), last_leaf: Cell::new(None) }
    }

    /// Same fresh-leaf-per-call caveat as Linear::forward - see its doc
    /// comment. Not weight-tie-safe across multiple calls within one tape;
    /// apply_grad catches misuse the same way Linear's does. Use
    /// forward_shared with a Var you leaf yourself for tied reuse.
    pub fn forward(&self, tape: &mut Tape, indices: &[usize]) -> EmbeddingOut {
        let table = tape.leaf(self.table.clone());
        self.last_leaf.set(Some(table));
        self.forward_shared(tape, indices, table)
    }

    /// The general form forward() delegates to - see Linear::forward_shared.
    pub fn forward_shared(&self, tape: &mut Tape, indices: &[usize], table: Var) -> EmbeddingOut {
        let y = tape.gather(table, indices);
        EmbeddingOut { y, table }
    }

    pub fn apply_grad(&mut self, tape: &Tape, out: &EmbeddingOut, opt: &Sgd) {
        assert_eq!(
            self.last_leaf.get(),
            Some(out.table),
            "Embedding::apply_grad called with a stale EmbeddingOut - see Linear::apply_grad's \
             panic message for the full explanation; use forward_shared for tied reuse."
        );
        opt.step(&mut self.table, tape.grad(out.table).unwrap());
    }

    pub fn to_flat(&self) -> Vec<f32> {
        self.table.data.clone()
    }

    pub fn from_flat(data: &[f32], offset: &mut usize, vocab_size: usize, d_model: usize) -> Self {
        let len = vocab_size * d_model;
        let table = NdArray::new(data[*offset..*offset + len].to_vec(), vec![vocab_size, d_model]);
        *offset += len;
        Self { table, last_leaf: Cell::new(None) }
    }
}

pub struct LayerNormOut {
    pub y: Var,
    pub gamma: Var,
    pub beta: Var,
}

/// Normalizes across the last (feature) axis to mean 0, variance 1, then
/// applies a learned per-feature scale/shift. Structurally like Linear (has
/// persistent trainable state), not like Tape::softmax (a stateless, purely
/// compositional function) - that's why this is a struct with its own
/// forward/apply_grad, not a bare Tape method.
///
/// Composed almost entirely from existing ops - mean/variance are just
/// SumLastAxis + Scale(1/d), centering is the already-broadcasting Sub,
/// normalizing is the already-broadcasting Div. The one new primitive
/// this needed was a differentiable Sqrt (Tape only had NdArray::sqrt,
/// built non-differentiable for Adam's use).
#[derive(Clone)]
pub struct LayerNorm {
    pub gamma: NdArray, // [1, d_model]
    pub beta: NdArray,  // [1, d_model]
    eps: f32,
    last_leaf: Cell<Option<(Var, Var)>>, // see Linear's identical field for why
}

impl LayerNorm {
    /// gamma=1, beta=0 (identity transform beyond normalization) - not
    /// randomized like Linear's He-init. Linear needed randomness to break
    /// symmetry between hidden units; gamma/beta are a single row broadcast
    /// identically over every position, so there's no symmetry-collapse
    /// risk here to break.
    pub fn new(d_model: usize) -> Self {
        Self {
            gamma: NdArray::new(vec![1.0; d_model], vec![1, d_model]),
            beta: NdArray::new(vec![0.0; d_model], vec![1, d_model]),
            eps: 1e-5,
            last_leaf: Cell::new(None),
        }
    }

    /// Same fresh-leaf-per-call caveat as Linear::forward - see its doc
    /// comment. Not weight-tie-safe across multiple calls within one tape;
    /// apply_grad catches misuse the same way Linear's does. Use
    /// forward_shared with Vars you leaf yourself for tied reuse.
    pub fn forward(&self, tape: &mut Tape, x: Var) -> LayerNormOut {
        let gamma = tape.leaf(self.gamma.clone());
        let beta = tape.leaf(self.beta.clone());
        self.last_leaf.set(Some((gamma, beta)));
        self.forward_shared(tape, x, gamma, beta)
    }

    /// The general form forward() delegates to - see Linear::forward_shared.
    pub fn forward_shared(&self, tape: &mut Tape, x: Var, gamma: Var, beta: Var) -> LayerNormOut {
        let d = tape.value(x).shape[1] as f32;
        let sum = tape.sum_last_axis(x);
        let mean = tape.scale(sum, 1.0 / d);
        let centered = tape.sub(x, mean);
        let sq = tape.mul(centered, centered);
        let sum_sq = tape.sum_last_axis(sq);
        // Biased variance (divide by d, not d-1) - matches real LayerNorm
        // implementations; Bessel's correction would be the wrong "fix" here.
        let variance = tape.scale(sum_sq, 1.0 / d);
        let eps_leaf = tape.leaf(NdArray::scalar(self.eps));
        let variance_eps = tape.add(variance, eps_leaf);
        let std_dev = tape.sqrt(variance_eps);
        let normalized = tape.div(centered, std_dev);
        let scaled = tape.mul(normalized, gamma);
        let y = tape.add(scaled, beta);
        LayerNormOut { y, gamma, beta }
    }

    pub fn apply_grad(&mut self, tape: &Tape, out: &LayerNormOut, opt: &Sgd) {
        assert_eq!(
            self.last_leaf.get(),
            Some((out.gamma, out.beta)),
            "LayerNorm::apply_grad called with a stale LayerNormOut - see Linear::apply_grad's \
             panic message for the full explanation; use forward_shared for tied reuse."
        );
        opt.step(&mut self.gamma, tape.grad(out.gamma).unwrap());
        opt.step(&mut self.beta, tape.grad(out.beta).unwrap());
    }

    pub fn to_flat(&self) -> Vec<f32> {
        let mut out = self.gamma.data.clone();
        out.extend_from_slice(&self.beta.data);
        out
    }

    /// eps isn't serialized - it's always the same fixed constant `new`
    /// sets, not something training ever changes.
    pub fn from_flat(data: &[f32], offset: &mut usize, d_model: usize) -> Self {
        let gamma = NdArray::new(data[*offset..*offset + d_model].to_vec(), vec![1, d_model]);
        *offset += d_model;
        let beta = NdArray::new(data[*offset..*offset + d_model].to_vec(), vec![1, d_model]);
        *offset += d_model;
        Self { gamma, beta, eps: 1e-5, last_leaf: Cell::new(None) }
    }

    /// A table of `rows` gamma/beta rows ([rows, d], gamma=1, beta=0) for
    /// `forward_rows` - one LayerNorm per head, stored stacked.
    pub fn table(rows: usize, d: usize) -> Self {
        let mut ln = Self::new(rows * d);
        ln.gamma.shape = vec![rows, d];
        ln.beta.shape = vec![rows, d];
        ln
    }

    /// `from_flat` for a `table`.
    pub fn table_from_flat(data: &[f32], offset: &mut usize, rows: usize, d: usize) -> Self {
        let mut ln = Self::from_flat(data, offset, rows * d);
        ln.gamma.shape = vec![rows, d];
        ln.beta.shape = vec![rows, d];
        ln
    }

    /// `forward` for a `table`: x's row i is scaled and shifted by table row
    /// `rows[i]`, gathered. Its LayerNormOut's gamma/beta are the whole
    /// tables, so `apply_grad` works unchanged.
    pub fn forward_rows(&self, tape: &mut Tape, x: Var, rows: &[usize]) -> LayerNormOut {
        let gamma = tape.leaf(self.gamma.clone());
        let beta = tape.leaf(self.beta.clone());
        self.last_leaf.set(Some((gamma, beta)));
        let (g, b) = (tape.gather(gamma, rows), tape.gather(beta, rows));
        let y = self.forward_shared(tape, x, g, b).y;
        LayerNormOut { y, gamma, beta }
    }
}

/// Strictly upper-triangular -inf (j > i only - the diagonal stays
/// unmasked so a position can always attend to itself, avoiding an
/// all-masked row -> -inf - -inf = NaN in softmax's max subtraction). Safe
/// with this engine's softmax: the row max is finite (the diagonal), and
/// exp(-inf) is an exact, finite 0.0 under IEEE 754, so -inf never survives
/// past the Exp step - nothing downstream ever sees an infinity.
/// Tiled batch_size times down the rows: each batch chunk gets its own
/// independent seq_len x seq_len causal pattern. This mask is added
/// (exact-shape, no broadcast) directly to batched attention scores
/// (shape [batch_size*seq_len, seq_len], from Tape::batched_matmul), which
/// only ever contains within-chunk scores in the first place - no
/// cross-batch masking needed here, unlike a naive dense-stack approach
/// that would need a block-diagonal mask to hide cross-batch entries it
/// never should have computed.
fn causal_mask(seq_len: usize, batch_size: usize) -> NdArray {
    let mut data = vec![0.0f32; batch_size * seq_len * seq_len];
    for chunk in 0..batch_size {
        let base = chunk * seq_len * seq_len;
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                data[base + i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    NdArray::new(data, vec![batch_size * seq_len, seq_len])
}

pub struct TransformerBlockOut {
    pub y: Var,
    /// Every head's softmax attention weights as one [B*H*T, T] Var, head
    /// (b, h) at rows (b*H + h)*T.. (Tape::split_heads' order). Computed
    /// regardless, exposed for diagnostics; `head_weights_of` gives one
    /// head's [B*T, T] view.
    pub head_weights: Var,
    /// Every sublayer's own *Out below is `pub` (not just `y`/`head_weights`
    /// above) so a caller can read `tape.grad(...)` on any individual
    /// sublayer's parameters directly - needed for per-sublayer diagnostics
    /// like Fisher-information estimation (memory_tier_multigen_diverse_consolidate_fisher.rs),
    /// which previously had to treat an entire block as one opaque unit
    /// with no way to tell which of its 8 sublayers the loss actually
    /// depended on.
    pub ln1_out: LayerNormOut,
    /// The fused Q/K/V projection (see `TransformerBlock::qkv` for columns).
    pub qkv_out: LinearOut,
    pub out_proj_out: LinearOut,
    pub ln2_out: LayerNormOut,
    pub ffn1_out: LinearOut,
    pub ffn2_out: LinearOut,
    /// (Q-norm, K-norm) outputs, gamma/beta being the [H, d_k] tables;
    /// None unless the block was built `with_qk_norm`.
    pub qk_norm_out: Option<(LayerNormOut, LayerNormOut)>,
    /// The fused gate Linear's output (pre-sigmoid) and the sigmoid values
    /// applied to the merged heads, both [B*T, D]; None unless built
    /// `with_attn_gate`.
    pub attn_gate_out: Option<LinearOut>,
    pub attn_gate_value: Option<Var>,
    /// The [H, 1] sink-logit leaf; None unless built `with_sink_logit`.
    pub sink_leaf: Option<Var>,
    n_heads: usize,
    batch_size: usize,
}

impl TransformerBlockOut {
    /// Head h's attention weights as [B*T, T], samples stacked as in the
    /// block's input.
    pub fn head_weights_of(&self, tape: &Tape, h: usize) -> NdArray {
        let w = tape.value(self.head_weights);
        let cols = w.shape[1];
        let t_len = w.shape[0] / (self.batch_size * self.n_heads);
        let mut data = Vec::with_capacity(self.batch_size * t_len * cols);
        for b in 0..self.batch_size {
            let start = (b * self.n_heads + h) * t_len * cols;
            data.extend_from_slice(&w.data[start..start + t_len * cols]);
        }
        NdArray::new(data, vec![self.batch_size * t_len, cols])
    }
}

/// Linears side by side: one Linear whose columns are each part's columns
/// in order.
fn fuse_columns(parts: &[Linear]) -> Linear {
    let ws: Vec<&NdArray> = parts.iter().map(|l| &l.w).collect();
    let bs: Vec<&NdArray> = parts.iter().map(|l| &l.b).collect();
    Linear::from_parts(NdArray::concat_last_axis(&ws), NdArray::concat_last_axis(&bs))
}

/// Pre-norm transformer block: x + Attn(LN(x)), then x + FFN(LN(x)).
/// Every sublayer already exists (LayerNorm, Linear, relu, Add, softmax) -
/// this is composition, not new machinery. Its multi-head attention is NOT
/// multihead_attention_recall.rs's hand-set identity-slice version (that
/// demo was illustrative, deliberately untrained) - every head has its own
/// real He-initialized, trainable Q/K/V projection, stored fused in one
/// Linear. Heads are batched: split_heads stacks every (sample, head) pair
/// so attention is one batched_matmul over B*H, the layout the GPU step's
/// kernels use (docs/gpu_step_design.md). Kept inlined rather than factored
/// into its own MultiHeadAttention struct - this block is currently its
/// only consumer.
#[derive(Clone)]
pub struct TransformerBlock {
    n_heads: usize,
    d_k: usize,
    ln1: LayerNorm,
    /// Every head's Q, K and V projections as one [D, 3D] Linear: column
    /// `part * D + h * d_k + j` is column j of head h's Q (part 0), K (1)
    /// or V (2).
    qkv: Linear,
    out_proj: Linear,
    ln2: LayerNorm,
    ffn1: Linear,
    ffn2: Linear,
    /// (Q, K) LayerNorms over d_k, per head: gamma/beta are [H, d_k]
    /// tables, row h for head h. QK-norm as ViT-22B does it (Dehghani et
    /// al. 2023), 1/sqrt(d_k) kept. None = no QK-norm. An earlier version
    /// L2-normalized Q/K with no learnable scale, capping logits at
    /// +/-1/sqrt(d_k) = +/-0.25 and making every trained head a near-exact
    /// mean-pool (attention_uniformity_check.rs); LayerNorm's gamma is the
    /// learnable scale that version lacked, and at init logits already span
    /// ~+/-sqrt(d_k) rather than +/-1/sqrt(d_k).
    qk_norm: Option<(LayerNorm, LayerNorm)>,
    /// Per-head output gates, Qwen's gated attention (Qiu et al. 2025, the
    /// elementwise "G1" form): head_out *= sigmoid(LN1(x) W_g + b_g). A way
    /// for a head to abstain (gate -> 0) while its scores stay under plain,
    /// shift-invariant softmax. One [D, D] Linear: column h*d_k + j gates
    /// head h's output column j. None = no gate.
    attn_gate: Option<Linear>,
    /// Per-head learnable sink logits s_h, [H, 1] - a phantom key with zero
    /// value: weights = exp(x_i) / (exp(s_h) + sum_j exp(x_j)) =
    /// softmax1(x - s_h), so s_h = 0 is exactly softmax1. None = no sink.
    sink_logits: Option<NdArray>,
    /// Causal mask is identical on every call for a fixed (seq_len,
    /// batch_size) - cached rather than rebuilt (fresh allocation +
    /// O(seq_len^2) fill loop, now O(batch_size*n_heads*seq_len^2)) on every
    /// forward pass. RefCell, not a signature change to &mut self: forward()
    /// stays &self so every existing call site (tiny_lm.rs,
    /// catastrophic_forgetting.rs, etc.) needs no changes. Recomputed only
    /// when seq_len/batch_size actually change from what's cached; still
    /// pays one clone per call to hand ownership to the leaf, since
    /// Tape::leaf takes an owned NdArray and the tape itself is rebuilt
    /// fresh every step - only the fill loop is eliminated, not the
    /// per-call allocation entirely.
    mask_cache: RefCell<Option<(usize, usize, NdArray)>>,
}

impl TransformerBlock {
    /// Draws each head's Q, then K, then V projection in the same RNG
    /// order as the per-head layout before batching, then fuses them, so a
    /// given seed gives the same parameters as before.
    pub fn new(rng: &mut Rng, d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let d_k = d_model / n_heads;
        let heads: Vec<Linear> = (0..3 * n_heads).map(|_| Linear::new(rng, d_model, d_k)).collect();
        Self {
            n_heads,
            d_k,
            ln1: LayerNorm::new(d_model),
            qkv: fuse_columns(&heads),
            out_proj: Linear::new(rng, d_model, d_model),
            ln2: LayerNorm::new(d_model),
            ffn1: Linear::new(rng, d_model, d_ff),
            ffn2: Linear::new(rng, d_ff, d_model),
            qk_norm: None,
            attn_gate: None,
            sink_logits: None,
            mask_cache: RefCell::new(None),
        }
    }

    /// Adds fresh per-head Q/K LayerNorms (gamma=1, beta=0). Consumes no
    /// rng, so every other parameter matches a plain block from the same
    /// seed exactly.
    pub fn with_qk_norm(mut self) -> Self {
        self.qk_norm = Some((LayerNorm::table(self.n_heads, self.d_k), LayerNorm::table(self.n_heads, self.d_k)));
        self
    }

    /// Adds per-head output gates. Takes its own rng so a caller can keep
    /// the rest of the model's init identical to a gateless run.
    pub fn with_attn_gate(mut self, rng: &mut Rng) -> Self {
        let d_model = self.out_proj.w.shape[0];
        let gates: Vec<Linear> = (0..self.n_heads).map(|_| Linear::new(rng, d_model, self.d_k)).collect();
        self.attn_gate = Some(fuse_columns(&gates));
        self
    }

    /// Adds per-head sink logits, initialized to 0 (= softmax1). Consumes
    /// no rng.
    pub fn with_sink_logit(mut self) -> Self {
        self.sink_logits = Some(NdArray::new(vec![0.0; self.n_heads], vec![self.n_heads, 1]));
        self
    }

    /// Checkpoint counterpart of the `with_*` builders: reads whichever
    /// extras the block was saved with, in `to_flat`'s order (Q/K norms,
    /// gate, sinks - all after a plain block's layout).
    pub fn extras_from_flat(mut self, data: &[f32], offset: &mut usize, qk_norm: bool, gate: bool, sink: bool) -> Self {
        let (h, d_k, d_model) = (self.n_heads, self.d_k, self.out_proj.w.shape[0]);
        if qk_norm {
            let q = LayerNorm::table_from_flat(data, offset, h, d_k);
            self.qk_norm = Some((q, LayerNorm::table_from_flat(data, offset, h, d_k)));
        }
        if gate {
            self.attn_gate = Some(Linear::from_flat(data, offset, d_model, d_model));
        }
        if sink {
            self.sink_logits = Some(NdArray::new(data[*offset..*offset + h].to_vec(), vec![h, 1]));
            *offset += h;
        }
        self
    }

    /// Unbatched convenience wrapper - every existing call site (tiny_lm.rs,
    /// catastrophic_forgetting.rs, etc.) uses this unchanged; batch_size=1
    /// makes forward_batched's per-chunk logic degenerate to exactly what
    /// this function used to compute directly.
    pub fn forward(&self, tape: &mut Tape, x: Var) -> TransformerBlockOut {
        self.forward_batched(tape, x, 1)
    }

    /// `x` holds `batch_size` independent sequences stacked along the row
    /// axis (rows [0,seq_len) = sample 0, [seq_len,2*seq_len) = sample 1,
    /// etc. - not a genuine batch dimension, since NdArray stays 2D-only).
    /// Every sublayer except attention is already row-independent (Linear,
    /// LayerNorm, the residual adds, softmax's own last-axis reduction) and
    /// needs no change at all under stacking. Attention is the one op that
    /// mixes rows together (Q@Kᵀ, weights@V), so it alone uses
    /// Tape::batched_matmul to keep each (sample, head)'s attention confined
    /// to its own chunk - see that function's doc comment for why a naive
    /// dense-stack-then-mask version was rejected (wastes O(batch) more
    /// compute than this).
    pub fn forward_batched(&self, tape: &mut Tape, x: Var, batch_size: usize) -> TransformerBlockOut {
        self.forward_full(tape, x, batch_size, false)
    }

    /// Same as forward_batched, plus an opt-in switch for softmax1
    /// (Tape::softmax1, "Attention Is Off By One") instead of plain softmax.
    /// QK-norm is not a switch here: it has parameters, so it's part of the
    /// architecture (`with_qk_norm`) and applies whenever the block has it.
    /// A separate most-general method rather than new parameters on
    /// forward_batched itself, so every existing call site stays on plain
    /// softmax - not a default change to already-recorded experiments.
    ///
    /// Inherits Linear::forward's fresh-leaf-per-call caveat transitively,
    /// through every Linear/LayerNorm this composes (qkv, out_proj, ln1,
    /// ln2, ffn1, ffn2, Q/K norms, gate) - calling this more than once on the
    /// same TransformerBlock within one tape is not weight-tie-safe either.
    pub fn forward_full(&self, tape: &mut Tape, x: Var, batch_size: usize, use_softmax1: bool) -> TransformerBlockOut {
        let (n_heads, d) = (self.n_heads, self.n_heads * self.d_k);
        let heads = batch_size * n_heads;
        let seq_len = tape.value(x).shape[0] / batch_size;
        let mask_value = {
            let mut cache = self.mask_cache.borrow_mut();
            let needs_recompute = !matches!(&*cache, Some((cached_len, cached_heads, _)) if *cached_len == seq_len && *cached_heads == heads);
            if needs_recompute {
                *cache = Some((seq_len, heads, causal_mask(seq_len, heads)));
            }
            cache.as_ref().unwrap().2.clone()
        };
        let mask = tape.leaf(mask_value);
        // Stacked-head row i belongs to head (i / seq_len) % H: the table
        // row per-head parameters (QK-norm, sinks) gather for it.
        let head_of_row: Vec<usize> = (0..heads * seq_len).map(|i| (i / seq_len) % n_heads).collect();

        let ln1_out = self.ln1.forward(tape, x);
        let qkv_out = self.qkv.forward(tape, ln1_out.y);
        let q = tape.split_heads(qkv_out.y, 0..d, n_heads, batch_size);
        let k = tape.split_heads(qkv_out.y, d..2 * d, n_heads, batch_size);
        let v = tape.split_heads(qkv_out.y, 2 * d..3 * d, n_heads, batch_size);

        let (q, k, qk_norm_out) = match &self.qk_norm {
            Some((q_norm, k_norm)) => {
                let (qn, kn) = (q_norm.forward_rows(tape, q, &head_of_row), k_norm.forward_rows(tape, k, &head_of_row));
                (qn.y, kn.y, Some((qn, kn)))
            }
            None => (q, k, None),
        };
        let scores = tape.batched_matmul(q, k, heads, true);
        let scaled = tape.scale(scores, 1.0 / (self.d_k as f32).sqrt());
        let masked = tape.add(scaled, mask);
        let (head_weights, sink_leaf) = if let Some(s) = &self.sink_logits {
            let s_leaf = tape.leaf(s.clone());
            let s_rows = tape.gather(s_leaf, &head_of_row);
            let shifted = tape.sub(masked, s_rows);
            (tape.softmax1(shifted), Some(s_leaf))
        } else if use_softmax1 {
            (tape.softmax1(masked), None)
        } else {
            (tape.softmax(masked), None)
        };
        let head_outs = tape.batched_matmul(head_weights, v, heads, false);
        let merged = tape.merge_heads(head_outs, n_heads, batch_size);
        let (attn, attn_gate_out, attn_gate_value) = match &self.attn_gate {
            Some(gate) => {
                let gate_out = gate.forward(tape, ln1_out.y);
                // sigmoid(z) = 1 / (1 + exp(-z)), composed from existing ops.
                let neg = tape.scale(gate_out.y, -1.0);
                let e = tape.exp(neg);
                let one = tape.leaf(NdArray::scalar(1.0));
                let denom = tape.add(e, one);
                let one_again = tape.leaf(NdArray::scalar(1.0));
                let g = tape.div(one_again, denom);
                (tape.mul(merged, g), Some(gate_out), Some(g))
            }
            None => (merged, None, None),
        };

        let out_proj_out = self.out_proj.forward(tape, attn);
        let x1 = tape.add(x, out_proj_out.y); // residual

        let ln2_out = self.ln2.forward(tape, x1);
        let ffn1_out = self.ffn1.forward(tape, ln2_out.y);
        let hidden = tape.relu(ffn1_out.y);
        let ffn2_out = self.ffn2.forward(tape, hidden);
        let y = tape.add(x1, ffn2_out.y); // residual

        TransformerBlockOut {
            y,
            head_weights,
            ln1_out,
            qkv_out,
            out_proj_out,
            ln2_out,
            ffn1_out,
            ffn2_out,
            qk_norm_out,
            attn_gate_out,
            attn_gate_value,
            sink_leaf,
            n_heads,
            batch_size,
        }
    }

    pub fn apply_grad(&mut self, tape: &Tape, out: &TransformerBlockOut, opt: &Sgd) {
        self.ln1.apply_grad(tape, &out.ln1_out, opt);
        self.qkv.apply_grad(tape, &out.qkv_out, opt);
        self.out_proj.apply_grad(tape, &out.out_proj_out, opt);
        self.ln2.apply_grad(tape, &out.ln2_out, opt);
        self.ffn1.apply_grad(tape, &out.ffn1_out, opt);
        self.ffn2.apply_grad(tape, &out.ffn2_out, opt);
        if let (Some((q_norm, k_norm)), Some((q_out, k_out))) = (&mut self.qk_norm, &out.qk_norm_out) {
            q_norm.apply_grad(tape, q_out, opt);
            k_norm.apply_grad(tape, k_out, opt);
        }
        if let (Some(gate), Some(gate_out)) = (&mut self.attn_gate, &out.attn_gate_out) {
            gate.apply_grad(tape, gate_out, opt);
        }
        if let (Some(s), Some(leaf)) = (&mut self.sink_logits, out.sink_leaf) {
            opt.step(s, tape.grad(leaf).unwrap());
        }
    }

    /// Purely mechanical - concatenates each sub-component's own to_flat in
    /// a fixed order: ln1, qkv, out_proj, ln2, ffn1, ffn2 - the layout
    /// `gpu_step` uses too. Extras (Q/K norm tables, gate, sinks - if any)
    /// go last, so a plain block's layout is unchanged and `from_flat`
    /// reads it; `extras_from_flat` reads the rest.
    pub fn to_flat(&self) -> Vec<f32> {
        let mut out = self.ln1.to_flat();
        out.extend(self.qkv.to_flat());
        out.extend(self.out_proj.to_flat());
        out.extend(self.ln2.to_flat());
        out.extend(self.ffn1.to_flat());
        out.extend(self.ffn2.to_flat());
        if let Some((q_norm, k_norm)) = &self.qk_norm {
            out.extend(q_norm.to_flat());
            out.extend(k_norm.to_flat());
        }
        if let Some(gate) = &self.attn_gate {
            out.extend(gate.to_flat());
        }
        if let Some(s) = &self.sink_logits {
            out.extend_from_slice(&s.data);
        }
        out
    }

    /// Same (d_model, n_heads, d_ff) arguments as `new` - shapes aren't
    /// self-described in the file, so the caller must reconstruct with the
    /// identical architecture that produced the checkpoint (chain
    /// `extras_from_flat` for a block built with any `with_*` extras).
    pub fn from_flat(data: &[f32], offset: &mut usize, d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        let ln1 = LayerNorm::from_flat(data, offset, d_model);
        let qkv = Linear::from_flat(data, offset, d_model, 3 * d_model);
        let out_proj = Linear::from_flat(data, offset, d_model, d_model);
        let ln2 = LayerNorm::from_flat(data, offset, d_model);
        let ffn1 = Linear::from_flat(data, offset, d_model, d_ff);
        let ffn2 = Linear::from_flat(data, offset, d_ff, d_model);
        Self { n_heads, d_k: d_model / n_heads, ln1, qkv, out_proj, ln2, ffn1, ffn2, qk_norm: None, attn_gate: None, sink_logits: None, mask_cache: RefCell::new(None) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Statistical, not exact-match - next_gaussian is inherently stochastic.
    /// Large sample size keeps this from being flaky: standard error of the
    /// mean here is ~1/sqrt(20000) =~ 0.007, so a 0.05 tolerance is well clear
    /// of normal sampling noise, not a hand-tuned threshold.
    #[test]
    fn next_gaussian_has_mean_zero_var_one() {
        let mut rng = Rng::new(7);
        let n = 20_000;
        let samples: Vec<f32> = (0..n).map(|_| rng.next_gaussian()).collect();
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        let var: f32 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean {mean} too far from 0");
        assert!((var - 1.0).abs() < 0.1, "variance {var} too far from 1");
    }

    fn transformer_block_loss(block: &TransformerBlock, x_data: &[f32], seq_len: usize, d_model: usize) -> f32 {
        let mut tape = Tape::new();
        let x = tape.leaf(NdArray::new(x_data.to_vec(), vec![seq_len, d_model]));
        let out = block.forward(&mut tape, x);
        let loss = tape.sum(out.y);
        tape.value(loss).data[0]
    }

    /// dL/dx is the primary check - x flows through every sublayer, both
    /// residual paths, both layer norms, all heads, and the FFN, so a
    /// correct dx is strong evidence the whole graph composes correctly
    /// (same logic as the very first gradient-check test in tape.rs).
    /// Spot-checks one attention weight and one FFN weight as insurance
    /// against a bug that happens to cancel out in dx alone.
    #[test]
    fn transformer_block_backward_matches_finite_difference() {
        let mut rng = Rng::new(1);
        let (d_model, n_heads, d_ff, seq_len) = (8, 2, 16, 4);
        let mut block = TransformerBlock::new(&mut rng, d_model, n_heads, d_ff);

        let x_data: Vec<f32> = (0..seq_len * d_model).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();

        let mut tape = Tape::new();
        let x = tape.leaf(NdArray::new(x_data.clone(), vec![seq_len, d_model]));
        let out = block.forward(&mut tape, x);
        let loss = tape.sum(out.y);
        tape.backward(loss);
        let x_grad = tape.grad(x).unwrap().clone();
        let out_proj_w_grad = tape.grad(out.out_proj_out.w).unwrap().clone();
        let ffn1_w_grad = tape.grad(out.ffn1_out.w).unwrap().clone();

        let eps = 1e-3;
        for i in 0..x_data.len() {
            let mut xp = x_data.clone();
            xp[i] += eps;
            let mut xm = x_data.clone();
            xm[i] -= eps;
            let numerical = (transformer_block_loss(&block, &xp, seq_len, d_model)
                - transformer_block_loss(&block, &xm, seq_len, d_model))
                / (2.0 * eps);
            assert!(
                (numerical - x_grad.data[i]).abs() < 1e-2,
                "x grad[{i}] mismatch: numerical {numerical} vs analytical {}",
                x_grad.data[i]
            );
        }

        let orig = block.out_proj.w.data[0];
        block.out_proj.w.data[0] = orig + eps;
        let lp = transformer_block_loss(&block, &x_data, seq_len, d_model);
        block.out_proj.w.data[0] = orig - eps;
        let lm = transformer_block_loss(&block, &x_data, seq_len, d_model);
        block.out_proj.w.data[0] = orig;
        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (numerical - out_proj_w_grad.data[0]).abs() < 1e-2,
            "out_proj.w[0] grad mismatch: numerical {numerical} vs analytical {}",
            out_proj_w_grad.data[0]
        );

        let orig = block.ffn1.w.data[0];
        block.ffn1.w.data[0] = orig + eps;
        let lp = transformer_block_loss(&block, &x_data, seq_len, d_model);
        block.ffn1.w.data[0] = orig - eps;
        let lm = transformer_block_loss(&block, &x_data, seq_len, d_model);
        block.ffn1.w.data[0] = orig;
        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (numerical - ffn1_w_grad.data[0]).abs() < 1e-2,
            "ffn1.w[0] grad mismatch: numerical {numerical} vs analytical {}",
            ffn1_w_grad.data[0]
        );
    }

    /// Same shape of check as transformer_block_backward_matches_finite_difference,
    /// on a `with_qk_norm` block - QK-norm is an already-gradient-checked
    /// LayerNorm, but this checks the real wiring inside forward_full itself.
    /// Spot-checks a Q weight (a bug computing scores from the
    /// un-normalized q/k would show here) and head 1's K-norm gamma (the
    /// learnable scale itself must reach the loss, through the right table
    /// row), plus the checkpoint round trip, since the norms are appended
    /// after the plain layout.
    #[test]
    fn transformer_block_qknorm_backward_matches_finite_difference() {
        let mut rng = Rng::new(2);
        let (d_model, n_heads, d_ff, seq_len) = (8, 2, 16, 4);
        let block = TransformerBlock::new(&mut rng, d_model, n_heads, d_ff).with_qk_norm();

        let flat = block.to_flat();
        let mut offset = 0;
        let restored = TransformerBlock::from_flat(&flat, &mut offset, d_model, n_heads, d_ff).extras_from_flat(&flat, &mut offset, true, false, false);
        assert_eq!(offset, flat.len());
        assert_eq!(restored.to_flat(), flat);

        let x_data: Vec<f32> = (0..seq_len * d_model).map(|i| ((i as f32) * 0.53).cos() * 0.5).collect();

        let loss_with = |block: &TransformerBlock, x_data: &[f32]| -> f32 {
            let mut tape = Tape::new();
            let x = tape.leaf(NdArray::new(x_data.to_vec(), vec![seq_len, d_model]));
            let out = block.forward_full(&mut tape, x, 1, false);
            let loss = tape.sum(out.y);
            tape.value(loss).data[0]
        };

        let mut tape = Tape::new();
        let x = tape.leaf(NdArray::new(x_data.clone(), vec![seq_len, d_model]));
        let out = block.forward_full(&mut tape, x, 1, false);
        let loss = tape.sum(out.y);
        tape.backward(loss);
        let x_grad = tape.grad(x).unwrap().clone();
        let q0_w_grad = tape.grad(out.qkv_out.w).unwrap().clone();
        // Head 1's row of the K-norm gamma table.
        let k1 = d_model / n_heads;
        let k1_gamma_grad = tape.grad(out.qk_norm_out.as_ref().unwrap().1.gamma).unwrap().data[k1];

        let eps = 1e-3;
        for i in 0..x_data.len() {
            let mut xp = x_data.clone();
            xp[i] += eps;
            let mut xm = x_data.clone();
            xm[i] -= eps;
            let numerical = (loss_with(&block, &xp) - loss_with(&block, &xm)) / (2.0 * eps);
            assert!(
                (numerical - x_grad.data[i]).abs() < 1e-2,
                "x grad[{i}] mismatch: numerical {numerical} vs analytical {}",
                x_grad.data[i]
            );
        }

        let mut block = block;
        let orig = block.qkv.w.data[0];
        block.qkv.w.data[0] = orig + eps;
        let lp = loss_with(&block, &x_data);
        block.qkv.w.data[0] = orig - eps;
        let lm = loss_with(&block, &x_data);
        block.qkv.w.data[0] = orig;
        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (numerical - q0_w_grad.data[0]).abs() < 1e-2,
            "qkv.w[0] grad mismatch: numerical {numerical} vs analytical {}",
            q0_w_grad.data[0]
        );

        let orig = block.qk_norm.as_ref().unwrap().1.gamma.data[k1];
        block.qk_norm.as_mut().unwrap().1.gamma.data[k1] = orig + eps;
        let lp = loss_with(&block, &x_data);
        block.qk_norm.as_mut().unwrap().1.gamma.data[k1] = orig - eps;
        let lm = loss_with(&block, &x_data);
        block.qk_norm.as_mut().unwrap().1.gamma.data[k1] = orig;
        let numerical = (lp - lm) / (2.0 * eps);
        assert!(
            (numerical - k1_gamma_grad).abs() < 1e-2,
            "k-norm head 1 gamma[0] grad mismatch: numerical {numerical} vs analytical {k1_gamma_grad}"
        );
        assert!(k1_gamma_grad.abs() > 1e-6, "k-norm gamma gets no gradient - scale not wired into scores");
    }

    /// Gate and sink logit are new wiring (a composed sigmoid, a broadcast
    /// scalar subtracted before softmax1): finite-difference check on x, a
    /// gate weight and a sink logit, plus the checkpoint round trip.
    #[test]
    fn transformer_block_gate_and_sink_backward_matches_finite_difference() {
        let mut rng = Rng::new(3);
        let (d_model, n_heads, d_ff, seq_len) = (8, 2, 16, 4);
        let mut gate_rng = Rng::new(4);
        let mut block = TransformerBlock::new(&mut rng, d_model, n_heads, d_ff).with_attn_gate(&mut gate_rng).with_sink_logit();
        block.sink_logits.as_mut().unwrap().data[1] = 0.3;

        let flat = block.to_flat();
        let mut offset = 0;
        let restored = TransformerBlock::from_flat(&flat, &mut offset, d_model, n_heads, d_ff).extras_from_flat(&flat, &mut offset, false, true, true);
        assert_eq!(offset, flat.len());
        assert_eq!(restored.to_flat(), flat);

        let x_data: Vec<f32> = (0..seq_len * d_model).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
        let loss_with = |block: &TransformerBlock, x_data: &[f32]| -> f32 {
            let mut tape = Tape::new();
            let x = tape.leaf(NdArray::new(x_data.to_vec(), vec![seq_len, d_model]));
            let out = block.forward_full(&mut tape, x, 1, false);
            let loss = tape.sum(out.y);
            tape.value(loss).data[0]
        };

        let mut tape = Tape::new();
        let x = tape.leaf(NdArray::new(x_data.clone(), vec![seq_len, d_model]));
        let out = block.forward_full(&mut tape, x, 1, false);
        let loss = tape.sum(out.y);
        tape.backward(loss);
        let x_grad = tape.grad(x).unwrap().clone();
        let gate_w_grad = tape.grad(out.attn_gate_out.as_ref().unwrap().w).unwrap().data[0];
        let sink_grad = tape.grad(out.sink_leaf.unwrap()).unwrap().data[1];

        let eps = 1e-3;
        for i in 0..x_data.len() {
            let (mut xp, mut xm) = (x_data.clone(), x_data.clone());
            xp[i] += eps;
            xm[i] -= eps;
            let numerical = (loss_with(&block, &xp) - loss_with(&block, &xm)) / (2.0 * eps);
            assert!((numerical - x_grad.data[i]).abs() < 1e-2, "x grad[{i}]: numerical {numerical} vs analytical {}", x_grad.data[i]);
        }

        let orig = block.attn_gate.as_ref().unwrap().w.data[0];
        block.attn_gate.as_mut().unwrap().w.data[0] = orig + eps;
        let lp = loss_with(&block, &x_data);
        block.attn_gate.as_mut().unwrap().w.data[0] = orig - eps;
        let lm = loss_with(&block, &x_data);
        block.attn_gate.as_mut().unwrap().w.data[0] = orig;
        let numerical = (lp - lm) / (2.0 * eps);
        assert!((numerical - gate_w_grad).abs() < 1e-2, "gate w grad: numerical {numerical} vs analytical {gate_w_grad}");

        let orig = block.sink_logits.as_ref().unwrap().data[1];
        block.sink_logits.as_mut().unwrap().data[1] = orig + eps;
        let lp = loss_with(&block, &x_data);
        block.sink_logits.as_mut().unwrap().data[1] = orig - eps;
        let lm = loss_with(&block, &x_data);
        block.sink_logits.as_mut().unwrap().data[1] = orig;
        let numerical = (lp - lm) / (2.0 * eps);
        assert!((numerical - sink_grad).abs() < 1e-2, "sink grad: numerical {numerical} vs analytical {sink_grad}");
        assert!(sink_grad.abs() > 1e-6, "sink logit gets no gradient");
    }

    /// forward_batched must compute EXACTLY what running the unbatched
    /// forward separately per sample would - batching only changes how
    /// compute is grouped (Tape::batched_matmul), never what gets computed.
    /// Checked directly here rather than relying on batched_matmul's own
    /// gradient-check test alone, which only verifies (forward,backward)
    /// self-consistency in isolation, not that forward_batched's semantics
    /// match the unbatched reference at the whole-block level.
    #[test]
    fn forward_batched_matches_per_sample_unbatched() {
        let mut rng = Rng::new(3);
        let (d_model, n_heads, d_ff, seq_len) = (8, 2, 16, 4);
        let block = TransformerBlock::new(&mut rng, d_model, n_heads, d_ff);
        let batch_size = 3;

        let x0: Vec<f32> = (0..seq_len * d_model).map(|i| ((i as f32) * 0.31).sin() * 0.5).collect();
        let x1: Vec<f32> = (0..seq_len * d_model).map(|i| ((i as f32) * 0.53 + 1.0).sin() * 0.5).collect();
        let x2: Vec<f32> = (0..seq_len * d_model).map(|i| ((i as f32) * 0.71 + 2.0).sin() * 0.5).collect();
        let stacked: Vec<f32> = x0.iter().chain(x1.iter()).chain(x2.iter()).cloned().collect();

        let mut batched_tape = Tape::new();
        let x_batched = batched_tape.leaf(NdArray::new(stacked, vec![batch_size * seq_len, d_model]));
        let batched_out = block.forward_batched(&mut batched_tape, x_batched, batch_size);
        let batched_y = batched_tape.value(batched_out.y).clone();

        for (i, x_i) in [x0, x1, x2].iter().enumerate() {
            let mut tape = Tape::new();
            let x = tape.leaf(NdArray::new(x_i.clone(), vec![seq_len, d_model]));
            let out = block.forward(&mut tape, x);
            let expected = tape.value(out.y);
            let actual_chunk = &batched_y.data[i * seq_len * d_model..(i + 1) * seq_len * d_model];
            for (a, e) in actual_chunk.iter().zip(expected.data.iter()) {
                assert!((a - e).abs() < 1e-4, "batch chunk {i} mismatch: batched {a} vs unbatched {e}");
            }
        }
    }

    /// Sanity check, not a finite-difference test - Embedding has no new
    /// backward math, it delegates entirely to Gather's already-verified
    /// backward. Checks forward correctness (right rows selected) and that
    /// gradient reaches the table at all, including for a repeated index.
    #[test]
    fn embedding_forward_and_gradient_reach_table() {
        let mut rng = Rng::new(1);
        let emb = Embedding::new(&mut rng, 5, 3);

        let mut tape = Tape::new();
        let out = emb.forward(&mut tape, &[2, 0, 2]);
        let y = tape.value(out.y);
        assert_eq!(y.shape, vec![3, 3]);
        assert_eq!(&y.data[0..3], &emb.table.data[2 * 3..2 * 3 + 3]);
        assert_eq!(&y.data[3..6], &emb.table.data[0..3]);

        let loss = tape.sum(out.y);
        tape.backward(loss);
        let table_grad = tape.grad(out.table).unwrap();
        // Row 2 was looked up twice - its gradient should reflect both uses.
        assert!(table_grad.data[2 * 3] > table_grad.data[0 * 3]);
    }

    /// apply_grad_with::<Sgd> must compute EXACTLY what apply_grad already
    /// does - same discipline as forward_batched's degenerate-case test.
    /// Sgd::State = () means this path costs nothing extra over the
    /// existing Sgd-specific apply_grad; the two must never diverge.
    #[test]
    fn apply_grad_with_sgd_matches_apply_grad() {
        let mut rng = Rng::new(4);
        let mut layer_a = Linear::new(&mut rng, 3, 2);
        let mut layer_b = layer_a.clone();
        let opt = Sgd { lr: 0.1 };

        let x_data = vec![0.5, -0.3, 0.8];
        let mut tape_a = Tape::new();
        let x_a = tape_a.leaf(NdArray::new(x_data.clone(), vec![1, 3]));
        let out_a = layer_a.forward(&mut tape_a, x_a);
        let loss_a = tape_a.sum(out_a.y);
        tape_a.backward(loss_a);
        layer_a.apply_grad(&tape_a, &out_a, &opt);

        let mut tape_b = Tape::new();
        let x_b = tape_b.leaf(NdArray::new(x_data, vec![1, 3]));
        let out_b = layer_b.forward(&mut tape_b, x_b);
        let loss_b = tape_b.sum(out_b.y);
        tape_b.backward(loss_b);
        let mut w_state = <Sgd as Optimizer>::new_state(&layer_b.w.shape);
        let mut b_state = <Sgd as Optimizer>::new_state(&layer_b.b.shape);
        layer_b.apply_grad_with(&tape_b, &out_b, &opt, &mut w_state, &mut b_state);

        assert_eq!(layer_a.w.data, layer_b.w.data);
        assert_eq!(layer_a.b.data, layer_b.b.data);
    }

    /// Batched heads must compute what the per-head block did (2026-09-24).
    /// src/testdata/batched_heads_reference.bin was captured from the
    /// per-head block before the switch, for a plain block and a softmax1
    /// block with QK-norm, gate and sinks (non-trivial norm and sink values,
    /// so a wrong table row shows), batch 2: its output y, its parameter
    /// gradients in to_flat order, and dx, under loss = sum(y * r).
    /// The forward must match bit for bit: every dot product accumulates in
    /// the same order. Gradients to 1e-4 relative: sums over heads and
    /// broadcasts now group differently. Measured worst 2.6e-5 (dx, plain),
    /// f32 reassociation; the design note's 1e-5 target was too tight.
    #[test]
    fn batched_heads_match_per_head_reference() {
        let bytes = include_bytes!("testdata/batched_heads_reference.bin");
        let mut pos = 0;
        let mut next = || -> Vec<f32> {
            let n = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let v = bytes[pos..pos + 4 * n].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            pos += 4 * n;
            v
        };
        let (d, h_n, d_ff, t, b_n) = (16usize, 4usize, 24usize, 5usize, 2usize);
        let d_k = d / h_n;
        for extras in [false, true] {
            let mut rng = Rng::new(11);
            let mut block = TransformerBlock::new(&mut rng, d, h_n, d_ff);
            if extras {
                block = block.with_qk_norm().with_attn_gate(&mut rng).with_sink_logit();
                let (qn, kn) = block.qk_norm.as_mut().unwrap();
                for h in 0..h_n {
                    for j in 0..d_k {
                        qn.gamma.data[h * d_k + j] = 1.0 + 0.03 * ((h * 7 + j * 3) % 9) as f32;
                        qn.beta.data[h * d_k + j] = 0.02 * ((h * 5 + j * 11) % 7) as f32 - 0.06;
                        kn.gamma.data[h * d_k + j] = 1.0 + 0.03 * ((h * 7 + j * 3 + 1) % 9) as f32;
                        kn.beta.data[h * d_k + j] = 0.02 * ((h * 5 + j * 11 + 1) % 7) as f32 - 0.06;
                    }
                    block.sink_logits.as_mut().unwrap().data[h] = 0.3 * h as f32 - 0.4;
                }
            }
            let x_data: Vec<f32> = (0..b_n * t * d).map(|_| rng.next_gaussian()).collect();
            let r_data: Vec<f32> = (0..b_n * t * d).map(|_| rng.next_gaussian()).collect();
            let mut tape = Tape::new();
            let x = tape.leaf(NdArray::new(x_data, vec![b_n * t, d]));
            let out = block.forward_full(&mut tape, x, b_n, extras);
            let r = tape.leaf(NdArray::new(r_data, vec![b_n * t, d]));
            let weighted = tape.mul(out.y, r);
            let loss = tape.sum(weighted);
            tape.backward(loss);

            let mut vars = vec![out.ln1_out.gamma, out.ln1_out.beta, out.qkv_out.w, out.qkv_out.b, out.out_proj_out.w, out.out_proj_out.b];
            vars.extend([out.ln2_out.gamma, out.ln2_out.beta, out.ffn1_out.w, out.ffn1_out.b, out.ffn2_out.w, out.ffn2_out.b]);
            if let Some((qn, kn)) = &out.qk_norm_out {
                vars.extend([qn.gamma, qn.beta, kn.gamma, kn.beta]);
            }
            if let Some(g) = &out.attn_gate_out {
                vars.extend([g.w, g.b]);
            }
            vars.extend(out.sink_leaf);
            let grads: Vec<f32> = vars.iter().flat_map(|&v| tape.grad(v).unwrap().data.clone()).collect();
            assert_eq!(grads.len(), block.to_flat().len(), "gradient layout must follow to_flat");

            let (want_y, want_grads, want_dx) = (next(), next(), next());
            let y = &tape.value(out.y).data;
            assert!(y.iter().zip(&want_y).all(|(a, b)| a.to_bits() == b.to_bits()), "extras={extras}: forward not bit-identical");
            let close = |got: &[f32], want: &[f32], what: &str| {
                assert_eq!(got.len(), want.len(), "{what}: length");
                let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                for (i, (a, b)) in got.iter().zip(want).enumerate() {
                    assert!((a - b).abs() <= 1e-4 * b.abs().max(1e-3 * scale), "extras={extras} {what}[{i}]: {a} vs {b}");
                }
            };
            close(&grads, &want_grads, "param grads");
            close(&tape.grad(x).unwrap().data, &want_dx, "dx");
        }
    }
}
