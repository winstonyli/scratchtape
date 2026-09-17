use crate::optim::Sgd;
use crate::tape::{Tape, Var};
use crate::tensor::NdArray;

/// xorshift64 PRNG. Hand-rolled, not from a crate - deterministic given a
/// seed, which matters for reproducing a training run exactly.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
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
/// same explicit style as the tape itself. w/b are pub - Adam's training
/// loop needs them directly (bypasses apply_grad, which is Sgd-specific).
pub struct LinearOut {
    pub y: Var,
    pub w: Var,
    pub b: Var,
}

pub struct Linear {
    pub w: NdArray, // [in_dim, out_dim]
    pub b: NdArray, // [out_dim]
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
        }
    }

    pub fn forward(&self, tape: &mut Tape, x: Var) -> LinearOut {
        let w = tape.leaf(self.w.clone());
        let b = tape.leaf(self.b.clone());
        let mm = tape.matmul(x, w);
        let y = tape.add(mm, b);
        LinearOut { y, w, b }
    }

    pub fn apply_grad(&mut self, tape: &Tape, out: &LinearOut, opt: &Sgd) {
        opt.step(&mut self.w, tape.grad(out.w).unwrap());
        opt.step(&mut self.b, tape.grad(out.b).unwrap());
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
pub struct LayerNorm {
    pub gamma: NdArray, // [1, d_model]
    pub beta: NdArray,  // [1, d_model]
    eps: f32,
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
        }
    }

    pub fn forward(&self, tape: &mut Tape, x: Var) -> LayerNormOut {
        let d = tape.value(x).shape[1] as f32;
        let gamma = tape.leaf(self.gamma.clone());
        let beta = tape.leaf(self.beta.clone());

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
        opt.step(&mut self.gamma, tape.grad(out.gamma).unwrap());
        opt.step(&mut self.beta, tape.grad(out.beta).unwrap());
    }
}

/// Strictly upper-triangular -inf (j > i only - the diagonal stays
/// unmasked so a position can always attend to itself, avoiding an
/// all-masked row -> 0/0 -> NaN). Safe with this engine's softmax (which
/// has no max-subtraction stability trick): exp(-inf) is an exact, finite
/// 0.0 under IEEE 754, so -inf never survives past the Exp step - nothing
/// downstream ever sees an infinity.
fn causal_mask(seq_len: usize) -> NdArray {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    NdArray::new(data, vec![seq_len, seq_len])
}

pub struct TransformerBlockOut {
    pub y: Var,
    ln1_out: LayerNormOut,
    q_outs: Vec<LinearOut>,
    k_outs: Vec<LinearOut>,
    v_outs: Vec<LinearOut>,
    out_proj_out: LinearOut,
    ln2_out: LayerNormOut,
    ffn1_out: LinearOut,
    ffn2_out: LinearOut,
}

/// Pre-norm transformer block: x + Attn(LN(x)), then x + FFN(LN(x)).
/// Every sublayer already exists (LayerNorm, Linear, relu, Add, Concat,
/// softmax) - this is composition, not new machinery. Its multi-head
/// attention is NOT multihead_attention_recall.rs's hand-set
/// identity-slice version (that demo was illustrative, deliberately
/// untrained) - here each head gets its own real He-initialized, trainable
/// Linear(d_model, d_k) for Q/K/V, same as every other Linear in this
/// codebase. Kept inlined rather than factored into its own
/// MultiHeadAttention struct - this block is currently its only consumer.
pub struct TransformerBlock {
    n_heads: usize,
    d_k: usize,
    ln1: LayerNorm,
    q_heads: Vec<Linear>,
    k_heads: Vec<Linear>,
    v_heads: Vec<Linear>,
    out_proj: Linear,
    ln2: LayerNorm,
    ffn1: Linear,
    ffn2: Linear,
}

impl TransformerBlock {
    pub fn new(rng: &mut Rng, d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        let d_k = d_model / n_heads;
        Self {
            n_heads,
            d_k,
            ln1: LayerNorm::new(d_model),
            q_heads: (0..n_heads).map(|_| Linear::new(rng, d_model, d_k)).collect(),
            k_heads: (0..n_heads).map(|_| Linear::new(rng, d_model, d_k)).collect(),
            v_heads: (0..n_heads).map(|_| Linear::new(rng, d_model, d_k)).collect(),
            out_proj: Linear::new(rng, d_model, d_model),
            ln2: LayerNorm::new(d_model),
            ffn1: Linear::new(rng, d_model, d_ff),
            ffn2: Linear::new(rng, d_ff, d_model),
        }
    }

    pub fn forward(&self, tape: &mut Tape, x: Var) -> TransformerBlockOut {
        let seq_len = tape.value(x).shape[0];
        let mask = tape.leaf(causal_mask(seq_len));

        let ln1_out = self.ln1.forward(tape, x);
        let normed1 = ln1_out.y;

        let mut q_outs = Vec::with_capacity(self.n_heads);
        let mut k_outs = Vec::with_capacity(self.n_heads);
        let mut v_outs = Vec::with_capacity(self.n_heads);
        let mut head_outputs = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            let q_out = self.q_heads[h].forward(tape, normed1);
            let k_out = self.k_heads[h].forward(tape, normed1);
            let v_out = self.v_heads[h].forward(tape, normed1);

            let kt = tape.transpose(k_out.y);
            let scores = tape.matmul(q_out.y, kt);
            let scaled = tape.scale(scores, 1.0 / (self.d_k as f32).sqrt());
            let masked = tape.add(scaled, mask);
            let weights = tape.softmax(masked);
            head_outputs.push(tape.matmul(weights, v_out.y));

            q_outs.push(q_out);
            k_outs.push(k_out);
            v_outs.push(v_out);
        }

        let concat = tape.concat(&head_outputs);
        let out_proj_out = self.out_proj.forward(tape, concat);
        let x1 = tape.add(x, out_proj_out.y); // residual

        let ln2_out = self.ln2.forward(tape, x1);
        let ffn1_out = self.ffn1.forward(tape, ln2_out.y);
        let hidden = tape.relu(ffn1_out.y);
        let ffn2_out = self.ffn2.forward(tape, hidden);
        let y = tape.add(x1, ffn2_out.y); // residual

        TransformerBlockOut { y, ln1_out, q_outs, k_outs, v_outs, out_proj_out, ln2_out, ffn1_out, ffn2_out }
    }

    pub fn apply_grad(&mut self, tape: &Tape, out: &TransformerBlockOut, opt: &Sgd) {
        self.ln1.apply_grad(tape, &out.ln1_out, opt);
        for h in 0..self.n_heads {
            self.q_heads[h].apply_grad(tape, &out.q_outs[h], opt);
            self.k_heads[h].apply_grad(tape, &out.k_outs[h], opt);
            self.v_heads[h].apply_grad(tape, &out.v_outs[h], opt);
        }
        self.out_proj.apply_grad(tape, &out.out_proj_out, opt);
        self.ln2.apply_grad(tape, &out.ln2_out, opt);
        self.ffn1.apply_grad(tape, &out.ffn1_out, opt);
        self.ffn2.apply_grad(tape, &out.ffn2_out, opt);
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
}
