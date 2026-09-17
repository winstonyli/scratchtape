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
}

/// Output of a Linear layer's forward pass. Carries the leaf `Var`s
/// (not just the result `y`) because the training loop needs them to read
/// gradients back off the tape and step the optimizer - no hidden state,
/// same explicit style as the tape itself.
pub struct LinearOut {
    pub y: Var,
    w: Var,
    b: Var,
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
