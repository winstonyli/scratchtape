use crate::tensor::NdArray;

/// Plain gradient descent. Params live outside the tape as ordinary
/// NdArrays - the tape is rebuilt fresh each step, params persist across steps.
pub struct Sgd {
    pub lr: f32,
}

impl Sgd {
    pub fn step(&self, param: &mut NdArray, grad: &NdArray) {
        for (p, g) in param.data.iter_mut().zip(grad.data.iter()) {
            *p -= self.lr * g;
        }
    }
}

/// Per-parameter persistent state Adam needs (SGD needs none). Lives
/// explicitly outside both the tape and the layer that owns the param -
/// same explicit, no-hidden-state pattern as the tape's own Var handles.
/// One AdamState per parameter tensor, created once, passed to `step` every
/// training step.
pub struct AdamState {
    m: NdArray,
    v: NdArray,
    t: usize,
}

impl AdamState {
    pub fn zeros_like(param: &NdArray) -> Self {
        Self { m: NdArray::zeros(param.shape.clone()), v: NdArray::zeros(param.shape.clone()), t: 0 }
    }
}

/// No `Optimizer` trait unifying this with `Sgd` - nothing yet needs them to
/// be runtime-interchangeable, and forcing an associated-state trait now
/// would be guessing at a shape from one example. Swap at the call site
/// (which optimizer's `step` the training loop calls), not via polymorphism.
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Adam {
    pub fn step(&self, param: &mut NdArray, grad: &NdArray, state: &mut AdamState) {
        state.t += 1;
        // m and v are exact NdArrays (not scalars) - every op below is
        // elementwise/broadcast, same primitives the tensor core already has.
        state.m = state.m.scale(self.beta1).add(&grad.scale(1.0 - self.beta1));
        state.v = state.v.scale(self.beta2).add(&grad.mul(grad).scale(1.0 - self.beta2));

        let bias_correct1 = 1.0 - self.beta1.powi(state.t as i32);
        let bias_correct2 = 1.0 - self.beta2.powi(state.t as i32);
        let m_hat = state.m.scale(1.0 / bias_correct1);
        let v_hat = state.v.scale(1.0 / bias_correct2);

        let denom = v_hat.sqrt().add(&NdArray::scalar(self.eps));
        let update = m_hat.div(&denom).scale(self.lr);
        param.data.iter_mut().zip(update.data.iter()).for_each(|(p, u)| *p -= u);
    }
}
