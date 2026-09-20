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

/// Unifies Sgd and Adam behind one generic `step`, so a layer's apply_grad
/// can be written once and work with either. Not built speculatively -
/// Adam now has 5 real consumers (xor_mlp_adam.rs, ssm_recall.rs,
/// predictive_coding.rs, gnn_byte_classification.rs, tiny_lm_corpus.rs),
/// each hand-rolling the same per-parameter adam.step loop because every
/// layer's apply_grad was hardcoded to `&Sgd`. `State` stays an explicit
/// associated type rather than a `Default`-bounded one - Adam's state is
/// shape-dependent (needs the param's own shape to zero-init m/v), which a
/// bare `Default` can't express; `new_state` takes the shape directly
/// instead. State keeps living outside the layer/tape (same explicit,
/// no-hidden-state principle AdamState's own doc comment already states),
/// so callers still own and thread it themselves - this trait only removes
/// the duplication in how `step` gets invoked, not where state lives.
pub trait Optimizer {
    type State;
    fn new_state(shape: &[usize]) -> Self::State;
    fn step(&self, param: &mut NdArray, grad: &NdArray, state: &mut Self::State);
}

impl Optimizer for Sgd {
    type State = ();
    fn new_state(_shape: &[usize]) -> Self::State {}
    fn step(&self, param: &mut NdArray, grad: &NdArray, _state: &mut Self::State) {
        Sgd::step(self, param, grad);
    }
}

impl Optimizer for Adam {
    type State = AdamState;
    fn new_state(shape: &[usize]) -> Self::State {
        AdamState { m: NdArray::zeros(shape.to_vec()), v: NdArray::zeros(shape.to_vec()), t: 0 }
    }
    fn step(&self, param: &mut NdArray, grad: &NdArray, state: &mut Self::State) {
        Adam::step(self, param, grad, state);
    }
}
