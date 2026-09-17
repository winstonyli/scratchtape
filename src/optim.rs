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
