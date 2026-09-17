use crate::tensor::NdArray;

/// Index into the tape's arena. Copy type - cheap to pass around,
/// no lifetime to fight since it doesn't borrow the tape it points into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var {
    idx: usize,
}

/// All fields are Copy (usize/f32) so OpKind itself derives Copy -
/// lets `backward` read an op by value and release the borrow on `nodes[i]`
/// before it needs a second, mutable borrow to accumulate into a parent.
#[derive(Clone, Copy, Debug)]
enum OpKind {
    Leaf,
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    MatMul(usize, usize),
    Relu(usize),
    Sum(usize),
    Scale(usize, f32),
}

struct Node {
    value: NdArray,
    grad: Option<NdArray>,
    op: OpKind,
}

/// Append-only arena. Because an op can only reference Vars that already
/// exist, every parent index is strictly less than its child's - topological
/// order falls out of construction, backward() just walks the arena in reverse.
pub struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn leaf(&mut self, value: NdArray) -> Var {
        self.nodes.push(Node { value, grad: None, op: OpKind::Leaf });
        Var { idx: self.nodes.len() - 1 }
    }

    fn push(&mut self, value: NdArray, op: OpKind) -> Var {
        self.nodes.push(Node { value, grad: None, op });
        Var { idx: self.nodes.len() - 1 }
    }

    pub fn value(&self, v: Var) -> &NdArray {
        &self.nodes[v.idx].value
    }

    pub fn grad(&self, v: Var) -> Option<&NdArray> {
        self.nodes[v.idx].grad.as_ref()
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.idx].value.add(&self.nodes[b.idx].value);
        self.push(val, OpKind::Add(a.idx, b.idx))
    }

    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.idx].value.sub(&self.nodes[b.idx].value);
        self.push(val, OpKind::Sub(a.idx, b.idx))
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.idx].value.mul(&self.nodes[b.idx].value);
        self.push(val, OpKind::Mul(a.idx, b.idx))
    }

    pub fn matmul(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.idx].value.matmul(&self.nodes[b.idx].value);
        self.push(val, OpKind::MatMul(a.idx, b.idx))
    }

    pub fn relu(&mut self, a: Var) -> Var {
        let val = self.nodes[a.idx].value.relu();
        self.push(val, OpKind::Relu(a.idx))
    }

    pub fn sum(&mut self, a: Var) -> Var {
        let val = self.nodes[a.idx].value.sum();
        self.push(val, OpKind::Sum(a.idx))
    }

    pub fn scale(&mut self, a: Var, c: f32) -> Var {
        let val = self.nodes[a.idx].value.scale(c);
        self.push(val, OpKind::Scale(a.idx, c))
    }

    /// A node's value may feed multiple downstream ops, so incoming gradient
    /// contributions must sum (multivariable chain rule), never overwrite.
    fn accumulate(&mut self, idx: usize, g: NdArray) {
        let shape = &self.nodes[idx].value.shape;
        let g = if &g.shape == shape { g } else { g.reduce_to_shape(shape) };
        match &mut self.nodes[idx].grad {
            Some(existing) => {
                for (e, gv) in existing.data.iter_mut().zip(g.data.iter()) {
                    *e += gv;
                }
            }
            None => self.nodes[idx].grad = Some(g),
        }
    }

    pub fn backward(&mut self, loss: Var) {
        let ones = NdArray::ones(self.nodes[loss.idx].value.shape.clone());
        self.nodes[loss.idx].grad = Some(ones);

        for i in (0..=loss.idx).rev() {
            let grad = match &self.nodes[i].grad {
                Some(g) => g.clone(),
                None => continue, // node not on path from loss - no gradient to propagate
            };
            let op = self.nodes[i].op;
            match op {
                OpKind::Leaf => {}
                OpKind::Add(a, b) => {
                    self.accumulate(a, grad.clone());
                    self.accumulate(b, grad);
                }
                OpKind::Sub(a, b) => {
                    self.accumulate(a, grad.clone());
                    self.accumulate(b, grad.scale(-1.0));
                }
                OpKind::Mul(a, b) => {
                    let a_val = self.nodes[a].value.clone();
                    let b_val = self.nodes[b].value.clone();
                    self.accumulate(a, grad.mul(&b_val));
                    self.accumulate(b, grad.mul(&a_val));
                }
                OpKind::MatMul(a, b) => {
                    let a_val = self.nodes[a].value.clone();
                    let b_val = self.nodes[b].value.clone();
                    self.accumulate(a, grad.matmul(&b_val.transpose()));
                    self.accumulate(b, a_val.transpose().matmul(&grad));
                }
                OpKind::Relu(a) => {
                    let a_val = &self.nodes[a].value;
                    let mask = NdArray {
                        data: a_val.data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect(),
                        shape: a_val.shape.clone(),
                    };
                    self.accumulate(a, grad.mul(&mask));
                }
                OpKind::Sum(a) => {
                    let a_shape = self.nodes[a].value.shape.clone();
                    self.accumulate(a, grad.broadcast_to(&a_shape));
                }
                OpKind::Scale(a, c) => {
                    self.accumulate(a, grad.scale(c));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forward_loss(x: &NdArray, w_data: &[f32], b_data: &[f32], w_shape: Vec<usize>, b_shape: Vec<usize>) -> f32 {
        let mut tape = Tape::new();
        let xv = tape.leaf(x.clone());
        let wv = tape.leaf(NdArray::new(w_data.to_vec(), w_shape));
        let bv = tape.leaf(NdArray::new(b_data.to_vec(), b_shape));
        let mm = tape.matmul(xv, wv);
        let pred = tape.add(mm, bv);
        let sq = tape.mul(pred, pred);
        let loss = tape.sum(sq);
        tape.value(loss).data[0]
    }

    /// Gold-standard autograd correctness check: analytical gradient (backward())
    /// must match central-difference numerical gradient within tolerance.
    /// Exercises MatMul + broadcast Add + Mul(x,x) + Sum in one graph.
    #[test]
    fn backward_matches_finite_difference() {
        let x = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let w0 = vec![0.5, -0.3, 0.2, 0.7];
        let b0 = vec![0.1, -0.2];

        let mut tape = Tape::new();
        let xv = tape.leaf(x.clone());
        let wv = tape.leaf(NdArray::new(w0.clone(), vec![2, 2]));
        let bv = tape.leaf(NdArray::new(b0.clone(), vec![2]));
        let mm = tape.matmul(xv, wv);
        let pred = tape.add(mm, bv);
        let sq = tape.mul(pred, pred);
        let loss = tape.sum(sq);
        tape.backward(loss);
        let w_grad = tape.grad(wv).unwrap().clone();
        let b_grad = tape.grad(bv).unwrap().clone();

        let eps = 1e-3;
        for i in 0..w0.len() {
            let mut wp = w0.clone();
            wp[i] += eps;
            let mut wm = w0.clone();
            wm[i] -= eps;
            let numerical = (forward_loss(&x, &wp, &b0, vec![2, 2], vec![2])
                - forward_loss(&x, &wm, &b0, vec![2, 2], vec![2]))
                / (2.0 * eps);
            assert!(
                (numerical - w_grad.data[i]).abs() < 1e-2,
                "w grad[{i}] mismatch: numerical {numerical} vs analytical {}",
                w_grad.data[i]
            );
        }
        for i in 0..b0.len() {
            let mut bp = b0.clone();
            bp[i] += eps;
            let mut bm = b0.clone();
            bm[i] -= eps;
            let numerical = (forward_loss(&x, &w0, &bp, vec![2, 2], vec![2])
                - forward_loss(&x, &w0, &bm, vec![2, 2], vec![2]))
                / (2.0 * eps);
            assert!(
                (numerical - b_grad.data[i]).abs() < 1e-2,
                "b grad[{i}] mismatch: numerical {numerical} vs analytical {}",
                b_grad.data[i]
            );
        }
    }
}
