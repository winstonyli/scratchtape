use crate::tensor::NdArray;

/// Index into the tape's arena. Copy type - cheap to pass around,
/// no lifetime to fight since it doesn't borrow the tape it points into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Var {
    idx: usize,
}

/// Every field used to be usize/f32 (Copy), letting `backward` read an op
/// by value and release the borrow on `nodes[i]` before it needs a second,
/// mutable borrow to accumulate into a parent. Concat is variadic (arity
/// isn't known ahead of time, unlike every binary/unary op above it), so it
/// needs a Vec - which breaks the Copy derive for the whole enum. `backward`
/// now clones the op instead of copying it; same effect (borrow released
/// before mutating), just one Vec clone instead of a bitwise copy.
#[derive(Clone, Debug)]
enum OpKind {
    Leaf,
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    MatMul(usize, usize),
    Relu(usize),
    Sum(usize),
    Scale(usize, f32),
    Transpose(usize),
    Exp(usize),
    SumLastAxis(usize),
    Div(usize, usize),
    Concat(Vec<usize>),
    Gather(usize, Vec<usize>),
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

    /// Distinct from Tape::transpose only used inside MatMul's own backward
    /// (that one operates on raw NdArray values, not a graph node). This is
    /// a first-class differentiable op - needed so Kᵀ in attention's QKᵀ has
    /// its own gradient path back to whatever produced K.
    pub fn transpose(&mut self, a: Var) -> Var {
        let val = self.nodes[a.idx].value.transpose();
        self.push(val, OpKind::Transpose(a.idx))
    }

    pub fn exp(&mut self, a: Var) -> Var {
        let val = self.nodes[a.idx].value.exp();
        self.push(val, OpKind::Exp(a.idx))
    }

    pub fn sum_last_axis(&mut self, a: Var) -> Var {
        let val = self.nodes[a.idx].value.sum_last_axis();
        self.push(val, OpKind::SumLastAxis(a.idx))
    }

    /// Differentiable division - distinct from NdArray::div, which stays
    /// non-differentiable/optimizer-only (Adam's parameter update runs
    /// outside the graph entirely, doesn't need a gradient of its own).
    pub fn div(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.idx].value.div(&self.nodes[b.idx].value);
        self.push(val, OpKind::Div(a.idx, b.idx))
    }

    /// Concatenates along the last axis - e.g. recombining multi-head
    /// attention's per-head outputs before a final output projection.
    pub fn concat(&mut self, vars: &[Var]) -> Var {
        let val = {
            let values: Vec<&NdArray> = vars.iter().map(|v| &self.nodes[v.idx].value).collect();
            NdArray::concat_last_axis(&values)
        };
        let parents: Vec<usize> = vars.iter().map(|v| v.idx).collect();
        self.push(val, OpKind::Concat(parents))
    }

    /// Row-selection by index (e.g. token/positional embedding lookup).
    /// `indices` is plain data, not a Var - nobody differentiates with
    /// respect to which row got looked up, only the table itself is
    /// trainable. Deliberately added to the library rather than kept in an
    /// example first: unlike ES/SSM/PC (genuine single-use explorations),
    /// this has multiple known consumers from the start (token embedding,
    /// positional embedding, cross-entropy's target-selection).
    pub fn gather(&mut self, table: Var, indices: &[usize]) -> Var {
        let val = self.nodes[table.idx].value.gather_rows(indices);
        self.push(val, OpKind::Gather(table.idx, indices.to_vec()))
    }

    /// No max-subtraction stability trick (would need a differentiable
    /// MaxLastAxis op that doesn't exist yet) - fine for small controlled
    /// demo magnitudes, would need addressing before real unnormalized
    /// logits at scale.
    pub fn softmax(&mut self, a: Var) -> Var {
        let e = self.exp(a);
        let s = self.sum_last_axis(e);
        self.div(e, s)
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
            let op = self.nodes[i].op.clone();
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
                OpKind::Transpose(a) => {
                    // Transpose is self-inverse: transposing the incoming
                    // gradient back undoes the forward transpose exactly.
                    self.accumulate(a, grad.transpose());
                }
                OpKind::Exp(a) => {
                    // d(exp(x))/dx = exp(x) - which is this very node's own
                    // forward value (node i IS y=exp(x)), not a's value.
                    let out_val = self.nodes[i].value.clone();
                    self.accumulate(a, grad.mul(&out_val));
                }
                OpKind::SumLastAxis(a) => {
                    // d(sum)/dx_j = 1 for every summed element - grad
                    // (shape [...,1]) broadcasts (copies) back across the
                    // reduced axis via the existing broadcast machinery.
                    let a_shape = self.nodes[a].value.shape.clone();
                    self.accumulate(a, grad.broadcast_to(&a_shape));
                }
                OpKind::Div(a, b) => {
                    let a_val = self.nodes[a].value.clone();
                    let b_val = self.nodes[b].value.clone();
                    // d(a/b)/da = 1/b
                    self.accumulate(a, grad.div(&b_val));
                    // d(a/b)/db = -a/b^2
                    let b_sq = b_val.mul(&b_val);
                    self.accumulate(b, grad.mul(&a_val).div(&b_sq).scale(-1.0));
                }
                OpKind::Concat(parents) => {
                    // Concat doesn't mix values, just places them side by
                    // side - so the incoming gradient just gets sliced back
                    // into the same ranges, no cross-contamination between
                    // parents. Simplest backward rule so far.
                    let mut offset = 0usize;
                    for p in parents {
                        let width = *self.nodes[p].value.shape.last().unwrap();
                        self.accumulate(p, grad.slice_last_axis(offset, width));
                        offset += width;
                    }
                }
                OpKind::Gather(table, indices) => {
                    // Scatter-add, not scatter-overwrite: a repeated index
                    // (the same token looked up twice) must accumulate both
                    // contributions into that table row, not have the
                    // second lookup's gradient replace the first's.
                    let table_rows = self.nodes[table].value.shape[0];
                    self.accumulate(table, NdArray::scatter_add_rows(&indices, &grad, table_rows));
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

    /// Same gold-standard check, exercising the full attention formula:
    /// Transpose, MatMul, Scale, then Exp+SumLastAxis+Div (softmax) chained
    /// together, then a final MatMul. Checks Q and K, which is where the
    /// new ops (transpose into scores, softmax normalization) actually get
    /// exercised - V's gradient is a plain MatMul backward, already covered
    /// by the test above.
    fn attention_loss(q_data: &[f32], k_data: &[f32], v_data: &[f32]) -> f32 {
        let mut tape = Tape::new();
        let q = tape.leaf(NdArray::new(q_data.to_vec(), vec![2, 3]));
        let k = tape.leaf(NdArray::new(k_data.to_vec(), vec![2, 3]));
        let v = tape.leaf(NdArray::new(v_data.to_vec(), vec![2, 3]));
        let kt = tape.transpose(k);
        let scores = tape.matmul(q, kt);
        let scaled = tape.scale(scores, 1.0 / (3.0f32).sqrt());
        let weights = tape.softmax(scaled);
        let out = tape.matmul(weights, v);
        let loss = tape.sum(out);
        tape.value(loss).data[0]
    }

    #[test]
    fn attention_backward_matches_finite_difference() {
        let q0 = vec![0.5, -0.3, 0.2, 0.7, 0.1, -0.4];
        let k0 = vec![0.2, 0.4, -0.1, -0.3, 0.6, 0.2];
        let v0 = vec![1.0, 0.5, -0.5, 0.2, 0.3, -0.1];

        let mut tape = Tape::new();
        let q = tape.leaf(NdArray::new(q0.clone(), vec![2, 3]));
        let k = tape.leaf(NdArray::new(k0.clone(), vec![2, 3]));
        let v = tape.leaf(NdArray::new(v0.clone(), vec![2, 3]));
        let kt = tape.transpose(k);
        let scores = tape.matmul(q, kt);
        let scaled = tape.scale(scores, 1.0 / (3.0f32).sqrt());
        let weights = tape.softmax(scaled);
        let out = tape.matmul(weights, v);
        let loss = tape.sum(out);
        tape.backward(loss);
        let q_grad = tape.grad(q).unwrap().clone();
        let k_grad = tape.grad(k).unwrap().clone();

        let eps = 1e-3;
        for i in 0..q0.len() {
            let mut qp = q0.clone();
            qp[i] += eps;
            let mut qm = q0.clone();
            qm[i] -= eps;
            let numerical = (attention_loss(&qp, &k0, &v0) - attention_loss(&qm, &k0, &v0)) / (2.0 * eps);
            assert!(
                (numerical - q_grad.data[i]).abs() < 1e-2,
                "q grad[{i}] mismatch: numerical {numerical} vs analytical {}",
                q_grad.data[i]
            );
        }
        for i in 0..k0.len() {
            let mut kp = k0.clone();
            kp[i] += eps;
            let mut km = k0.clone();
            km[i] -= eps;
            let numerical = (attention_loss(&q0, &kp, &v0) - attention_loss(&q0, &km, &v0)) / (2.0 * eps);
            assert!(
                (numerical - k_grad.data[i]).abs() < 1e-2,
                "k grad[{i}] mismatch: numerical {numerical} vs analytical {}",
                k_grad.data[i]
            );
        }
    }

    /// Two independent "heads" (own matmul each), concatenated, then a
    /// shared loss - checks Concat's backward correctly routes gradient
    /// back to each head's own input without cross-contamination.
    fn concat_loss(a_data: &[f32], b_data: &[f32], wa: &[f32], wb: &[f32]) -> f32 {
        let mut tape = Tape::new();
        let a = tape.leaf(NdArray::new(a_data.to_vec(), vec![2, 2]));
        let b = tape.leaf(NdArray::new(b_data.to_vec(), vec![2, 2]));
        let wa_v = tape.leaf(NdArray::new(wa.to_vec(), vec![2, 3]));
        let wb_v = tape.leaf(NdArray::new(wb.to_vec(), vec![2, 3]));
        let out_a = tape.matmul(a, wa_v);
        let out_b = tape.matmul(b, wb_v);
        let cat = tape.concat(&[out_a, out_b]);
        let loss = tape.sum(cat);
        tape.value(loss).data[0]
    }

    #[test]
    fn concat_backward_matches_finite_difference() {
        let a0 = vec![0.5, -0.3, 0.2, 0.7];
        let b0 = vec![0.1, -0.2, 0.4, -0.5];
        let wa = vec![0.2, 0.4, -0.1, -0.3, 0.6, 0.2];
        let wb = vec![-0.4, 0.1, 0.3, 0.2, -0.2, 0.5];

        let mut tape = Tape::new();
        let a = tape.leaf(NdArray::new(a0.clone(), vec![2, 2]));
        let b = tape.leaf(NdArray::new(b0.clone(), vec![2, 2]));
        let wa_v = tape.leaf(NdArray::new(wa.clone(), vec![2, 3]));
        let wb_v = tape.leaf(NdArray::new(wb.clone(), vec![2, 3]));
        let out_a = tape.matmul(a, wa_v);
        let out_b = tape.matmul(b, wb_v);
        let cat = tape.concat(&[out_a, out_b]);
        let loss = tape.sum(cat);
        tape.backward(loss);
        let a_grad = tape.grad(a).unwrap().clone();
        let b_grad = tape.grad(b).unwrap().clone();

        let eps = 1e-3;
        for i in 0..a0.len() {
            let mut ap = a0.clone();
            ap[i] += eps;
            let mut am = a0.clone();
            am[i] -= eps;
            let numerical = (concat_loss(&ap, &b0, &wa, &wb) - concat_loss(&am, &b0, &wa, &wb)) / (2.0 * eps);
            assert!(
                (numerical - a_grad.data[i]).abs() < 1e-2,
                "a grad[{i}] mismatch: numerical {numerical} vs analytical {}",
                a_grad.data[i]
            );
        }
        for i in 0..b0.len() {
            let mut bp = b0.clone();
            bp[i] += eps;
            let mut bm = b0.clone();
            bm[i] -= eps;
            let numerical = (concat_loss(&a0, &bp, &wa, &wb) - concat_loss(&a0, &bm, &wa, &wb)) / (2.0 * eps);
            assert!(
                (numerical - b_grad.data[i]).abs() < 1e-2,
                "b grad[{i}] mismatch: numerical {numerical} vs analytical {}",
                b_grad.data[i]
            );
        }
    }

    /// Deliberately uses a REPEATED index (2 looked up twice) - the one
    /// scenario where a scatter-overwrite bug would silently produce a
    /// wrong-but-plausible gradient instead of an obvious crash.
    fn gather_loss(table_data: &[f32], indices: &[usize]) -> f32 {
        let mut tape = Tape::new();
        let table = tape.leaf(NdArray::new(table_data.to_vec(), vec![5, 3]));
        let gathered = tape.gather(table, indices);
        let sq = tape.mul(gathered, gathered);
        let loss = tape.sum(sq);
        tape.value(loss).data[0]
    }

    #[test]
    fn gather_backward_matches_finite_difference_with_repeated_index() {
        let table0: Vec<f32> = vec![
            0.5, -0.3, 0.2, //
            0.1, 0.4, -0.6, //
            -0.2, 0.7, 0.3, //
            0.9, -0.1, 0.5, //
            -0.4, 0.2, 0.8,
        ];
        let indices = [2usize, 0, 2, 4];

        let mut tape = Tape::new();
        let table = tape.leaf(NdArray::new(table0.clone(), vec![5, 3]));
        let gathered = tape.gather(table, &indices);
        let sq = tape.mul(gathered, gathered);
        let loss = tape.sum(sq);
        tape.backward(loss);
        let table_grad = tape.grad(table).unwrap().clone();

        let eps = 1e-3;
        for i in 0..table0.len() {
            let mut tp = table0.clone();
            tp[i] += eps;
            let mut tm = table0.clone();
            tm[i] -= eps;
            let numerical = (gather_loss(&tp, &indices) - gather_loss(&tm, &indices)) / (2.0 * eps);
            assert!(
                (numerical - table_grad.data[i]).abs() < 1e-2,
                "table grad[{i}] mismatch: numerical {numerical} vs analytical {}",
                table_grad.data[i]
            );
        }
        // Row 2 was looked up twice - its gradient must reflect BOTH uses
        // (2x what a single lookup would produce), not just one.
        assert!(
            table_grad.data[6].abs() > 0.1 && table_grad.data[7].abs() > 0.1 && table_grad.data[8].abs() > 0.1,
            "row 2's gradient looks like only one of its two lookups contributed: {:?}",
            &table_grad.data[6..9]
        );
    }
}
