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
    BatchedMatMul(usize, usize, usize, bool),
    Relu(usize),
    Sum(usize),
    Scale(usize, f32),
    Transpose(usize),
    Exp(usize),
    SumLastAxis(usize),
    MaxLastAxis(usize),
    Div(usize, usize),
    Concat(Vec<usize>),
    Gather(usize, Vec<usize>),
    Sqrt(usize),
    Log(usize),
    /// (a, first column, n_heads, batch_size)
    SplitHeads(usize, usize, usize, usize),
    /// (a, n_heads, batch_size)
    MergeHeads(usize, usize, usize),
}

struct Node {
    value: NdArray,
    grad: Option<NdArray>,
    op: OpKind,
}

/// Extracts one contiguous row-chunk. Row-major layout makes this a plain
/// sub-range copy, no gather needed - simpler than concat_last_axis/
/// slice_last_axis, which operate on the non-contiguous last axis instead.
fn row_chunk(arr: &NdArray, chunk_idx: usize, num_chunks: usize) -> NdArray {
    let total_rows = arr.shape[0];
    let cols = arr.shape[1];
    let rows_per_chunk = total_rows / num_chunks;
    let start = chunk_idx * rows_per_chunk * cols;
    let end = start + rows_per_chunk * cols;
    NdArray::new(arr.data[start..end].to_vec(), vec![rows_per_chunk, cols])
}

/// Batched matmul forward: loops batch_size independent 2D matmuls over
/// row-chunks of `a`/`b`, writing each chunk's result into the right slice
/// of one output buffer - reuses NdArray::matmul/transpose unchanged, no
/// NdArray rank generalization needed (surveyed and rejected - too much
/// ripple through Linear/LayerNorm/softmax/cross_entropy for what a single
/// dedicated op can do instead). transpose_b handles attention's Q@K^T case:
/// each chunk's B operand is transposed individually, before that chunk's
/// matmul - never the whole stacked B, which would wrongly mix batches
/// together (the bug a naive dense-stack-then-block-diagonal-mask approach
/// has: computes a full (batch*seq)x(batch*seq) matrix and masks out the
/// cross-batch entries afterward, wasting O(batch) more compute than this
/// per-chunk loop does).
fn batched_matmul_forward(a: &NdArray, b: &NdArray, batch_size: usize, transpose_b: bool) -> NdArray {
    let a_rows_per_chunk = a.shape[0] / batch_size;
    let out_cols = if transpose_b { b.shape[0] / batch_size } else { b.shape[1] };
    let mut out = vec![0.0f32; a.shape[0] * out_cols];
    for i in 0..batch_size {
        let a_chunk = row_chunk(a, i, batch_size);
        let b_chunk = row_chunk(b, i, batch_size);
        let chunk_out = if transpose_b { a_chunk.matmul(&b_chunk.transpose()) } else { a_chunk.matmul(&b_chunk) };
        let start = i * a_rows_per_chunk * out_cols;
        out[start..start + a_rows_per_chunk * out_cols].copy_from_slice(&chunk_out.data);
    }
    NdArray::new(out, vec![a.shape[0], out_cols])
}

/// Backward for batched_matmul_forward - same per-chunk derivation as plain
/// MatMul's backward (C=A@B: dA=dC@Bᵀ, dB=Aᵀ@dC; C=A@Bᵀ: dA=dC@B, dB=dCᵀ@A),
/// just looped per batch chunk instead of applied once.
fn batched_matmul_backward(a: &NdArray, b: &NdArray, grad: &NdArray, batch_size: usize, transpose_b: bool) -> (NdArray, NdArray) {
    let mut grad_a = vec![0.0f32; a.data.len()];
    let mut grad_b = vec![0.0f32; b.data.len()];
    let a_cols = a.shape[1];
    let b_cols = b.shape[1];
    let a_rows_per_chunk = a.shape[0] / batch_size;
    let b_rows_per_chunk = b.shape[0] / batch_size;
    for i in 0..batch_size {
        let a_chunk = row_chunk(a, i, batch_size);
        let b_chunk = row_chunk(b, i, batch_size);
        let grad_chunk = row_chunk(grad, i, batch_size);
        let (da_chunk, db_chunk) = if transpose_b {
            (grad_chunk.matmul(&b_chunk), grad_chunk.transpose().matmul(&a_chunk))
        } else {
            (grad_chunk.matmul(&b_chunk.transpose()), a_chunk.transpose().matmul(&grad_chunk))
        };
        let a_start = i * a_rows_per_chunk * a_cols;
        grad_a[a_start..a_start + a_rows_per_chunk * a_cols].copy_from_slice(&da_chunk.data);
        let b_start = i * b_rows_per_chunk * b_cols;
        grad_b[b_start..b_start + b_rows_per_chunk * b_cols].copy_from_slice(&db_chunk.data);
    }
    (NdArray::new(grad_a, a.shape.clone()), NdArray::new(grad_b, b.shape.clone()))
}

/// Multi-head attention's layout change, as a copy (llm.c's CUDA path does
/// the same with permute kernels). `a` is [B*T, C] with batch b's rows at
/// b*T..; columns col0 + h*w + j (j < w) belong to head h. The result is
/// [B*H*T, w] with head (b, h)'s rows at (b*H + h)*T.., so batched_matmul
/// over B*H chunks runs every head of every sample in one op.
fn split_heads_forward(a: &NdArray, col0: usize, n_heads: usize, width: usize, batch_size: usize) -> NdArray {
    let (cols, t_len) = (a.shape[1], a.shape[0] / batch_size);
    let mut out = vec![0.0f32; a.shape[0] * n_heads * width];
    for b in 0..batch_size {
        for h in 0..n_heads {
            for t in 0..t_len {
                let src = (b * t_len + t) * cols + col0 + h * width;
                let dst = ((b * n_heads + h) * t_len + t) * width;
                out[dst..dst + width].copy_from_slice(&a.data[src..src + width]);
            }
        }
    }
    NdArray::new(out, vec![a.shape[0] * n_heads, width])
}

/// Inverse of split_heads_forward with col0 = 0: [B*H*T, w] -> [B*T, H*w].
fn merge_heads_forward(a: &NdArray, n_heads: usize, batch_size: usize) -> NdArray {
    let width = a.shape[1];
    let t_len = a.shape[0] / (batch_size * n_heads);
    let cols = n_heads * width;
    let mut out = vec![0.0f32; a.data.len()];
    for b in 0..batch_size {
        for h in 0..n_heads {
            for t in 0..t_len {
                let src = ((b * n_heads + h) * t_len + t) * width;
                let dst = (b * t_len + t) * cols + h * width;
                out[dst..dst + width].copy_from_slice(&a.data[src..src + width]);
            }
        }
    }
    NdArray::new(out, vec![batch_size * t_len, cols])
}

/// Append-only arena. Because an op can only reference Vars that already
/// exist, every parent index is strictly less than its child's - topological
/// order falls out of construction, backward() just walks the arena in reverse.
pub struct Tape {
    nodes: Vec<Node>,
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Tape {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Opt-in for perf-sensitive callers who know roughly how many nodes
    /// their graph will have - avoids Vec reallocation as the arena grows.
    /// Not the default for `new()`: a hardcoded capacity baked in for every
    /// caller regardless of actual graph size would be an unearned guess
    /// (a tiny XOR demo doesn't need the same hint as a full transformer
    /// block), not a genuine optimization.
    pub fn with_capacity(cap: usize) -> Self {
        Self { nodes: Vec::with_capacity(cap) }
    }

    /// Number of nodes currently in the arena - lets a caller measure a
    /// real graph's size to inform a with_capacity hint, rather than guess.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every node as (op name, parent indices, output element count), in
    /// tape order - enough to count what a step would launch on a device
    /// and which elementwise chains could fuse (step_profile's census).
    pub fn ops(&self) -> Vec<(&'static str, Vec<usize>, usize)> {
        self.nodes
            .iter()
            .map(|n| {
                let (name, parents) = match &n.op {
                    OpKind::Leaf => ("Leaf", vec![]),
                    OpKind::Add(a, b) => ("Add", vec![*a, *b]),
                    OpKind::Sub(a, b) => ("Sub", vec![*a, *b]),
                    OpKind::Mul(a, b) => ("Mul", vec![*a, *b]),
                    OpKind::MatMul(a, b) => ("MatMul", vec![*a, *b]),
                    OpKind::BatchedMatMul(a, b, _, _) => ("BatchedMatMul", vec![*a, *b]),
                    OpKind::Relu(a) => ("Relu", vec![*a]),
                    OpKind::Sum(a) => ("Sum", vec![*a]),
                    OpKind::Scale(a, _) => ("Scale", vec![*a]),
                    OpKind::Transpose(a) => ("Transpose", vec![*a]),
                    OpKind::Exp(a) => ("Exp", vec![*a]),
                    OpKind::SumLastAxis(a) => ("SumLastAxis", vec![*a]),
                    OpKind::MaxLastAxis(a) => ("MaxLastAxis", vec![*a]),
                    OpKind::Div(a, b) => ("Div", vec![*a, *b]),
                    OpKind::Concat(v) => ("Concat", v.clone()),
                    OpKind::Gather(a, _) => ("Gather", vec![*a]),
                    OpKind::Sqrt(a) => ("Sqrt", vec![*a]),
                    OpKind::Log(a) => ("Log", vec![*a]),
                    OpKind::SplitHeads(a, ..) => ("SplitHeads", vec![*a]),
                    OpKind::MergeHeads(a, ..) => ("MergeHeads", vec![*a]),
                };
                (name, parents, n.value.data.len())
            })
            .collect()
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

    /// See batched_matmul_forward's own doc comment for what this buys over
    /// plain matmul and the alternatives it was surveyed against.
    pub fn batched_matmul(&mut self, a: Var, b: Var, batch_size: usize, transpose_b: bool) -> Var {
        let val = batched_matmul_forward(&self.nodes[a.idx].value, &self.nodes[b.idx].value, batch_size, transpose_b);
        self.push(val, OpKind::BatchedMatMul(a.idx, b.idx, batch_size, transpose_b))
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

    /// Differentiable max along the last axis - distinct from
    /// NdArray::max_last_axis, which stays non-differentiable/detached-leaf
    /// use (softmax's stability trick, where the gradient through the max
    /// is provably zero by shift-invariance - see that method's doc
    /// comment). Needed for a genuine OR-module (existential search over
    /// candidate substitutions, differentiable_backward_chaining.rs):
    /// there the gradient through which candidate wins must NOT be zero -
    /// that's the whole point of learning through the search. 2D input
    /// only ([rows, cols] -> [rows, 1]), same scope every other reduction
    /// in this codebase is actually used at (matmul, sum_last_axis, etc.
    /// are never called past rank 2 in practice either).
    pub fn max_last_axis(&mut self, a: Var) -> Var {
        let val = self.nodes[a.idx].value.max_last_axis();
        self.push(val, OpKind::MaxLastAxis(a.idx))
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

    /// Differentiable sqrt - distinct from NdArray::sqrt, which stays
    /// non-differentiable/optimizer-only (Adam's parameter update runs
    /// outside the graph). Needed for layer norm's variance normalization.
    pub fn sqrt(&mut self, a: Var) -> Var {
        let val = self.nodes[a.idx].value.sqrt();
        self.push(val, OpKind::Sqrt(a.idx))
    }

    /// Columns `cols` of `a` ([B*T, C]) as `n_heads` heads stacked by
    /// (sample, head): [B*H*T, cols.len()/H]. A permutation, so its backward
    /// is the inverse permutation (zero outside `cols`). See
    /// split_heads_forward for the row order.
    pub fn split_heads(&mut self, a: Var, cols: std::ops::Range<usize>, n_heads: usize, batch_size: usize) -> Var {
        let val = split_heads_forward(&self.nodes[a.idx].value, cols.start, n_heads, cols.len() / n_heads, batch_size);
        self.push(val, OpKind::SplitHeads(a.idx, cols.start, n_heads, batch_size))
    }

    /// Inverse of split_heads: [B*H*T, w] -> [B*T, H*w], head h in columns
    /// h*w..(h+1)*w.
    pub fn merge_heads(&mut self, a: Var, n_heads: usize, batch_size: usize) -> Var {
        let val = merge_heads_forward(&self.nodes[a.idx].value, n_heads, batch_size);
        self.push(val, OpKind::MergeHeads(a.idx, n_heads, batch_size))
    }

    /// Max-subtraction stability trick, forced by concrete evidence (an
    /// unregularized training run's logits grew until exp() overflowed,
    /// producing NaN that corrupted the whole model) rather than built
    /// speculatively. The max is a detached constant (a leaf), not routed
    /// through a differentiable MaxLastAxis op - softmax is shift-invariant
    /// (softmax(x) = softmax(x-c) for any per-row c), so the gradient
    /// contribution that would flow through the max's own dependence on the
    /// input is provably exactly zero. This fixes overflow (exp of a large
    /// positive logit) specifically - it does NOT fix underflow (a
    /// legitimately tiny probability rounding to exact 0.0), which is a
    /// separate failure mode cross_entropy's epsilon guard still handles.
    pub fn softmax(&mut self, a: Var) -> Var {
        let max_val = self.nodes[a.idx].value.max_last_axis();
        let max_leaf = self.leaf(max_val);
        let shifted = self.sub(a, max_leaf);
        let e = self.exp(shifted);
        let s = self.sum_last_axis(e);
        self.div(e, s)
    }

    /// "softmax1" / QuietAttention (Evan Miller, "Attention Is Off By One").
    /// Adds 1 to softmax's denominator - the only change - so weights can
    /// sum to LESS than 1 instead of always exactly 1. Plain softmax forces
    /// a head to dump its full attention budget somewhere even when nothing
    /// is relevant (the mechanistic explanation for the observed "attention
    /// sink" phenomenon); this lets it express "nothing here" by driving
    /// logits very negative, converging toward zero total weight instead of
    /// a forced-uniform distribution. Side benefit: also slightly more
    /// robust than plain softmax against the all-very-negative-logits case
    /// (the +1 floors the denominator at 1, so it can't collapse toward
    /// zero the way plain softmax's denominator can).
    ///
    /// Overflow-safe form: softmax1(x) is exactly softmax over [x, 0] (a
    /// phantom key with logit 0 and zero value), so shift by
    /// m = max(max(x), 0) and use exp(-m) for the phantom term:
    /// exp(x-m) / (sum exp(x-m) + exp(-m)). Detaching m is valid only
    /// because that full [x, 0] softmax is shift-invariant. The version
    /// used from 6f22b36 until this fix shifted by max(x) but kept "+1",
    /// which computes exp(x) / (sum exp(x) + exp(max x)) - a phantom key
    /// at the row's max, not at 0 - so the forward pass was shift-invariant
    /// while the detached-max gradient was not. Parameters that shift a
    /// whole row equally (a key bias) got a steady gradient with no effect
    /// on the loss, and grew without bound (attention_uniformity_check.rs:
    /// K-norm gains ~1e9). Every softmax1 result before the fix used it.
    pub fn softmax1(&mut self, a: Var) -> Var {
        let mut max_val = self.nodes[a.idx].value.max_last_axis();
        for m in max_val.data.iter_mut() {
            *m = m.max(0.0);
        }
        let phantom = NdArray { data: max_val.data.iter().map(|m| (-m).exp()).collect(), shape: max_val.shape.clone() };
        let max_leaf = self.leaf(max_val);
        let shifted = self.sub(a, max_leaf);
        let e = self.exp(shifted);
        let s = self.sum_last_axis(e);
        let phantom_leaf = self.leaf(phantom);
        let denom = self.add(s, phantom_leaf);
        self.div(e, denom)
    }

    /// Differentiable log. Unlike Exp/Sqrt, this needs the PARENT's value,
    /// not its own output - d(log(x))/dx = 1/x uses x itself, not log(x).
    /// Reusing "self.nodes[i].value" out of habit from Exp/Sqrt's backward
    /// would be wrong here.
    pub fn log(&mut self, a: Var) -> Var {
        let val = self.nodes[a.idx].value.log();
        self.push(val, OpKind::Log(a.idx))
    }

    /// Mean cross-entropy loss over `logits` [N, vocab_size] against integer
    /// class labels. Composed entirely from existing ops (softmax, one-hot
    /// times log, sum, scale) - no dedicated "select the true class's
    /// probability" op needed, since one-hot-multiply-then-sum achieves the
    /// same result using machinery that already exists. A small eps guards
    /// log(0) = -inf (a real risk once softmax's output can genuinely
    /// underflow to exact 0.0 over a real vocabulary). `softmax` is already
    /// max-subtraction stable; that only prevents overflow, not underflow.
    pub fn cross_entropy(&mut self, logits: Var, targets: &[usize]) -> Var {
        let vocab_size = self.nodes[logits.idx].value.shape[1];
        let n = targets.len();
        let one_hot = self.leaf(NdArray::one_hot(targets, vocab_size));
        let probs = self.softmax(logits);
        let eps = self.leaf(NdArray::scalar(1e-9));
        let probs_eps = self.add(probs, eps);
        let log_probs = self.log(probs_eps);
        let selected = self.mul(log_probs, one_hot);
        let selected_sum = self.sum_last_axis(selected);
        let total = self.sum(selected_sum);
        self.scale(total, -1.0 / n as f32)
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
                OpKind::BatchedMatMul(a, b, batch_size, transpose_b) => {
                    let a_val = self.nodes[a].value.clone();
                    let b_val = self.nodes[b].value.clone();
                    let (grad_a, grad_b) = batched_matmul_backward(&a_val, &b_val, &grad, batch_size, transpose_b);
                    self.accumulate(a, grad_a);
                    self.accumulate(b, grad_b);
                }
                OpKind::Relu(a) => {
                    let a_val = &self.nodes[a].value;
                    let mask = NdArray { data: a_val.data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect(), shape: a_val.shape.clone() };
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
                OpKind::MaxLastAxis(a) => {
                    // Standard max-pool backward: the whole incoming
                    // gradient routes to whichever column actually WAS the
                    // max in that row, zero everywhere else - only the
                    // winner influenced the output, so only the winner
                    // gets credit/blame. Ties route to the first occurrence
                    // (matches this project's other tie-break-by-first-
                    // match conventions, and real float ties are
                    // vanishingly rare on learned data anyway). Assumes
                    // 2D input, same scope max_last_axis's own doc comment
                    // states.
                    let a_val = self.nodes[a].value.clone();
                    let max_val = self.nodes[i].value.clone();
                    let rows = a_val.shape[0];
                    let cols = a_val.shape[1];
                    let mut grad_a = vec![0.0f32; a_val.data.len()];
                    for row in 0..rows {
                        let target = max_val.data[row];
                        let row_start = row * cols;
                        let winner = (0..cols).find(|&k| a_val.data[row_start + k] == target).unwrap_or(0);
                        grad_a[row_start + winner] = grad.data[row];
                    }
                    self.accumulate(a, NdArray::new(grad_a, a_val.shape.clone()));
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
                OpKind::Sqrt(a) => {
                    // d(sqrt(x))/dx = 1/(2*sqrt(x)) - and sqrt(x) is this
                    // very node's own forward value, same self-referencing
                    // trick as Exp's backward.
                    let out_val = self.nodes[i].value.clone();
                    self.accumulate(a, grad.div(&out_val).scale(0.5));
                }
                OpKind::Log(a) => {
                    // d(log(x))/dx = 1/x - needs the PARENT's value, not
                    // this node's own output, unlike Exp/Sqrt above.
                    let a_val = self.nodes[a].value.clone();
                    self.accumulate(a, grad.div(&a_val));
                }
                OpKind::SplitHeads(a, col0, n_heads, batch_size) => {
                    // Merge the heads back, then place them at their
                    // columns; columns outside the split get zero.
                    let merged = merge_heads_forward(&grad, n_heads, batch_size);
                    let shape = self.nodes[a].value.shape.clone();
                    let (rows, cols, w) = (shape[0], shape[1], merged.shape[1]);
                    let mut g = vec![0.0f32; rows * cols];
                    for r in 0..rows {
                        g[r * cols + col0..][..w].copy_from_slice(&merged.data[r * w..][..w]);
                    }
                    self.accumulate(a, NdArray::new(g, shape));
                }
                OpKind::MergeHeads(a, n_heads, batch_size) => {
                    let w = self.nodes[a].value.shape[1];
                    self.accumulate(a, split_heads_forward(&grad, 0, n_heads, w, batch_size));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gold-standard autograd correctness check: for every element of every
    /// input, the analytical gradient of the scalar loss `build` returns
    /// (from backward()) must match a central difference of it. `build`
    /// runs once per evaluation on fresh leaves, so a graph is written once.
    /// Returns the analytical gradients, for checks beyond the match.
    fn check_grads(inputs: &[(&[f32], &[usize])], build: impl Fn(&mut Tape, &[Var]) -> Var) -> Vec<Vec<f32>> {
        let run = |vals: &[Vec<f32>]| {
            let mut tape = Tape::new();
            let vars: Vec<Var> = inputs.iter().zip(vals).map(|((_, shape), v)| tape.leaf(NdArray::new(v.clone(), shape.to_vec()))).collect();
            let loss = build(&mut tape, &vars);
            (tape, vars, loss)
        };
        let base: Vec<Vec<f32>> = inputs.iter().map(|(data, _)| data.to_vec()).collect();
        let (mut tape, vars, loss) = run(&base);
        tape.backward(loss);
        let grads: Vec<Vec<f32>> = vars.iter().map(|&v| tape.grad(v).expect("no gradient reached an input").data.clone()).collect();
        let eps = 1e-3;
        for (k, g) in grads.iter().enumerate() {
            for (i, analytical) in g.iter().enumerate() {
                let loss_at = |delta: f32| {
                    let mut vals = base.clone();
                    vals[k][i] += delta;
                    let (tape, _, loss) = run(&vals);
                    tape.value(loss).data[0]
                };
                let numerical = (loss_at(eps) - loss_at(-eps)) / (2.0 * eps);
                assert!((numerical - analytical).abs() < 1e-2, "input {k} grad[{i}] mismatch: numerical {numerical} vs analytical {analytical}");
            }
        }
        grads
    }

    #[test]
    #[should_panic(expected = "mismatch")]
    fn check_grads_catches_a_wrong_gradient() {
        // The detached copy of x hides half of d(x * x)/dx from backward.
        check_grads(&[(&[1.0, 2.0], &[2])], |tape, v| {
            let detached = tape.leaf(tape.value(v[0]).clone());
            let sq = tape.mul(v[0], detached);
            tape.sum(sq)
        });
    }

    /// Exercises MatMul + broadcast Add + Mul(x,x) + Sum in one graph.
    #[test]
    fn backward_matches_finite_difference() {
        check_grads(&[(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), (&[0.5, -0.3, 0.2, 0.7], &[2, 2]), (&[0.1, -0.2], &[2])], |tape, v| {
            let mm = tape.matmul(v[0], v[1]);
            let pred = tape.add(mm, v[2]);
            let sq = tape.mul(pred, pred);
            tape.sum(sq)
        });
    }

    /// Both transpose_b variants of Tape::batched_matmul (the plain C=A@B
    /// case used for weights@V, the C=A@Bᵀ case used for Q@Kᵀ).
    #[test]
    fn batched_matmul_backward_matches_finite_difference() {
        for transpose_b in [false, true] {
            let batch_size = 2;
            let (m, k, n) = (2, 3, 2);
            let b_shape = if transpose_b { [batch_size * n, k] } else { [batch_size * k, n] };
            let a0: Vec<f32> = (0..batch_size * m * k).map(|i| 0.1 * (i as f32) - 0.5).collect();
            let b0: Vec<f32> = (0..b_shape[0] * b_shape[1]).map(|i| 0.05 * (i as f32) - 0.3).collect();
            check_grads(&[(&a0, &[batch_size * m, k]), (&b0, &b_shape)], |tape, v| {
                let c = tape.batched_matmul(v[0], v[1], batch_size, transpose_b);
                let sq = tape.mul(c, c);
                tape.sum(sq)
            });
        }
    }

    /// The full attention formula: Transpose, MatMul, Scale, then
    /// Exp+SumLastAxis+Div (softmax) chained together, then a final MatMul.
    /// Q and K are where the new ops (transpose into scores, softmax
    /// normalization) get exercised; V's gradient is a plain MatMul backward.
    #[test]
    fn attention_backward_matches_finite_difference() {
        let q = [0.5, -0.3, 0.2, 0.7, 0.1, -0.4];
        let k = [0.2, 0.4, -0.1, -0.3, 0.6, 0.2];
        let v = [1.0, 0.5, -0.5, 0.2, 0.3, -0.1];
        check_grads(&[(&q, &[2, 3]), (&k, &[2, 3]), (&v, &[2, 3])], |tape, x| {
            let kt = tape.transpose(x[1]);
            let scores = tape.matmul(x[0], kt);
            let scaled = tape.scale(scores, 1.0 / (3.0f32).sqrt());
            let weights = tape.softmax(scaled);
            let out = tape.matmul(weights, x[2]);
            tape.sum(out)
        });
    }

    /// Two independent "heads" (own matmul each), concatenated, then a
    /// shared loss - Concat's backward must route gradient back to each
    /// head's own input without cross-contamination.
    #[test]
    fn concat_backward_matches_finite_difference() {
        let (a, b) = ([0.5, -0.3, 0.2, 0.7], [0.1, -0.2, 0.4, -0.5]);
        let (wa, wb) = ([0.2, 0.4, -0.1, -0.3, 0.6, 0.2], [-0.4, 0.1, 0.3, 0.2, -0.2, 0.5]);
        check_grads(&[(&a, &[2, 2]), (&b, &[2, 2]), (&wa, &[2, 3]), (&wb, &[2, 3])], |tape, v| {
            let out_a = tape.matmul(v[0], v[2]);
            let out_b = tape.matmul(v[1], v[3]);
            let cat = tape.concat(&[out_a, out_b]);
            tape.sum(cat)
        });
    }

    /// Deliberately uses a REPEATED index (2 looked up twice) - the one
    /// scenario where a scatter-overwrite bug would silently produce a
    /// wrong-but-plausible gradient instead of an obvious crash.
    #[test]
    fn gather_backward_matches_finite_difference_with_repeated_index() {
        let table: Vec<f32> = vec![
            0.5, -0.3, 0.2, //
            0.1, 0.4, -0.6, //
            -0.2, 0.7, 0.3, //
            0.9, -0.1, 0.5, //
            -0.4, 0.2, 0.8,
        ];
        let grads = check_grads(&[(&table, &[5, 3])], |tape, v| {
            let gathered = tape.gather(v[0], &[2, 0, 2, 4]);
            let sq = tape.mul(gathered, gathered);
            tape.sum(sq)
        });
        // Row 2 was looked up twice - its gradient must reflect BOTH uses
        // (2x what a single lookup would produce), not just one.
        assert!(grads[0][6..9].iter().all(|g| g.abs() > 0.1), "row 2's gradient looks like only one of its two lookups contributed: {:?}", &grads[0][6..9]);
    }

    /// The full layer-norm-shaped chain: SumLastAxis -> Scale -> Sub -> Mul
    /// -> SumLastAxis -> Scale -> Add(eps) -> Sqrt -> Div -> Mul(gamma) ->
    /// Add(beta). Checks x, gamma and beta - Sqrt is the only genuinely new
    /// op here, but this confirms the rest compose correctly.
    #[test]
    fn layer_norm_backward_matches_finite_difference() {
        let x = [0.5, -0.3, 1.2, 2.0, -1.0, 0.4];
        check_grads(&[(&x, &[2, 3]), (&[1.2, 0.8, 1.5], &[1, 3]), (&[0.1, -0.2, 0.05], &[1, 3])], |tape, v| {
            let d = 3.0f32;
            let sum = tape.sum_last_axis(v[0]);
            let mean = tape.scale(sum, 1.0 / d);
            let centered = tape.sub(v[0], mean);
            let sq = tape.mul(centered, centered);
            let sum_sq = tape.sum_last_axis(sq);
            let variance = tape.scale(sum_sq, 1.0 / d);
            let eps_leaf = tape.leaf(NdArray::scalar(1e-5));
            let variance_eps = tape.add(variance, eps_leaf);
            let std_dev = tape.sqrt(variance_eps);
            let normalized = tape.div(centered, std_dev);
            let scaled = tape.mul(normalized, v[1]);
            let y = tape.add(scaled, v[2]);
            tape.sum(y)
        });
    }

    /// dL/d(logits) - cross_entropy composes softmax (covered elsewhere)
    /// with two new pieces (Log, one-hot-then-sum-last-axis selection), so
    /// this checks that Log's "needs the parent's value, not its own
    /// output" backward rule is implemented correctly, not just documented.
    #[test]
    fn cross_entropy_backward_matches_finite_difference() {
        let logits = [
            0.5, -0.3, 1.2, 0.1, //
            -0.2, 0.8, 0.1, 0.4, //
            1.0, 0.2, -0.5, 0.3,
        ];
        check_grads(&[(&logits, &[3, 4])], |tape, v| tape.cross_entropy(v[0], &[2, 1, 0]));
    }

    /// The differentiable MaxLastAxis. Row 0's max is at index 2 (1.2),
    /// row 1's at index 0 (2.0) - both non-edge positions, so this also
    /// confirms non-argmax entries get exactly zero gradient, not just that
    /// the argmax entry gets a nonzero one.
    #[test]
    fn max_last_axis_backward_matches_finite_difference() {
        check_grads(&[(&[0.5, -0.3, 1.2, 2.0, 0.1, -1.5], &[2, 3])], |tape, v| {
            let m = tape.max_last_axis(v[0]);
            let sq = tape.mul(m, m);
            tape.sum(sq)
        });
    }

    /// softmax1 had no direct check, and its max-subtraction silently
    /// changed the function (see its doc comment). Values must equal
    /// exp(x)/(1+sum exp(x)) - including rows with positive logits, where
    /// the old form diverged from it - stay finite at large logits, honor
    /// -inf masking, and the gradient must match finite differences.
    #[test]
    fn softmax1_matches_definition_and_finite_difference() {
        let x0 = vec![2.0f32, -1.0, 0.5, f32::NEG_INFINITY, 3.0, 1.0, -0.5, 0.2];
        let c = [0.7f32, -1.3, 0.4, 2.0, -0.6, 1.1, 0.9, -0.2];
        let run = |x: &[f32]| -> (f32, Vec<f32>) {
            let mut tape = Tape::new();
            let xv = tape.leaf(NdArray::new(x.to_vec(), vec![2, 4]));
            let w = tape.softmax1(xv);
            let weights = tape.value(w).data.clone();
            let cv = tape.leaf(NdArray::new(c.to_vec(), vec![2, 4]));
            let prod = tape.mul(w, cv);
            let loss = tape.sum(prod);
            (tape.value(loss).data[0], weights)
        };

        let (_, weights) = run(&x0);
        for row in 0..2 {
            let r = &x0[row * 4..row * 4 + 4];
            let denom = 1.0 + r.iter().map(|v| v.exp()).sum::<f32>();
            for j in 0..4 {
                let expected = r[j].exp() / denom;
                assert!((weights[row * 4 + j] - expected).abs() < 1e-6, "weight [{row},{j}]: {} vs {expected}", weights[row * 4 + j]);
            }
        }

        let (_, big) = run(&[80.0, 79.0, 0.0, f32::NEG_INFINITY, 80.0, 80.0, 80.0, 80.0]);
        assert!(big.iter().all(|w| w.is_finite()), "overflow at large logits: {big:?}");

        let mut tape = Tape::new();
        let xv = tape.leaf(NdArray::new(x0.clone(), vec![2, 4]));
        let w = tape.softmax1(xv);
        let cv = tape.leaf(NdArray::new(c.to_vec(), vec![2, 4]));
        let prod = tape.mul(w, cv);
        let loss = tape.sum(prod);
        tape.backward(loss);
        let grad = tape.grad(xv).unwrap().clone();
        let eps = 1e-3;
        for i in (0..x0.len()).filter(|&i| x0[i].is_finite()) {
            let (mut xp, mut xm) = (x0.clone(), x0.clone());
            xp[i] += eps;
            xm[i] -= eps;
            let numerical = (run(&xp).0 - run(&xm).0) / (2.0 * eps);
            assert!((numerical - grad.data[i]).abs() < 1e-2, "grad[{i}]: numerical {numerical} vs analytical {}", grad.data[i]);
        }
    }

    /// split_heads puts element (b*T + t, col0 + h*w + j) at
    /// ((b*H + h)*T + t, j); merge_heads undoes it. Both are permutations,
    /// so each backward must route every gradient element back to where its
    /// value came from, with zero for columns the split didn't take.
    #[test]
    fn split_and_merge_heads_permute_and_backprop_exactly() {
        let (b_n, t_n, h_n, w, cols, col0) = (2, 3, 2, 2, 7, 1);
        let x: Vec<f32> = (0..b_n * t_n * cols).map(|i| i as f32).collect();
        let mut tape = Tape::new();
        let xv = tape.leaf(NdArray::new(x.clone(), vec![b_n * t_n, cols]));
        let split = tape.split_heads(xv, col0..col0 + h_n * w, h_n, b_n);
        assert_eq!(tape.value(split).shape, vec![b_n * h_n * t_n, w]);
        for b in 0..b_n {
            for h in 0..h_n {
                for t in 0..t_n {
                    for j in 0..w {
                        let got = tape.value(split).data[((b * h_n + h) * t_n + t) * w + j];
                        assert_eq!(got, x[(b * t_n + t) * cols + col0 + h * w + j], "b{b} h{h} t{t} j{j}");
                    }
                }
            }
        }
        let merged = tape.merge_heads(split, h_n, b_n);
        let expect: Vec<f32> = (0..b_n * t_n).flat_map(|r| x[r * cols + col0..][..h_n * w].to_vec()).collect();
        assert_eq!(tape.value(merged).data, expect);

        // loss = sum(split * c): d loss / d x = c un-permuted.
        let c: Vec<f32> = (0..b_n * h_n * t_n * w).map(|i| 0.5 + i as f32).collect();
        let cv = tape.leaf(NdArray::new(c.clone(), vec![b_n * h_n * t_n, w]));
        let prod = tape.mul(split, cv);
        let loss = tape.sum(prod);
        tape.backward(loss);
        let g = &tape.grad(xv).unwrap().data;
        for b in 0..b_n {
            for t in 0..t_n {
                for col in 0..cols {
                    let want = if (col0..col0 + h_n * w).contains(&col) {
                        let (h, j) = ((col - col0) / w, (col - col0) % w);
                        c[((b * h_n + h) * t_n + t) * w + j]
                    } else {
                        0.0
                    };
                    assert_eq!(g[(b * t_n + t) * cols + col], want, "b{b} t{t} col{col}");
                }
            }
        }

        // merge_heads' backward: loss = sum(merge(y) * c2) routes c2 back.
        let mut tape = Tape::new();
        let yv = tape.leaf(NdArray::new(c.clone(), vec![b_n * h_n * t_n, w]));
        let m = tape.merge_heads(yv, h_n, b_n);
        let c2 = tape.leaf(NdArray::new(x[..b_n * t_n * h_n * w].to_vec(), vec![b_n * t_n, h_n * w]));
        let prod = tape.mul(m, c2);
        let loss = tape.sum(prod);
        tape.backward(loss);
        let back = split_heads_forward(tape.value(c2), 0, h_n, w, b_n);
        assert_eq!(tape.grad(yv).unwrap().data, back.data);
    }
}
