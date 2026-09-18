#[derive(Clone, Debug, PartialEq)]
pub struct NdArray {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// 2x headroom over anything ever actually used in this codebase - every
/// shape here is rank 1 or 2 (matches the established "stay 2D" discipline).
/// Fixed-size stack array, not Vec<usize> or a smallvec-style crate:
/// unravel_index/ravel_index were found (via a real profile, not a guess)
/// to be called once PER ELEMENT inside broadcast_to/reduce_to_shape/
/// sum_last_axis/max_last_axis/concat_last_axis/slice_last_axis - a [16,32]
/// tensor going through broadcast_to allocated 512 separate heap Vecs just
/// for index bookkeeping. Only the first `nd` slots of the returned array
/// are meaningful; callers must slice to `nd`, not assume the whole
/// capacity is valid data.
const MAX_RANK: usize = 4;

/// Row-major strides for `shape`, computed once per call site (profiling
/// found unravel_index/ravel_index recomputing this per ELEMENT via a fresh
/// `shape[i+1..].iter().product()` scan - O(rank) redone on every one of a
/// tensor's `total` elements, at 5 call sites, 36% of self-time combined).
/// Only the first `shape.len()` slots are meaningful, same convention as
/// unravel_index's returned array.
fn strides_for(shape: &[usize]) -> [usize; MAX_RANK] {
    let nd = shape.len();
    assert!(nd <= MAX_RANK, "shape rank {nd} exceeds MAX_RANK {MAX_RANK}");
    let mut strides = [0usize; MAX_RANK];
    let mut acc = 1usize;
    for i in (0..nd).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

fn unravel_index(lin: usize, strides: &[usize]) -> [usize; MAX_RANK] {
    let nd = strides.len();
    let mut idx = [0usize; MAX_RANK];
    let mut rem = lin;
    for i in 0..nd {
        idx[i] = rem / strides[i];
        rem %= strides[i];
    }
    idx
}

fn ravel_index(idx: &[usize], strides: &[usize]) -> usize {
    idx.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum()
}

impl NdArray {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data len {} does not match shape {:?}",
            data.len(),
            shape
        );
        Self { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self { data: vec![0.0; n], shape }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self { data: vec![1.0; n], shape }
    }

    pub fn scalar(v: f32) -> Self {
        Self { data: vec![v], shape: vec![1] }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn relu(&self) -> Self {
        Self {
            data: self.data.iter().map(|&x| x.max(0.0)).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn sum(&self) -> Self {
        Self::scalar(self.data.iter().sum())
    }

    pub fn exp(&self) -> Self {
        Self {
            data: self.data.iter().map(|&x| x.exp()).collect(),
            shape: self.shape.clone(),
        }
    }

    /// Sums along the last axis only, keeping it as size 1 (not dropped) -
    /// so the result broadcasts back against the un-reduced tensor with the
    /// existing broadcast_to machinery, no axis-insertion logic needed.
    pub fn sum_last_axis(&self) -> Self {
        let nd = self.shape.len();
        assert!(nd >= 1, "sum_last_axis needs at least 1 dim");
        let mut out_shape = self.shape.clone();
        out_shape[nd - 1] = 1;
        let mut out = vec![0.0f32; out_shape.iter().product()];
        let total: usize = self.shape.iter().product();
        let in_strides = strides_for(&self.shape);
        let out_strides = strides_for(&out_shape);
        for lin in 0..total {
            let idx = unravel_index(lin, &in_strides[..nd]);
            let mut out_idx = idx;
            out_idx[nd - 1] = 0;
            let out_lin = ravel_index(&out_idx[..nd], &out_strides[..nd]);
            out[out_lin] += self.data[lin];
        }
        Self { data: out, shape: out_shape }
    }

    /// Max along the last axis only, keeping it as size 1 (mirrors
    /// sum_last_axis exactly - same structure, tracks max instead of sum).
    /// Deliberately non-differentiable, not a Tape op: softmax is
    /// shift-invariant (softmax(x) = softmax(x - c) for any per-row
    /// constant c), so the gradient contribution that would flow back
    /// through this term's own dependence on the input is provably exactly
    /// zero - not an approximation. Treating it as a detached constant
    /// (a leaf, in Tape::softmax) gives the exact correct gradient with no
    /// argmax-routing backward rule needed at all.
    pub fn max_last_axis(&self) -> Self {
        let nd = self.shape.len();
        assert!(nd >= 1, "max_last_axis needs at least 1 dim");
        let mut out_shape = self.shape.clone();
        out_shape[nd - 1] = 1;
        let mut out = vec![f32::NEG_INFINITY; out_shape.iter().product()];
        let total: usize = self.shape.iter().product();
        let in_strides = strides_for(&self.shape);
        let out_strides = strides_for(&out_shape);
        for lin in 0..total {
            let idx = unravel_index(lin, &in_strides[..nd]);
            let mut out_idx = idx;
            out_idx[nd - 1] = 0;
            let out_lin = ravel_index(&out_idx[..nd], &out_strides[..nd]);
            out[out_lin] = out[out_lin].max(self.data[lin]);
        }
        Self { data: out, shape: out_shape }
    }

    /// Concatenates along the last axis only - all other dims must match.
    /// Variadic (unlike add/mul/etc), which is why the corresponding tape op
    /// can't keep OpKind::Copy (see tape.rs).
    pub fn concat_last_axis(arrays: &[&NdArray]) -> Self {
        assert!(!arrays.is_empty(), "concat_last_axis needs at least one array");
        let nd = arrays[0].shape.len();
        let mut out_shape = arrays[0].shape.clone();
        let mut total_last = 0usize;
        for a in arrays {
            assert_eq!(a.shape.len(), nd, "concat_last_axis: rank mismatch");
            for d in 0..nd - 1 {
                assert_eq!(
                    a.shape[d], arrays[0].shape[d],
                    "concat_last_axis: shapes must match on all but the last axis"
                );
            }
            total_last += a.shape[nd - 1];
        }
        out_shape[nd - 1] = total_last;
        let mut out = vec![0.0f32; out_shape.iter().product()];
        let out_strides = strides_for(&out_shape);
        let mut offset = 0usize;
        for a in arrays {
            let total: usize = a.shape.iter().product();
            let a_strides = strides_for(&a.shape);
            for lin in 0..total {
                let mut idx = unravel_index(lin, &a_strides[..nd]);
                idx[nd - 1] += offset;
                let out_lin = ravel_index(&idx[..nd], &out_strides[..nd]);
                out[out_lin] = a.data[lin];
            }
            offset += a.shape[nd - 1];
        }
        Self { data: out, shape: out_shape }
    }

    /// Inverse of concat_last_axis for one piece - extracts a [start,
    /// start+width) range along the last axis. Used by Concat's backward to
    /// route each gradient slice back to the parent that produced it.
    pub fn slice_last_axis(&self, start: usize, width: usize) -> Self {
        let nd = self.shape.len();
        let mut out_shape = self.shape.clone();
        out_shape[nd - 1] = width;
        let total_out: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; total_out];
        let out_strides = strides_for(&out_shape);
        let self_strides = strides_for(&self.shape);
        for lin in 0..total_out {
            let mut idx = unravel_index(lin, &out_strides[..nd]);
            idx[nd - 1] += start;
            let src_lin = ravel_index(&idx[..nd], &self_strides[..nd]);
            out[lin] = self.data[src_lin];
        }
        Self { data: out, shape: out_shape }
    }

    /// Row-selection by index - e.g. looking up embedding vectors by token
    /// id. `self` is the table (2D: [vocab_size, d_model]); each entry in
    /// `indices` selects one row, producing output [indices.len(), d_model].
    pub fn gather_rows(&self, indices: &[usize]) -> Self {
        assert_eq!(self.shape.len(), 2, "gather_rows expects a 2D table, got {:?}", self.shape);
        let d = self.shape[1];
        let mut out = Vec::with_capacity(indices.len() * d);
        for &idx in indices {
            assert!(idx < self.shape[0], "gather_rows: index {idx} out of bounds for {} rows", self.shape[0]);
            out.extend_from_slice(&self.data[idx * d..idx * d + d]);
        }
        Self { data: out, shape: vec![indices.len(), d] }
    }

    /// Inverse of gather_rows for backward: routes each row of `updates`
    /// back to the table row it came from, ADDING rather than overwriting -
    /// the same index can appear more than once in a lookup (a repeated
    /// token), and both uses must accumulate into that row's gradient.
    pub fn scatter_add_rows(indices: &[usize], updates: &NdArray, table_rows: usize) -> Self {
        let d = updates.shape[1];
        let mut out = vec![0.0f32; table_rows * d];
        for (i, &idx) in indices.iter().enumerate() {
            for c in 0..d {
                out[idx * d + c] += updates.data[i * d + c];
            }
        }
        Self { data: out, shape: vec![table_rows, d] }
    }

    /// Elementwise sqrt. NaN on negative input (IEEE 754 passthrough, no
    /// guard) - not reachable via Adam's use (v is an EMA of squares, always
    /// >= 0, no cancellation possible), and adding a defensive check here
    /// would just be validating an invariant that already holds upstream.
    pub fn sqrt(&self) -> Self {
        Self {
            data: self.data.iter().map(|&x| x.sqrt()).collect(),
            shape: self.shape.clone(),
        }
    }

    pub fn scale(&self, c: f32) -> Self {
        Self {
            data: self.data.iter().map(|&x| x * c).collect(),
            shape: self.shape.clone(),
        }
    }

    /// Elementwise natural log. -inf at exactly 0, NaN below 0 (IEEE 754
    /// passthrough) - cross-entropy's caller adds a small eps before this
    /// to keep inputs away from exactly 0 (cheap guard, parked: a properly
    /// max-subtraction-stable softmax would prevent needing it at all).
    pub fn log(&self) -> Self {
        Self {
            data: self.data.iter().map(|&x| x.ln()).collect(),
            shape: self.shape.clone(),
        }
    }

    /// One-hot encoding of class indices - plain constant data, not a Tape
    /// op (never differentiated with respect to which class is "true").
    pub fn one_hot(indices: &[usize], num_classes: usize) -> Self {
        let mut data = vec![0.0f32; indices.len() * num_classes];
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < num_classes, "one_hot: index {idx} out of bounds for {num_classes} classes");
            data[i * num_classes + idx] = 1.0;
        }
        Self { data, shape: vec![indices.len(), num_classes] }
    }

    /// 2D matmul only. Broadcasting deliberately unsupported here -
    /// batched matmul is a separate future decision (einsum-style vs explicit batch loop).
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.shape.len(), 2, "matmul lhs must be 2D, got {:?}", self.shape);
        assert_eq!(other.shape.len(), 2, "matmul rhs must be 2D, got {:?}", other.shape);
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "matmul inner dims must match: {:?} vs {:?}", self.shape, other.shape);
        let mut out = vec![0.0f32; m * n];
        // Cache-blocked on i/p only, j left full-width: blocking j too showed
        // a large regression at n=256 in one bench run (possibly confounded
        // by concurrent machine load, not confirmed in isolation) - leaving j
        // unblocked measured consistent improvement across all sizes with no
        // such downside, so that's the version kept.
        const BLOCK: usize = 64;
        for i0 in (0..m).step_by(BLOCK) {
            let i_max = (i0 + BLOCK).min(m);
            for p0 in (0..k).step_by(BLOCK) {
                let p_max = (p0 + BLOCK).min(k);
                for i in i0..i_max {
                    for p in p0..p_max {
                        let a_ip = self.data[i * k + p];
                        for j in 0..n {
                            out[i * n + j] += a_ip * other.data[p * n + j];
                        }
                    }
                }
            }
        }
        Self { data: out, shape: vec![m, n] }
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "transpose only supports 2D, got {:?}", self.shape);
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = self.data[i * n + j];
            }
        }
        Self { data: out, shape: vec![n, m] }
    }

    fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
        let nd = a.len().max(b.len());
        let mut out = vec![1usize; nd];
        for i in 0..nd {
            let ai = if i < nd - a.len() { 1 } else { a[i - (nd - a.len())] };
            let bi = if i < nd - b.len() { 1 } else { b[i - (nd - b.len())] };
            assert!(ai == bi || ai == 1 || bi == 1, "shapes not broadcastable: {:?} vs {:?}", a, b);
            out[i] = ai.max(bi);
        }
        out
    }

    /// Numpy-style broadcast: pad shape with leading 1s, then any size-1 dim
    /// reads the same source index for every position along that dim.
    pub fn broadcast_to(&self, target: &[usize]) -> Self {
        if self.shape == target {
            return self.clone();
        }
        let nd = target.len();
        let pad = nd - self.shape.len();
        let mut padded_shape = vec![1usize; pad];
        padded_shape.extend_from_slice(&self.shape);

        let total: usize = target.iter().product();
        let mut out = vec![0.0f32; total];
        let target_strides = strides_for(target);
        let self_strides = strides_for(&self.shape);
        for lin in 0..total {
            let t_idx = unravel_index(lin, &target_strides[..nd]);
            let mut s_idx = [0usize; MAX_RANK];
            for i in 0..nd {
                s_idx[i] = if padded_shape[i] == 1 { 0 } else { t_idx[i] };
            }
            let src_lin = ravel_index(&s_idx[pad..nd], &self_strides[..nd - pad]);
            out[lin] = self.data[src_lin];
        }
        Self { data: out, shape: target.to_vec() }
    }

    /// Inverse of broadcast_to: sums a gradient back down to a smaller shape
    /// by accumulating every broadcasted position into its single source slot.
    pub fn reduce_to_shape(&self, target: &[usize]) -> Self {
        if self.shape == target {
            return self.clone();
        }
        let nd = self.shape.len();
        let pad = nd - target.len();
        let mut padded_target = vec![1usize; pad];
        padded_target.extend_from_slice(target);

        let mut out = vec![0.0f32; target.iter().product()];
        let total_self: usize = self.shape.iter().product();
        let self_strides = strides_for(&self.shape);
        let target_strides = strides_for(target);
        for lin in 0..total_self {
            let s_idx = unravel_index(lin, &self_strides[..nd]);
            let mut d_idx = [0usize; MAX_RANK];
            for i in 0..nd {
                d_idx[i] = if padded_target[i] == 1 { 0 } else { s_idx[i] };
            }
            let dst_lin = ravel_index(&d_idx[pad..nd], &target_strides[..nd - pad]);
            out[dst_lin] += self.data[lin];
        }
        Self { data: out, shape: target.to_vec() }
    }

    /// Shared elementwise-with-broadcast implementation for add/sub/mul/div.
    /// Fast path when shapes already match exactly (the common case - most
    /// tape ops combine same-shape tensors, broadcasting is the exception
    /// e.g. a bias vector against a batch): skips broadcast_to entirely,
    /// which otherwise clones BOTH operands (its own fast path still calls
    /// .clone()) before the actual output allocation - 3 allocations where
    /// 1 suffices. Pure performance path: produces identical values either
    /// way, verified by the existing gradient-check tests passing unchanged.
    fn elementwise(&self, other: &Self, f: impl Fn(f32, f32) -> f32) -> Self {
        if self.shape == other.shape {
            let data = self.data.iter().zip(other.data.iter()).map(|(&x, &y)| f(x, y)).collect();
            return Self { data, shape: self.shape.clone() };
        }
        let shape = Self::broadcast_shape(&self.shape, &other.shape);
        let a = self.broadcast_to(&shape);
        let b = other.broadcast_to(&shape);
        let data = a.data.iter().zip(b.data.iter()).map(|(&x, &y)| f(x, y)).collect();
        Self { data, shape }
    }

    pub fn add(&self, other: &Self) -> Self {
        self.elementwise(other, |x, y| x + y)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.elementwise(other, |x, y| x - y)
    }

    pub fn mul(&self, other: &Self) -> Self {
        self.elementwise(other, |x, y| x * y)
    }

    /// +/-inf on division by zero (IEEE 754 passthrough, no guard): Adam's
    /// `+ eps` upstream is what prevents an exact zero denominator, not
    /// this op.
    pub fn div(&self, other: &Self) -> Self {
        self.elementwise(other, |x, y| x / y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_is_elementwise() {
        let a = NdArray::new(vec![4.0, 9.0, 0.0], vec![3]);
        assert_eq!(a.sqrt().data, vec![2.0, 3.0, 0.0]);
    }

    #[test]
    fn div_broadcasts_like_mul() {
        let a = NdArray::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]);
        let b = NdArray::new(vec![2.0], vec![1]);
        assert_eq!(a.div(&b).data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn div_by_zero_is_inf_not_panic() {
        let a = NdArray::new(vec![1.0], vec![1]);
        let b = NdArray::new(vec![0.0], vec![1]);
        assert!(a.div(&b).data[0].is_infinite());
    }

    #[test]
    fn concat_then_slice_round_trips() {
        let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = NdArray::new(vec![5.0, 6.0], vec![2, 1]);
        let cat = NdArray::concat_last_axis(&[&a, &b]);
        assert_eq!(cat.shape, vec![2, 3]);
        assert_eq!(cat.data, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
        assert_eq!(cat.slice_last_axis(0, 2).data, a.data);
        assert_eq!(cat.slice_last_axis(2, 1).data, b.data);
    }

    #[test]
    fn gather_then_scatter_add_accumulates_repeated_rows() {
        let table = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]); // rows: [1,2] [3,4] [5,6]
        let gathered = table.gather_rows(&[1, 0, 1]);
        assert_eq!(gathered.shape, vec![3, 2]);
        assert_eq!(gathered.data, vec![3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);

        let updates = NdArray::new(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![3, 2]);
        let scattered = NdArray::scatter_add_rows(&[1, 0, 1], &updates, 3);
        // row 1 received two contributions (indices 0 and 2 both target it), row 0 one, row 2 none.
        assert_eq!(scattered.data, vec![1.0, 1.0, 2.0, 2.0, 0.0, 0.0]);
    }
}
