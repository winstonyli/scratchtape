#[derive(Clone, Debug, PartialEq)]
pub struct NdArray {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

fn unravel_index(lin: usize, shape: &[usize]) -> Vec<usize> {
    let nd = shape.len();
    let mut idx = vec![0usize; nd];
    let mut rem = lin;
    for i in 0..nd {
        let stride: usize = shape[i + 1..].iter().product();
        idx[i] = rem / stride;
        rem %= stride;
    }
    idx
}

fn ravel_index(idx: &[usize], shape: &[usize]) -> usize {
    let nd = shape.len();
    let mut lin = 0usize;
    for i in 0..nd {
        let stride: usize = shape[i + 1..].iter().product();
        lin += idx[i] * stride;
    }
    lin
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
        for lin in 0..total {
            let idx = unravel_index(lin, &self.shape);
            let mut out_idx = idx;
            out_idx[nd - 1] = 0;
            let out_lin = ravel_index(&out_idx, &out_shape);
            out[out_lin] += self.data[lin];
        }
        Self { data: out, shape: out_shape }
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

    /// 2D matmul only. Broadcasting deliberately unsupported here -
    /// batched matmul is a separate future decision (einsum-style vs explicit batch loop).
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.shape.len(), 2, "matmul lhs must be 2D, got {:?}", self.shape);
        assert_eq!(other.shape.len(), 2, "matmul rhs must be 2D, got {:?}", other.shape);
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "matmul inner dims must match: {:?} vs {:?}", self.shape, other.shape);
        let mut out = vec![0.0f32; m * n];
        // i-k-j loop order: inner loop is contiguous in both `other` and `out` (row-major),
        // sequential memory access instead of strided - cache-friendly without needing SIMD/BLAS yet.
        for i in 0..m {
            for p in 0..k {
                let a_ip = self.data[i * k + p];
                for j in 0..n {
                    out[i * n + j] += a_ip * other.data[p * n + j];
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
        for lin in 0..total {
            let t_idx = unravel_index(lin, target);
            let s_idx: Vec<usize> = t_idx
                .iter()
                .enumerate()
                .map(|(i, &v)| if padded_shape[i] == 1 { 0 } else { v })
                .collect();
            let src_lin = ravel_index(&s_idx[pad..], &self.shape);
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
        for lin in 0..total_self {
            let s_idx = unravel_index(lin, &self.shape);
            let d_idx: Vec<usize> = s_idx
                .iter()
                .enumerate()
                .map(|(i, &v)| if padded_target[i] == 1 { 0 } else { v })
                .collect();
            let dst_lin = ravel_index(&d_idx[pad..], target);
            out[dst_lin] += self.data[lin];
        }
        Self { data: out, shape: target.to_vec() }
    }

    pub fn add(&self, other: &Self) -> Self {
        let shape = Self::broadcast_shape(&self.shape, &other.shape);
        let a = self.broadcast_to(&shape);
        let b = other.broadcast_to(&shape);
        Self { data: a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect(), shape }
    }

    pub fn sub(&self, other: &Self) -> Self {
        let shape = Self::broadcast_shape(&self.shape, &other.shape);
        let a = self.broadcast_to(&shape);
        let b = other.broadcast_to(&shape);
        Self { data: a.data.iter().zip(b.data.iter()).map(|(x, y)| x - y).collect(), shape }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let shape = Self::broadcast_shape(&self.shape, &other.shape);
        let a = self.broadcast_to(&shape);
        let b = other.broadcast_to(&shape);
        Self { data: a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect(), shape }
    }

    /// Broadcasting, same as add/sub/mul - reuses the same machinery rather
    /// than being the one sibling op that behaves differently. +/-inf on
    /// division by zero (IEEE 754 passthrough, no guard): Adam's `+ eps`
    /// upstream is what prevents an exact zero denominator, not this op.
    pub fn div(&self, other: &Self) -> Self {
        let shape = Self::broadcast_shape(&self.shape, &other.shape);
        let a = self.broadcast_to(&shape);
        let b = other.broadcast_to(&shape);
        Self { data: a.data.iter().zip(b.data.iter()).map(|(x, y)| x / y).collect(), shape }
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
}
