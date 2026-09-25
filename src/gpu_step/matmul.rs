//! The one matmul every coarse op uses (milestone 2): Linear and QKV
//! forward and backward, AttnScores, AttnOut. Tiled f32 from the spike
//! (`spikes/cubecl_spike`, `k_matmul_tiled`), generalized:
//! - either operand transposed in storage (NN, NT, TN), by comptime flag;
//! - any m, n, k (guarded loads and stores, zero padding);
//! - batched over the cube grid's z, each operand with its own per-batch
//!   stride, and an element offset into its buffer (parameters live in
//!   one flat buffer);
//! - an epilogue: + bias[col], ReLU, + residual, and accumulate into out.
use super::{buf, client};
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::sync::OnceLock;

/// A matrix inside a device buffer: element offset, per-batch stride (0 to
/// reuse one matrix for every batch), and whether it's stored transposed
/// (a logical [r, c] operand stored as [c, r]).
#[derive(Clone, Copy)]
pub struct MatRef<'a> {
    pub h: &'a Handle,
    pub off: usize,
    pub stride: usize,
    pub trans: bool,
}

impl<'a> MatRef<'a> {
    /// The whole buffer from 0, one batch, not transposed.
    pub fn new(h: &'a Handle) -> Self {
        Self { h, off: 0, stride: 0, trans: false }
    }
}

/// What to do with each output element after the dot product, in this
/// order: + bias[col], ReLU, + residual (same layout as out), + the value
/// already in out.
#[derive(Clone, Copy, Default)]
pub struct Epilogue<'a> {
    pub bias: Option<(&'a Handle, usize)>,
    pub relu: bool,
    pub residual: Option<(&'a Handle, usize)>,
    pub accumulate: bool,
}

/// out[z] (+)= epilogue(a[z] @ b[z]) for z in 0..batch, with a[z] logically
/// [m, k] and b[z] [k, n]; out is row-major [m, n] at `out.off +
/// z * out.stride` (out.trans must be false). One launch.
#[allow(clippy::too_many_arguments)]
pub fn matmul(a: MatRef, b: MatRef, out: MatRef, batch: usize, m: usize, k: usize, n: usize, epi: Epilogue) {
    assert!(!out.trans, "out is stored row-major");
    let dummy = dummy();
    let (bias_h, bias_off) = epi.bias.unwrap_or((dummy, 0));
    let (res_h, res_off) = epi.residual.unwrap_or((dummy, 0));
    let count = CubeCount::Static((n as u32).div_ceil(64), (m as u32).div_ceil(64), batch as u32);
    let u = |x: usize| x as u32;
    k_matmul::launch(
        client(),
        count,
        CubeDim::new_2d(16, 16),
        whole(a.h),
        whole(b.h),
        whole(out.h),
        whole(bias_h),
        whole(res_h),
        u(m),
        u(n),
        u(k),
        u(a.off),
        u(a.stride),
        u(b.off),
        u(b.stride),
        u(out.off),
        u(out.stride),
        u(bias_off),
        u(res_off),
        a.trans,
        b.trans,
        epi.bias.is_some(),
        epi.relu,
        epi.residual.is_some(),
        epi.accumulate,
    );
}

/// A whole buffer as a kernel argument.
fn whole(h: &Handle) -> BufferArg {
    buf(h, h.size_in_used() as usize / 4)
}

/// Bound in place of an absent bias or residual: wgpu rejects one buffer
/// bound as both read-only and read-write in the same dispatch, so `out`
/// can't stand in.
fn dummy() -> &'static Handle {
    static DUMMY: OnceLock<Handle> = OnceLock::new();
    DUMMY.get_or_init(|| client().create_from_slice(f32::as_bytes(&[0.0f32])))
}

/// Each 16x16 cube computes a 64x64 output tile, each unit a 4x4 block in
/// registers, with A and B staged through shared memory in 64x16 / 16x64
/// slabs. Loads of transposed operands walk the stored rows, so adjacent
/// units still read adjacent addresses. The comptime flags stay in their
/// own `if`s so each compiles away rather than mixing into a runtime test.
#[allow(clippy::collapsible_if)]
#[cube(launch)]
fn k_matmul(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    bias: &[f32],
    res: &[f32],
    m: u32,
    n: u32,
    k: u32,
    a_off: u32,
    a_stride: u32,
    b_off: u32,
    b_stride: u32,
    o_off: u32,
    o_stride: u32,
    bias_off: u32,
    res_off: u32,
    #[comptime] trans_a: bool,
    #[comptime] trans_b: bool,
    #[comptime] has_bias: bool,
    #[comptime] relu: bool,
    #[comptime] has_res: bool,
    #[comptime] accumulate: bool,
) {
    let tx = UNIT_POS_X;
    let ty = UNIT_POS_Y;
    let tid = ty * 16 + tx;
    let z = CUBE_POS_Z;
    let row0 = CUBE_POS_Y * 64;
    let col0 = CUBE_POS_X * 64;
    let a0 = a_off + z * a_stride;
    let b0 = b_off + z * b_stride;
    let mut a_s = Shared::<[f32]>::new_slice(1024usize);
    let mut b_s = Shared::<[f32]>::new_slice(1024usize);
    let mut acc = Array::<f32>::new(16usize);
    let mut bv = Array::<f32>::new(4usize);
    #[unroll]
    for i in 0..16u32 {
        acc[i as usize] = 0.0;
    }
    for kk in 0..k.div_ceil(16) {
        #[unroll]
        for i in 0..4u32 {
            let e = tid + 256 * i;
            // A slab [64 rows, 16 k].
            let mut r = e / 16;
            let mut c = e % 16;
            if trans_a {
                r = e % 64;
                c = e / 64;
            }
            let gm = row0 + r;
            let gk = kk * 16 + c;
            let mut v = 0.0f32;
            if gm < m && gk < k {
                let mut idx = gm * k + gk;
                if trans_a {
                    idx = gk * m + gm;
                }
                v = a[(a0 + idx) as usize];
            }
            a_s[(r * 16 + c) as usize] = v;
            // B slab [16 k, 64 cols].
            let mut br = e / 64;
            let mut bc = e % 64;
            if trans_b {
                br = e % 16;
                bc = e / 16;
            }
            let gk2 = kk * 16 + br;
            let gn = col0 + bc;
            let mut w = 0.0f32;
            if gk2 < k && gn < n {
                let mut idx = gk2 * n + gn;
                if trans_b {
                    idx = gn * k + gk2;
                }
                w = b[(b0 + idx) as usize];
            }
            b_s[(br * 64 + bc) as usize] = w;
        }
        sync_cube();
        #[unroll]
        for p in 0..16u32 {
            #[unroll]
            for j in 0..4u32 {
                bv[j as usize] = b_s[(p * 64 + tx * 4 + j) as usize];
            }
            #[unroll]
            for i in 0..4u32 {
                let av = a_s[((ty * 4 + i) * 16 + p) as usize];
                #[unroll]
                for j in 0..4u32 {
                    acc[(i * 4 + j) as usize] += av * bv[j as usize];
                }
            }
        }
        sync_cube();
    }
    #[unroll]
    for i in 0..4u32 {
        #[unroll]
        for j in 0..4u32 {
            let gm = row0 + ty * 4 + i;
            let gn = col0 + tx * 4 + j;
            if gm < m && gn < n {
                let mut v = acc[(i * 4 + j) as usize];
                if has_bias {
                    v += bias[(bias_off + gn) as usize];
                }
                if relu {
                    if v < 0.0 {
                        v = 0.0;
                    }
                }
                let idx = z * o_stride + gm * n + gn;
                if has_res {
                    v += res[(res_off + idx) as usize];
                }
                let o = (o_off + idx) as usize;
                if accumulate {
                    v += out[o];
                }
                out[o] = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Rng;

    fn upload(v: &[f32]) -> Handle {
        client().create_from_slice(f32::as_bytes(v))
    }

    fn read(h: &Handle) -> Vec<f32> {
        f32::from_bytes(&client().read_one(h.clone()).unwrap()).to_vec()
    }

    /// Plain-loop reference for one case, same argument meaning as `matmul`.
    #[allow(clippy::too_many_arguments)]
    fn reference(a: &[f32], ar: (usize, usize, bool), b: &[f32], br: (usize, usize, bool), out: &mut [f32], or: (usize, usize), batch: usize, m: usize, k: usize, n: usize, bias: Option<&[f32]>, relu: bool, res: Option<&[f32]>, accumulate: bool) {
        for z in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for p in 0..k {
                        let x = a[ar.0 + z * ar.1 + if ar.2 { p * m + i } else { i * k + p }];
                        let y = b[br.0 + z * br.1 + if br.2 { j * k + p } else { p * n + j }];
                        s += x * y;
                    }
                    if let Some(bias) = bias {
                        s += bias[j];
                    }
                    if relu {
                        s = s.max(0.0);
                    }
                    let idx = z * or.1 + i * n + j;
                    if let Some(res) = res {
                        s += res[idx];
                    }
                    let o = or.0 + idx;
                    if accumulate {
                        s += out[o];
                    }
                    out[o] = s;
                }
            }
        }
    }

    /// Needs the discrete GPU. Every transpose combination, ragged shapes
    /// (not multiples of the tile), batch strides and buffer offsets, each
    /// epilogue, and the step's own shapes, against a plain-loop reference.
    /// Dot products sum in a different order, so compare to 1e-5 of the
    /// output scale.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn matmul_matches_reference() {
        let mut rng = Rng::new(5);
        let mut gauss = |len: usize| -> Vec<f32> { (0..len).map(|_| rng.next_gaussian()).collect() };
        // (batch, m, k, n, trans_a, trans_b, bias, relu, residual, accumulate)
        let cases = [
            (1, 70, 35, 20, false, false, false, false, false, false),
            (1, 70, 35, 20, true, false, false, false, false, false),
            (1, 70, 35, 20, false, true, false, false, false, false),
            (1, 70, 35, 20, true, true, false, false, false, false),
            (3, 33, 17, 65, false, true, true, true, false, false),
            (3, 64, 64, 64, true, false, false, false, true, true),
            (1, 512, 128, 384, false, false, true, false, false, false), // QKV forward
            (1, 128, 512, 384, true, false, false, false, false, true),  // dW (TN), accumulated
            (1, 512, 256, 128, false, true, false, false, false, false), // dX (NT)
            (64, 64, 16, 64, false, true, false, false, false, false),  // AttnScores
            (64, 64, 64, 16, false, false, false, false, false, false), // AttnOut
            (1, 512, 128, 256, false, false, true, true, false, false), // FFN1
            (1, 512, 256, 128, false, false, true, false, true, false), // FFN2 + residual
        ];
        for &(batch, m, k, n, ta, tb, has_bias, relu, has_res, acc) in &cases {
            // Offsets and strides that aren't multiples of anything.
            let (a_off, b_off, o_off, bias_off, r_off) = (3, 5, 7, 2, 1);
            let (sa, sb, so) = (m * k + 1, if batch > 1 { k * n + 2 } else { 0 }, m * n + 3);
            let a = gauss(a_off + batch * sa);
            let b = gauss(b_off + batch.max(1) * sb.max(k * n));
            let bias = gauss(bias_off + n);
            let res = gauss(r_off + batch * so);
            let out0 = gauss(o_off + batch * so);
            let mut want = out0.clone();
            reference(&a, (a_off, sa, ta), &b, (b_off, sb, tb), &mut want, (o_off, so), batch, m, k, n, has_bias.then_some(&bias[bias_off..]), relu, has_res.then_some(&res[r_off..]), acc);

            let (ah, bh, bias_h, res_h, oh) = (upload(&a), upload(&b), upload(&bias), upload(&res), upload(&out0));
            let epi = Epilogue { bias: has_bias.then_some((&bias_h, bias_off)), relu, residual: has_res.then_some((&res_h, r_off)), accumulate: acc };
            matmul(MatRef { h: &ah, off: a_off, stride: sa, trans: ta }, MatRef { h: &bh, off: b_off, stride: sb, trans: tb }, MatRef { h: &oh, off: o_off, stride: so, trans: false }, batch, m, k, n, epi);
            let got = read(&oh);
            let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs()));
            let err = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
            let case = format!("batch {batch} m {m} k {k} n {n} ta {ta} tb {tb} bias {has_bias} relu {relu} res {has_res} acc {acc}");
            assert_eq!(got.len(), want.len(), "{case}");
            // Also covers the padding between batches and before the offset.
            assert!(err <= 1e-5 * scale, "{case}: max err {err} (scale {scale})");
        }
    }
}
