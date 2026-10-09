//! The one matmul every coarse op uses (milestone 2): Linear and QKV
//! forward and backward, AttnScores, AttnOut. Tiled f32 from the spike
//! (`spikes/cubecl_spike`, `k_matmul_tiled`), generalized:
//! - either operand transposed in storage (NN, NT, TN), by comptime flag;
//! - any m, n, k (guarded loads and stores, zero padding);
//! - batched over the cube grid's z, each operand with its own per-batch
//!   stride, and an element offset into its buffer (parameters live in
//!   one flat buffer);
//! - operands and output addressed in place as attention heads: a row
//!   stride, and matrix z at (z / group)·stride + (z % group)·inner, so
//!   Q, K, V are read straight out of the fused QKV buffer and the
//!   attention output lands in the merged [B·T, D] layout (no split or
//!   merge launches);
//! - an epilogue: + bias[col], ReLU, the ReLU backward mask, + residual,
//!   and accumulate into out;
//! - split-k for shapes with too few output tiles to fill the GPU (the
//!   weight gradients, Xᵀ·dY with k = the batch's rows): slices of k run
//!   in parallel into a scratch buffer, then one launch sums them in a
//!   fixed order, so results stay deterministic.
use super::{buf, client};
use cubecl::prelude::*;
use half::f16;
use cubecl::server::Handle;
use std::sync::OnceLock;

/// A matrix inside a device buffer: element offset, per-batch stride (0 to
/// reuse one matrix for every batch), and whether it's stored transposed
/// (a logical [r, c] operand stored as [c, r]). Matrix z starts at
/// off + (z / group)·stride + (z % group)·inner, and its stored rows are
/// `ld` apart (0: packed, the stored row length).
#[derive(Clone, Copy)]
pub struct MatRef<'a> {
    pub h: &'a Handle,
    pub off: usize,
    pub stride: usize,
    pub trans: bool,
    pub ld: usize,
    pub group: usize,
    pub inner: usize,
}

impl<'a> MatRef<'a> {
    /// The whole buffer from 0, one batch, not transposed.
    pub fn new(h: &'a Handle) -> Self {
        Self { h, off: 0, stride: 0, trans: false, ld: 0, group: 1, inner: 0 }
    }

    /// Attention heads in place: matrix z = b·heads + h is rows b·t ..
    /// b·t + t, columns col0 + h·width .. + width, of a row-major
    /// [B·t, cols] buffer.
    pub fn heads(h: &'a Handle, col0: usize, cols: usize, t: usize, heads: usize, width: usize) -> Self {
        Self { off: col0, stride: t * cols, ld: cols, group: heads, inner: width, ..Self::new(h) }
    }

    /// (group, inner, ld) as kernel arguments; `packed` is the stored row
    /// length when ld is 0.
    fn layout(&self, packed: usize) -> (u32, u32, u32) {
        (self.group as u32, self.inner as u32, if self.ld == 0 { packed } else { self.ld } as u32)
    }
}

/// What to do with each output element after the dot product, in this
/// order: + bias[col] (matrix z's bias `bias_stride`·z further on), ReLU, zero it where `mask` (same layout as out) is
/// not positive (ReLU's backward, given ReLU's output), + residual (same
/// layout as out), + the value already in out.
#[derive(Clone, Copy, Default)]
pub struct Epilogue<'a> {
    pub bias: Option<(&'a Handle, usize)>,
    pub bias_stride: usize,
    pub relu: bool,
    pub mask: Option<(&'a Handle, usize)>,
    pub residual: Option<(&'a Handle, usize)>,
    pub accumulate: bool,
}

/// out[z] (+)= epilogue(a[z] @ b[z]) for z in 0..batch, with a[z] logically
/// [m, k] and b[z] [k, n]; out[z] is row-major [m, n] (out.trans must be
/// false), addressed as `MatRef` says. One launch, or two when split-k
/// applies (packed out, no epilogue but +=, see `split_count`).
#[allow(clippy::too_many_arguments)]
pub fn matmul(a: MatRef, b: MatRef, out: MatRef, batch: usize, m: usize, k: usize, n: usize, epi: Epilogue) {
    matmul_with(a, b, out, batch, m, k, n, epi, f16_enabled());
}

/// `TRAIN_F16=1` and a device with f16 x f16 -> f32 matrix cores: eligible matmuls run on them (f32 operands are rounded
/// to f16 while staged into shared memory, products accumulate in f32). Read once.
fn f16_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("TRAIN_F16").is_ok_and(|v| v == "1") && client().features().matmul.cmma.contains(&super::knn::f16_config()))
}

/// `matmul` with the f16 matrix-core path chosen by the caller (tests); it still falls back to f32 for shapes it can't tile.
#[allow(clippy::too_many_arguments)]
fn matmul_with(a: MatRef, b: MatRef, out: MatRef, batch: usize, m: usize, k: usize, n: usize, epi: Epilogue, use_f16: bool) {
    assert!(!out.trans, "out is stored row-major");
    if use_f16 {
        if let Some(cfg) = pick_cmma(batch, m, k, n) {
            return matmul_cmma(a, b, out, batch, m, k, n, epi, cfg);
        }
    }
    let dest = out;
    let dummy = dummy();
    let (bias_h, bias_off) = epi.bias.unwrap_or((dummy, 0));
    let (mask_h, mask_off) = epi.mask.unwrap_or((dummy, 0));
    let (res_h, res_off) = epi.residual.unwrap_or((dummy, 0));
    let tiles = n.div_ceil(64) * m.div_ceil(64);
    let k_blocks = k.div_ceil(16);
    let epi_free = epi.bias.is_none() && !epi.relu && epi.mask.is_none() && epi.residual.is_none();
    let splits = if epi_free && out.ld == 0 && out.group == 1 { split_count(batch * tiles, k_blocks) } else { 1 };
    let slice_blocks = k_blocks.div_ceil(splits);
    let count = CubeCount::Static((n as u32).div_ceil(64), (m as u32).div_ceil(64), (batch * splits) as u32);
    let u = |x: usize| x as u32;
    // Split: each (matrix, slice) writes its own [m, n] partial, at
    // z = matrix * splits + slice, no epilogue.
    let scratch = (splits > 1).then(|| client().empty(splits * m * n * 4));
    let out = match &scratch {
        Some(h) => MatRef { stride: m * n, ..MatRef::new(h) },
        None => out,
    };
    let accumulate = epi.accumulate && scratch.is_none();
    let (ag, ai, lda) = a.layout(if a.trans { m } else { k });
    let (bg, bi, ldb) = b.layout(if b.trans { k } else { n });
    let (og, oi, ldo) = out.layout(n);
    super::count_launch_as(|| {
        let t = |x: bool| if x { "T" } else { "N" };
        let ops = [(epi.bias.is_some(), " +bias"), (epi.relu, " relu"), (epi.mask.is_some(), " mask"), (epi.residual.is_some(), " +res"), (epi.accumulate, " +=")];
        let epi: String = ops.iter().filter(|o| o.0).map(|o| o.1).collect();
        format!("{batch}x[{m}x{k}]{}·[{k}x{n}]{}{epi} split {splits}", t(a.trans), t(b.trans))
    });
    k_matmul::launch(
        client(),
        count,
        CubeDim::new_2d(16, 16),
        whole(a.h),
        whole(b.h),
        whole(out.h),
        whole(bias_h),
        whole(mask_h),
        whole(res_h),
        u(m),
        u(n),
        u(k),
        u(slice_blocks),
        u(splits),
        u(a.off),
        u(a.stride),
        ag,
        ai,
        lda,
        u(b.off),
        u(b.stride),
        bg,
        bi,
        ldb,
        u(out.off),
        u(out.stride),
        og,
        oi,
        ldo,
        u(bias_off),
        u(epi.bias_stride),
        u(mask_off),
        u(res_off),
        a.trans,
        b.trans,
        epi.bias.is_some(),
        epi.relu,
        epi.mask.is_some(),
        epi.residual.is_some(),
        accumulate,
    );
    if let Some(partial) = &scratch {
        let len = m * n;
        super::count_launch_as(|| format!("split-k sum {batch}x{splits}x[{m}x{n}]"));
        k_split_sum::launch(
            client(),
            CubeCount::Static(((batch * len) as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            whole(partial),
            whole(dest.h),
            u(len),
            u(splits),
            u(dest.off),
            u(dest.stride),
            u(batch),
            epi.accumulate,
        );
    }
}

/// How many slices to split k into: enough that the output tiles times
/// the slices reach `SPLIT_TARGET` cubes, each slice keeping at least 4
/// blocks of 16 (64 of k). 1 means no split.
fn split_count(tiles: usize, k_blocks: usize) -> usize {
    SPLIT_TARGET.div_ceil(tiles).min(k_blocks / 4).max(1)
}

const SPLIT_TARGET: usize = 64;

/// Matrix b's out[i] (+)= sum over s of partial[(b * splits + s) * len + i],
/// s ascending.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_split_sum(partial: &[f32], out: &mut [f32], len: u32, splits: u32, o_off: u32, o_stride: u32, batch: u32, #[comptime] accumulate: bool) {
    let e = ABSOLUTE_POS as u32;
    if e < batch * len {
        let (b, i) = (e / len, e % len);
        let mut v = 0.0f32;
        for s in 0..splits {
            v += partial[((b * splits + s) * len + i) as usize];
        }
        let o = (o_off + b * o_stride + i) as usize;
        if accumulate {
            v += out[o];
        }
        out[o] = v;
    }
}

/// A whole buffer as a kernel argument.
fn whole(h: &Handle) -> BufferArg {
    buf(h, h.size_in_used() as usize / 4)
}

/// Bound in place of an absent bias, mask or residual: wgpu rejects one buffer
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
    mask: &[f32],
    res: &[f32],
    m: u32,
    n: u32,
    k: u32,
    slice_blocks: u32,
    splits: u32,
    a_off: u32,
    a_stride: u32,
    a_group: u32,
    a_inner: u32,
    lda: u32,
    b_off: u32,
    b_stride: u32,
    b_group: u32,
    b_inner: u32,
    ldb: u32,
    o_off: u32,
    o_stride: u32,
    o_group: u32,
    o_inner: u32,
    ldo: u32,
    bias_off: u32,
    bias_stride: u32,
    mask_off: u32,
    res_off: u32,
    #[comptime] trans_a: bool,
    #[comptime] trans_b: bool,
    #[comptime] has_bias: bool,
    #[comptime] relu: bool,
    #[comptime] has_mask: bool,
    #[comptime] has_res: bool,
    #[comptime] accumulate: bool,
) {
    let tx = UNIT_POS_X;
    let ty = UNIT_POS_Y;
    let tid = ty * 16 + tx;
    // z = batch * splits + slice; the output index keeps the whole z, so a
    // split's slices land in separate partials.
    let z = CUBE_POS_Z;
    let zb = z / splits;
    let kb0 = (z % splits) * slice_blocks;
    let kb1 = u32::min(kb0 + slice_blocks, k.div_ceil(16));
    let row0 = CUBE_POS_Y * 64;
    let col0 = CUBE_POS_X * 64;
    let a0 = a_off + (zb / a_group) * a_stride + (zb % a_group) * a_inner;
    let b0 = b_off + (zb / b_group) * b_stride + (zb % b_group) * b_inner;
    let o0 = (z / o_group) * o_stride + (z % o_group) * o_inner;
    let mut a_s = Shared::<[f32]>::new_slice(1024usize);
    let mut b_s = Shared::<[f32]>::new_slice(1024usize);
    let mut acc = Array::<f32>::new(16usize);
    let mut bv = Array::<f32>::new(4usize);
    #[unroll]
    for i in 0..16u32 {
        acc[i as usize] = 0.0;
    }
    for kk in kb0..kb1 {
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
                let mut idx = gm * lda + gk;
                if trans_a {
                    idx = gk * lda + gm;
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
                let mut idx = gk2 * ldb + gn;
                if trans_b {
                    idx = gn * ldb + gk2;
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
                    v += bias[(bias_off + zb * bias_stride + gn) as usize];
                }
                if relu {
                    if v < 0.0 {
                        v = 0.0;
                    }
                }
                let idx = o0 + gm * ldo + gn;
                if has_mask {
                    if mask[(mask_off + idx) as usize] <= 0.0 {
                        v = 0.0;
                    }
                }
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

/// A matrix-core block tile: the cube computes `bm` x `bn` outputs, as `pm` x `pn` planes each holding
/// (bm / 16pm) x (bn / 16pn) 16x16 accumulator fragments.
#[derive(Clone, Copy, Debug)]
struct CmmaCfg {
    bm: usize,
    bn: usize,
    pm: usize,
    pn: usize,
}

const CMMA_CFGS: [CmmaCfg; 4] = [
    CmmaCfg { bm: 128, bn: 128, pm: 2, pn: 2 },
    CmmaCfg { bm: 128, bn: 64, pm: 4, pn: 2 },
    CmmaCfg { bm: 64, bn: 64, pm: 2, pn: 2 },
    CmmaCfg { bm: 64, bn: 32, pm: 2, pn: 1 }, // the attention head width
];

/// k is staged 32 at a time.
const CMMA_BK: usize = 32;

/// The largest tile that divides m and n and still gives `SPLIT_TARGET` cubes (else the smallest that divides); None
/// when k, m or n don't tile: those matmuls stay on the f32 kernel.
fn pick_cmma(batch: usize, m: usize, k: usize, n: usize) -> Option<CmmaCfg> {
    static PLANE: OnceLock<usize> = OnceLock::new();
    let plane = *PLANE.get_or_init(|| client().properties().hardware.plane_size_max as usize);
    if k == 0 || k % CMMA_BK != 0 {
        return None;
    }
    let fits = |c: &&CmmaCfg| {
        let threads = plane * c.pm * c.pn;
        m % c.bm == 0 && n % c.bn == 0 && threads <= 1024 && (c.bm * CMMA_BK) % threads == 0 && (c.bn * CMMA_BK) % threads == 0
    };
    let cubes = |c: &CmmaCfg| batch * (m / c.bm) * (n / c.bn);
    let fitting: Vec<&CmmaCfg> = CMMA_CFGS.iter().filter(fits).collect();
    fitting.iter().find(|c| cubes(c) >= SPLIT_TARGET).or(fitting.last()).map(|c| **c)
}

/// The f16 matrix-core version of the launch(es) in `matmul_with`, same semantics; m, n, k tile `cfg`.
#[allow(clippy::too_many_arguments)]
fn matmul_cmma(a: MatRef, b: MatRef, out: MatRef, batch: usize, m: usize, k: usize, n: usize, epi: Epilogue, cfg: CmmaCfg) {
    let dest = out;
    let dummy = dummy();
    let (bias_h, bias_off) = epi.bias.unwrap_or((dummy, 0));
    let (mask_h, mask_off) = epi.mask.unwrap_or((dummy, 0));
    let (res_h, res_off) = epi.residual.unwrap_or((dummy, 0));
    let plane = client().properties().hardware.plane_size_max;
    let tiles = (m / cfg.bm) * (n / cfg.bn);
    let stages = k / CMMA_BK;
    let epi_free = epi.bias.is_none() && !epi.relu && epi.mask.is_none() && epi.residual.is_none();
    let splits = if epi_free && out.ld == 0 && out.group == 1 { SPLIT_TARGET.div_ceil(batch * tiles).min(stages / 2).max(1) } else { 1 };
    let slice_stages = stages.div_ceil(splits);
    let splits = stages.div_ceil(slice_stages);
    let scratch = (splits > 1).then(|| client().empty(batch * splits * m * n * 4));
    let out = match &scratch {
        Some(h) => MatRef { stride: m * n, ..MatRef::new(h) },
        None => out,
    };
    let accumulate = epi.accumulate && scratch.is_none();
    let (ag, ai, lda) = a.layout(if a.trans { m } else { k });
    let (bg, bi, ldb) = b.layout(if b.trans { k } else { n });
    let (og, oi, ldo) = out.layout(n);
    // cmma::store straight to global memory needs aligned offsets; otherwise (or with an epilogue) go through shared memory.
    let direct = epi_free && !accumulate && out.off % 4 == 0 && out.stride % 4 == 0 && oi % 4 == 0 && ldo % 4 == 0;
    super::count_launch_as(|| {
        let t = |x: bool| if x { "T" } else { "N" };
        let ops = [(epi.bias.is_some(), " +bias"), (epi.relu, " relu"), (epi.mask.is_some(), " mask"), (epi.residual.is_some(), " +res"), (epi.accumulate, " +=")];
        let epi: String = ops.iter().filter(|o| o.0).map(|o| o.1).collect();
        format!("{batch}x[{m}x{k}]{}·[{k}x{n}]{}{epi} f16 {}x{} split {splits}", t(a.trans), t(b.trans), cfg.bm, cfg.bn)
    });
    let u = |x: usize| x as u32;
    k_matmul_cmma::launch(
        client(),
        CubeCount::Static(u(n / cfg.bn), u(m / cfg.bm), u(batch * splits)),
        CubeDim::new_2d(plane, u(cfg.pm * cfg.pn)),
        whole(a.h),
        whole(b.h),
        whole(out.h),
        whole(bias_h),
        whole(mask_h),
        whole(res_h),
        u(slice_stages),
        u(stages),
        u(splits),
        u(a.off),
        u(a.stride),
        ag,
        ai,
        lda,
        u(b.off),
        u(b.stride),
        bg,
        bi,
        ldb,
        u(out.off),
        u(out.stride),
        og,
        oi,
        ldo,
        u(bias_off),
        u(epi.bias_stride),
        u(mask_off),
        u(res_off),
        a.trans,
        b.trans,
        u(cfg.bm),
        u(cfg.bn),
        u(cfg.pm),
        u(cfg.pn),
        plane * u(cfg.pm * cfg.pn),
        direct,
        epi.bias.is_some(),
        epi.relu,
        epi.mask.is_some(),
        epi.residual.is_some(),
        accumulate,
    );
    if let Some(partial) = &scratch {
        let len = m * n;
        super::count_launch_as(|| format!("split-k sum {batch}x{splits}x[{m}x{n}]"));
        k_split_sum::launch(
            client(),
            CubeCount::Static(((batch * len) as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            whole(partial),
            whole(dest.h),
            u(len),
            u(splits),
            u(dest.off),
            u(dest.stride),
            u(batch),
            epi.accumulate,
        );
    }
}

/// `k_matmul` on matrix cores (the spike's `k_cmma` plus the epilogue). f32 operands are rounded to f16 on the way into
/// shared memory, 32 of k per stage; each plane accumulates (bm / 16pm) x (bn / 16pn) 16x16 f32 fragments. m, n, k are
/// multiples of the tile / 32, so nothing is guarded. With an epilogue (or accumulate, or an unaligned out) each fragment
/// goes through a per-plane 16x16 shared tile and the lanes apply `k_matmul`'s epilogue element by element; otherwise it
/// is stored straight to out. Split-k as in `k_matmul`: z = batch * splits + slice.
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
#[cube(launch)]
fn k_matmul_cmma(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    bias: &[f32],
    mask: &[f32],
    res: &[f32],
    slice_stages: u32,
    stages: u32,
    splits: u32,
    a_off: u32,
    a_stride: u32,
    a_group: u32,
    a_inner: u32,
    lda: u32,
    b_off: u32,
    b_stride: u32,
    b_group: u32,
    b_inner: u32,
    ldb: u32,
    o_off: u32,
    o_stride: u32,
    o_group: u32,
    o_inner: u32,
    ldo: u32,
    bias_off: u32,
    bias_stride: u32,
    mask_off: u32,
    res_off: u32,
    #[comptime] ta: bool,
    #[comptime] tb: bool,
    #[comptime] bm: u32,
    #[comptime] bn: u32,
    #[comptime] pm: u32,
    #[comptime] pn: u32,
    #[comptime] threads: u32,
    #[comptime] direct: bool,
    #[comptime] has_bias: bool,
    #[comptime] relu: bool,
    #[comptime] has_mask: bool,
    #[comptime] has_res: bool,
    #[comptime] accumulate: bool,
) {
    let pad = 8u32;
    let fm = comptime![bm / (pm * 16)];
    let fnn = comptime![bn / (pn * 16)];
    let a_ld = comptime![if ta { bm + pad } else { 32 + pad }];
    let b_ld = comptime![if tb { 32 + pad } else { bn + pad }];
    let a_len = comptime![(if ta { 32 * (bm + pad) } else { bm * (32 + pad) }) as usize];
    let b_len = comptime![(if tb { bn * (32 + pad) } else { 32 * (bn + pad) }) as usize];
    let a_iters = comptime![bm * 32 / threads];
    let b_iters = comptime![bn * 32 / threads];
    let mut a_s = Shared::<[f16]>::new_slice(a_len);
    let mut b_s = Shared::<[f16]>::new_slice(b_len);

    let lane = UNIT_POS_X;
    let pid = UNIT_POS_Y;
    let tid = UNIT_POS;
    let wm0 = (pid / pn) * fm * 16;
    let wn0 = (pid % pn) * fnn * 16;
    let row0 = CUBE_POS_Y * bm;
    let col0 = CUBE_POS_X * bn;
    let z = CUBE_POS_Z;
    let zb = z / splits;
    let s0 = (z % splits) * slice_stages;
    let s1 = u32::min(s0 + slice_stages, stages);
    let a0 = a_off + (zb / a_group) * a_stride + (zb % a_group) * a_inner;
    let b0 = b_off + (zb / b_group) * b_stride + (zb % b_group) * b_inner;
    let o0 = (z / o_group) * o_stride + (z % o_group) * o_inner;

    let fm_n = comptime![(bm / (pm * 16)) as usize];
    let fn_n = comptime![(bn / (pn * 16)) as usize];
    let la = comptime![if ta { cmma::MatrixLayout::ColMajor } else { cmma::MatrixLayout::RowMajor }];
    let lb = comptime![if tb { cmma::MatrixLayout::ColMajor } else { cmma::MatrixLayout::RowMajor }];
    let mut acc = Sequence::<cmma::Matrix<f32>>::new();
    #[unroll]
    for _i in 0..fm_n * fn_n {
        acc.push(cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0));
    }

    for s in s0..s1 {
        let k0 = s * 32;
        #[unroll]
        for i in 0..a_iters {
            let e = tid + threads * i;
            if ta {
                let c = e / bm;
                let r = e % bm;
                a_s[(c * a_ld + r) as usize] = f16::cast_from(a[(a0 + (k0 + c) * lda + row0 + r) as usize]);
            } else {
                let r = e / 32;
                let c = e % 32;
                a_s[(r * a_ld + c) as usize] = f16::cast_from(a[(a0 + (row0 + r) * lda + k0 + c) as usize]);
            }
        }
        #[unroll]
        for i in 0..b_iters {
            let e = tid + threads * i;
            if tb {
                let c = e / 32;
                let r = e % 32;
                b_s[(c * b_ld + r) as usize] = f16::cast_from(b[(b0 + (col0 + c) * ldb + k0 + r) as usize]);
            } else {
                let r = e / bn;
                let c = e % bn;
                b_s[(r * b_ld + c) as usize] = f16::cast_from(b[(b0 + (k0 + r) * ldb + col0 + c) as usize]);
            }
        }
        sync_cube();
        #[unroll]
        for kf in 0..2u32 {
            let mut bf = Sequence::<cmma::Matrix<f16>>::new();
            #[unroll]
            for j in 0..fn_n {
                let mut off = (kf * 16 * b_ld + wn0 + j as u32 * 16) as usize;
                if tb {
                    off = ((wn0 + j as u32 * 16) * b_ld + kf * 16) as usize;
                }
                bf.push(cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, lb, &b_s[off..b_len], b_ld));
            }
            #[unroll]
            for i in 0..fm_n {
                let mut off = ((wm0 + i as u32 * 16) * a_ld + kf * 16) as usize;
                if ta {
                    off = (kf * 16 * a_ld + wm0 + i as u32 * 16) as usize;
                }
                let af = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, la, &a_s[off..a_len], a_ld);
                #[unroll]
                for j in 0..fn_n {
                    cmma::execute(&af, bf.index(j), acc.index(i * fn_n + j), acc.index(i * fn_n + j));
                }
            }
        }
        sync_cube();
    }

    let len = out.len();
    let planes = comptime![(pm * pn) as usize];
    let mut tile = Shared::<[f32]>::new_slice(comptime![planes * 256]);
    #[unroll]
    for i in 0..fm_n {
        #[unroll]
        for j in 0..fn_n {
            let gm0 = row0 + wm0 + i as u32 * 16;
            let gn0 = col0 + wn0 + j as u32 * 16;
            if direct {
                cmma::store(&mut out[(o_off + o0 + gm0 * ldo + gn0) as usize..len], acc.index(i * fn_n + j), ldo, cmma::MatrixLayout::RowMajor);
            } else {
                let base = (pid * 256) as usize;
                cmma::store(&mut tile[base..base + 256usize], acc.index(i * fn_n + j), 16u32, cmma::MatrixLayout::RowMajor);
                sync_cube();
                for it in 0..(256u32 / CUBE_DIM_X) {
                    let e = lane + it * CUBE_DIM_X;
                    let gm = gm0 + e / 16;
                    let gn = gn0 + e % 16;
                    let mut v = tile[base + e as usize];
                    if has_bias {
                        v += bias[(bias_off + zb * bias_stride + gn) as usize];
                    }
                    if relu {
                        if v < 0.0 {
                            v = 0.0;
                        }
                    }
                    let idx = gm * ldo + gn;
                    if has_mask {
                        if mask[(mask_off + o0 + idx) as usize] <= 0.0 {
                            v = 0.0;
                        }
                    }
                    if has_res {
                        v += res[(res_off + o0 + idx) as usize];
                    }
                    let o = (o_off + o0 + idx) as usize;
                    if accumulate {
                        v += out[o];
                    }
                    out[o] = v;
                }
                sync_cube();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_step::{read, upload};
    use crate::nn::Rng;

    /// Plain-loop reference for one case, same argument meaning as `matmul`.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        a: &[f32],
        ar: (usize, usize, bool),
        b: &[f32],
        br: (usize, usize, bool),
        out: &mut [f32],
        or: (usize, usize),
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        bias: Option<(&[f32], usize)>,
        relu: bool,
        mask: Option<&[f32]>,
        res: Option<&[f32]>,
        accumulate: bool,
    ) {
        for z in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f32;
                    for p in 0..k {
                        let x = a[ar.0 + z * ar.1 + if ar.2 { p * m + i } else { i * k + p }];
                        let y = b[br.0 + z * br.1 + if br.2 { j * k + p } else { p * n + j }];
                        s += x * y;
                    }
                    if let Some((bias, stride)) = bias {
                        s += bias[z * stride + j];
                    }
                    if relu {
                        s = s.max(0.0);
                    }
                    let idx = z * or.1 + i * n + j;
                    if mask.is_some_and(|mask| mask[idx] <= 0.0) {
                        s = 0.0;
                    }
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
        check_matmul(false);
    }

    /// Needs the discrete GPU and matrix cores. The f16 path (operands rounded to f16, f32 accumulation) over the same kind
    /// of cases but with tileable sizes (m, n multiples of 64, k of 32), plus ragged ones that must fall back to f32 and
    /// head-view-free batched ones. Tolerance 1e-2 of the output scale: inputs lose ~11 bits.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn matmul_f16_matches_reference() {
        if !client().features().matmul.cmma.contains(&crate::gpu_step::knn::f16_config()) {
            return;
        }
        check_matmul(true);
    }

    fn check_matmul(use_f16: bool) {
        let mut rng = Rng::new(5);
        let mut gauss = |len: usize| -> Vec<f32> { (0..len).map(|_| rng.next_gaussian()).collect() };
        // (batch, m, k, n, trans_a, trans_b, bias, relu, mask, residual, accumulate)
        let cases_f16 = [
            (1, 128, 64, 64, false, false, false, false, false, false, false),
            (1, 128, 64, 64, true, false, false, false, false, false, false),
            (1, 128, 64, 64, false, true, false, false, false, false, false),
            (1, 128, 64, 64, true, true, false, false, false, false, false),
            (3, 64, 96, 128, false, true, true, true, false, false, false),
            (3, 64, 64, 64, true, false, false, false, false, true, true),
            (2, 128, 32, 128, false, false, true, false, true, true, true), // everything, bias per matrix
            (1, 128, 64, 64, false, false, false, false, true, false, false),
            (1, 256, 128, 256, false, true, false, false, false, false, false), // 128x128 tile
            (1, 256, 128, 128, false, false, true, true, false, false, false), // 128x64 tile
            (1, 128, 1024, 128, true, false, false, false, false, false, false), // split 8, no epilogue
            (1, 128, 1024, 128, true, false, false, false, false, false, true), // split, accumulated
            (3, 64, 1024, 64, true, false, false, false, false, false, true), // batched split
            (1, 64, 256, 64, true, false, false, false, false, false, false), // split 4
            (1, 64, 256, 64, false, false, false, false, false, false, true), // split, no transpose
            (1, 70, 64, 64, false, false, true, false, false, false, false),  // m ragged: f32
            (1, 64, 48, 64, false, false, true, false, false, false, false),  // k not a multiple of 32: f32
            (2, 33, 17, 65, true, false, true, true, true, true, true),       // f32
        ];
        let cases_f32 = [
            (1, 70, 35, 20, false, false, false, false, false, false, false),
            (1, 70, 35, 20, true, false, false, false, false, false, false),
            (1, 70, 35, 20, false, true, false, false, false, false, false),
            (1, 70, 35, 20, true, true, false, false, false, false, false),
            (3, 33, 17, 65, false, true, true, true, false, false, false),
            (3, 64, 64, 64, true, false, false, false, false, true, true),
            (1, 512, 128, 384, false, false, true, false, false, false, false), // QKV forward
            (1, 128, 512, 384, true, false, false, false, false, false, true),  // dW (TN), accumulated, split 6
            (1, 128, 512, 128, true, false, false, false, false, false, true),  // dW, split 8
            (1, 70, 300, 20, true, false, false, false, false, false, false),   // split 4, ragged last slice
            (3, 128, 512, 128, true, false, false, false, false, false, true),  // fused models' dW: 3 matrices, split 6 each
            (2, 70, 300, 20, true, false, false, false, false, false, false),   // batched split 4, ragged
            (1, 512, 256, 128, false, true, false, false, false, false, false), // dX (NT)
            (64, 64, 16, 64, false, true, false, false, false, false, false),   // AttnScores
            (64, 64, 64, 16, false, false, false, false, false, false, false),  // AttnOut
            (1, 512, 128, 256, false, false, true, true, false, false, false),  // FFN1
            (1, 512, 256, 128, false, false, true, false, false, true, false),  // FFN2 + residual
            (1, 512, 128, 256, false, true, false, false, true, false, false),  // FFN1 dX, ReLU-masked
            (2, 33, 17, 65, true, false, true, true, true, true, true),         // everything at once
        ];
        let cases: &[_] = if use_f16 { &cases_f16 } else { &cases_f32 };
        let tol = if use_f16 { 1e-2 } else { 1e-5 };
        for (ci, &(batch, m, k, n, ta, tb, has_bias, relu, has_mask, has_res, acc)) in cases.iter().enumerate() {
            // Offsets and strides that aren't multiples of anything; every other f16 case has an aligned out
            // (the direct cmma store) instead.
            let aligned = use_f16 && ci % 2 == 1;
            let (a_off, b_off, o_off, bias_off, m_off, r_off) = (3, 5, if aligned { 8 } else { 7 }, 2, 6, 1);
            let (sa, sb, so) = (m * k + 1, if batch > 1 { k * n + 2 } else { 0 }, m * n + if aligned { 0 } else { 3 });
            let a = gauss(a_off + batch * sa);
            let b = gauss(b_off + batch.max(1) * sb.max(k * n));
            // A bias per batch matrix (fused models' Linear) when batched.
            let bias_stride = if batch > 1 { n + 1 } else { 0 };
            let bias = gauss(bias_off + batch * (n + 1));
            let res = gauss(r_off + batch * so);
            let mask = gauss(m_off + batch * so); // about half positive
            let out0 = gauss(o_off + batch * so);
            let mut want = out0.clone();
            reference(
                &a,
                (a_off, sa, ta),
                &b,
                (b_off, sb, tb),
                &mut want,
                (o_off, so),
                batch,
                m,
                k,
                n,
                has_bias.then_some((&bias[bias_off..], bias_stride)),
                relu,
                has_mask.then_some(&mask[m_off..]),
                has_res.then_some(&res[r_off..]),
                acc,
            );

            let (ah, bh, bias_h, mask_h, res_h, oh) = (upload(&a), upload(&b), upload(&bias), upload(&mask), upload(&res), upload(&out0));
            let epi = Epilogue {
                bias: has_bias.then_some((&bias_h, bias_off)),
                bias_stride,
                relu,
                mask: has_mask.then_some((&mask_h, m_off)),
                residual: has_res.then_some((&res_h, r_off)),
                accumulate: acc,
            };
            matmul_with(
                MatRef { off: a_off, stride: sa, trans: ta, ..MatRef::new(&ah) },
                MatRef { off: b_off, stride: sb, trans: tb, ..MatRef::new(&bh) },
                MatRef { off: o_off, stride: so, trans: false, ..MatRef::new(&oh) },
                batch,
                m,
                k,
                n,
                epi,
                use_f16,
            );
            let got = read(&oh);
            let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs()));
            let err = got.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
            let case = format!("batch {batch} m {m} k {k} n {n} ta {ta} tb {tb} bias {has_bias} relu {relu} mask {has_mask} res {has_res} acc {acc}");
            assert_eq!(got.len(), want.len(), "{case}");
            // Also covers the padding between batches and before the offset.
            let case = format!("{case} f16 {use_f16} cfg {:?}", if use_f16 { pick_cmma(batch, m, k, n) } else { None });
            assert!(err <= tol * scale, "{case}: max err {err} (scale {scale})");
        }
    }

    /// Needs the discrete GPU. Head views (`MatRef::heads`) against plain
    /// loops, B 2, H 3, t 5, width 4, in a fused [B·t, 3·H·w] buffer:
    /// scores = Q Kᵀ into a stacked out; ctx += P V into the merged
    /// [B·t, H·w] layout; dV += Pᵀ dCtx, a transposed stacked operand times
    /// a head view, into V's columns of the fused buffer.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn matmul_head_views_match_reference() {
        check_head_views(2, 3, 5, 4, false, 1e-5);
    }

    /// The same at the attention shapes (head width 32, t 64), where the f16 path must take every one of the three.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn matmul_f16_head_views_match_reference() {
        if !client().features().matmul.cmma.contains(&crate::gpu_step::knn::f16_config()) {
            return;
        }
        check_head_views(2, 4, 64, 32, true, 1e-2);
        assert!(pick_cmma(8, 64, 32, 64).is_some() && pick_cmma(8, 64, 64, 32).is_some());
    }

    fn check_head_views(bsz: usize, heads: usize, t: usize, w: usize, use_f16: bool, tol: f32) {
        let (hw, fused) = (heads * w, 3 * heads * w);
        let batch = bsz * heads;
        let mut rng = Rng::new(9);
        let mut gauss = |len: usize| -> Vec<f32> { (0..len).map(|_| rng.next_gaussian()).collect() };
        let (qkv, p, ctx0, dctx) = (gauss(bsz * t * fused), gauss(batch * t * t), gauss(bsz * t * hw), gauss(bsz * t * hw));
        // Element (i, j) of head matrix z at column col0 of a [B·t, cols] buffer.
        let at = |cols: usize, col0: usize, z: usize, i: usize, j: usize| (z / heads * t + i) * cols + col0 + z % heads * w + j;
        let (qh, ph, dh) = (upload(&qkv), upload(&p), upload(&dctx));
        let err = |got: &[f32], want: &[f32]| {
            let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs()));
            got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max) / scale
        };

        let mut want = vec![0.0f32; batch * t * t];
        for z in 0..batch {
            for i in 0..t {
                for j in 0..t {
                    want[(z * t + i) * t + j] = (0..w).map(|c| qkv[at(fused, 0, z, i, c)] * qkv[at(fused, hw, z, j, c)]).sum();
                }
            }
        }
        let sh = client().empty(want.len() * 4);
        matmul_with(
            MatRef::heads(&qh, 0, fused, t, heads, w),
            MatRef { trans: true, ..MatRef::heads(&qh, hw, fused, t, heads, w) },
            MatRef { stride: t * t, ..MatRef::new(&sh) },
            batch,
            t,
            w,
            t,
            Epilogue::default(),
            use_f16,
        );
        let e = err(&read(&sh), &want);
        assert!(e < tol, "scores: {e}");

        let mut want = ctx0.clone();
        for z in 0..batch {
            for i in 0..t {
                for j in 0..w {
                    want[at(hw, 0, z, i, j)] += (0..t).map(|c| p[(z * t + i) * t + c] * qkv[at(fused, 2 * hw, z, c, j)]).sum::<f32>();
                }
            }
        }
        let ch = upload(&ctx0);
        matmul_with(
            MatRef { stride: t * t, ..MatRef::new(&ph) },
            MatRef::heads(&qh, 2 * hw, fused, t, heads, w),
            MatRef::heads(&ch, 0, hw, t, heads, w),
            batch,
            t,
            t,
            w,
            Epilogue { accumulate: true, ..Default::default() },
            use_f16,
        );
        let e = err(&read(&ch), &want);
        assert!(e < tol, "ctx: {e}");

        let mut want = qkv.clone();
        for z in 0..batch {
            for i in 0..t {
                for j in 0..w {
                    want[at(fused, 2 * hw, z, i, j)] += (0..t).map(|c| p[(z * t + c) * t + i] * dctx[at(hw, 0, z, c, j)]).sum::<f32>();
                }
            }
        }
        let gh = upload(&qkv);
        matmul_with(
            MatRef { stride: t * t, trans: true, ..MatRef::new(&ph) },
            MatRef::heads(&dh, 0, hw, t, heads, w),
            MatRef::heads(&gh, 2 * hw, fused, t, heads, w),
            batch,
            t,
            t,
            w,
            Epilogue { accumulate: true, ..Default::default() },
            use_f16,
        );
        let e = err(&read(&gh), &want);
        assert!(e < tol, "dV: {e}");
    }
}
