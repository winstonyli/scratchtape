//! Row-wise kernels (milestone 2): LayerNorm forward and backward, Softmax
//! (plain or softmax1, with the causal mask and score scale folded in) and
//! its backward. (Bias gradients are column sums folded into the weight-gradient
//! matmul, see `matmul::Epilogue::col_sum`.)
//!
//! A row reduction runs one cube of `ROW_DIM` units per row; a column
//! reduction runs cubes of `COL_W` columns × `COL_LANES` row lanes. Each
//! unit sums its strided share in a fixed order, then a fixed shared-memory
//! tree combines the partials, so results are deterministic (though not
//! bit-equal to the CPU's serial sums). The first version gave each row or
//! column a single serial unit: 512 rows filled 2 cubes, no load was
//! coalesced, and these kernels were 62% of the training step
//! (gpu_step_design.md, milestone 5).
//!
//! Parameter gradients always accumulate into the flat gradient buffer,
//! which `zero_grads` clears once per step.
use super::{Models, buf, client};
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Units per row-reduction cube. `row_reduce`'s tree (64 slots, 6 levels)
/// is written for this value.
const ROW_DIM: u32 = 64;
/// Column-reduction cubes: `COL_W` adjacent columns (so loads coalesce) ×
/// `COL_LANES` lanes, lane l summing rows l, l + COL_LANES, ...
/// `lane_reduce`'s tree (16 × 16 slots, 4 levels) is written for these.
const COL_W: u32 = 16;
const COL_LANES: u32 = 16;

fn whole(h: &Handle) -> BufferArg {
    buf(h, h.size_in_used() as usize / 4)
}

/// One cube per row: row CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X, the grid
/// folded into y (an exact divisor) once rows pass a dimension's 65535
/// limit, as fused models' attention rows do.
fn per_row(rows: usize) -> (CubeCount, CubeDim) {
    let y = (1..=rows).find(|y| rows.is_multiple_of(*y) && rows / y <= 65535).unwrap();
    assert!(y <= 65535, "one cube per row: {rows} rows");
    (CubeCount::Static((rows / y) as u32, y as u32, 1), CubeDim::new_1d(ROW_DIM))
}

/// One cube per COL_W columns per model (the grid's y).
fn per_cols(n: usize, models: usize) -> (CubeCount, CubeDim) {
    (CubeCount::Static((n as u32).div_ceil(COL_W), models as u32, 1), CubeDim::new_2d(COL_W, COL_LANES))
}

fn empty(len: usize) -> Handle {
    client().empty(len * 4)
}

/// What LayerNorm's backward needs from its forward.
#[derive(Clone)]
pub struct LnOut {
    pub y: Handle,
    pub mean: Handle,
    pub rstd: Handle,
}

/// y = (x - mean) / sqrt(var + eps) * gamma + beta per row of x [rows, d],
/// biased variance, as `LayerNorm::forward_shared`. gamma and beta sit at
/// `off` and `off + d` in `params` (`LayerNorm::to_flat` order; each
/// model's own, under `m`).
#[allow(clippy::too_many_arguments)]
pub fn layer_norm(x: &Handle, rows: usize, d: usize, params: &Handle, off: usize, eps: f32, m: Models) -> LnOut {
    let out = LnOut { y: empty(rows * d), mean: empty(rows), rstd: empty(rows) };
    let (count, dim) = per_row(rows);
    super::count_launch();
    k_ln_fwd::launch(
        client(),
        count,
        dim,
        whole(x),
        whole(params),
        whole(&out.y),
        whole(&out.mean),
        whole(&out.rstd),
        d as u32,
        off as u32,
        eps,
        (rows / m.k) as u32,
        m.stride as u32,
    );
    out
}

/// dx (+)= LayerNorm's input gradient, and gamma/beta's gradients
/// accumulate into `grads` at `off` / `off + d` (per model). Two launches.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward(dy: &Handle, x: &Handle, fwd: &LnOut, rows: usize, d: usize, params: &Handle, grads: &Handle, off: usize, dx: &Handle, accumulate: bool, m: Models) {
    let c = client();
    let (rows_pm, stride) = ((rows / m.k) as u32, m.stride as u32);
    let (count, dim) = per_row(rows);
    super::count_launch();
    k_ln_bwd_dx::launch(c, count, dim, whole(dy), whole(x), whole(params), whole(&fwd.mean), whole(&fwd.rstd), whole(dx), d as u32, off as u32, accumulate, rows_pm, stride);
    let (count, dim) = per_cols(d, m.k);
    super::count_launch();
    k_ln_bwd_params::launch(c, count, dim, whole(dy), whole(x), whole(&fwd.mean), whole(&fwd.rstd), whole(grads), rows_pm, d as u32, off as u32, stride);
}

/// Row-wise softmax of `scale * x` over x [rows, n]. `causal`: row r is
/// query position r % t and sees only columns 0..=r % t (the rest are
/// exactly 0, as the CPU's -inf mask gives). `one`: softmax1, a phantom
/// logit 0 in the denominator, shifted by max(row max, 0) as
/// `Tape::softmax1`.
pub fn softmax(x: &Handle, rows: usize, n: usize, t: usize, scale: f32, causal: bool, one: bool) -> Handle {
    let y = empty(rows * n);
    let (count, dim) = per_row(rows);
    super::count_launch();
    k_softmax_fwd::launch(client(), count, dim, whole(x), whole(&y), n as u32, t as u32, scale, causal, one);
    y
}

/// dx = scale * y * (dy - sum(dy * y)) per row: the input gradient of
/// `softmax` for either variant (softmax1 has the same Jacobian in terms
/// of its output; masked entries have y = 0, so they get 0).
pub fn softmax_backward(dy: &Handle, y: &Handle, rows: usize, n: usize, scale: f32) -> Handle {
    let dx = empty(rows * n);
    let (count, dim) = per_row(rows);
    super::count_launch();
    k_softmax_bwd::launch(client(), count, dim, whole(dy), whole(y), whole(&dx), n as u32, scale);
    dx
}

/// The sum over the cube's units of `v` (or the max, with `max`),
/// combined by a fixed tree in its own shared memory. Every unit of the
/// cube must call it, and every unit gets the result. (The shared array
/// can't be passed in: cubecl pre.4 emits SPIR-V that fails validation,
/// "Expected operand type cube.ptr, but found cube.index".)
#[cube]
fn row_reduce(v: f32, u: u32, #[comptime] max: bool) -> f32 {
    let mut sh = Shared::<[f32]>::new_slice(64usize);
    sh[u as usize] = v;
    sync_cube();
    #[unroll]
    for k in 0..6u32 {
        let s = 32u32 >> k;
        if u < s {
            let a = sh[u as usize];
            let b = sh[(u + s) as usize];
            if max {
                sh[u as usize] = f32::max(a, b);
            } else {
                sh[u as usize] = a + b;
            }
        }
        sync_cube();
    }
    let total = sh[0usize];
    // in case the next call reuses this memory
    sync_cube();
    total
}

/// The column-cube version: the sum of `v` over the lanes of column tx,
/// combined by a fixed tree. Every unit must call it; the result is valid
/// at lane 0.
#[cube]
fn lane_reduce(v: f32, tx: u32, lane: u32) -> f32 {
    let mut sh = Shared::<[f32]>::new_slice(256usize);
    sh[(lane * 16 + tx) as usize] = v;
    sync_cube();
    #[unroll]
    for k in 0..4u32 {
        let s = 8u32 >> k;
        if lane < s {
            let i = lane * 16 + tx;
            sh[i as usize] += sh[(i + s * 16) as usize];
        }
        sync_cube();
    }
    let total = sh[tx as usize];
    sync_cube();
    total
}

#[cube(launch)]
fn k_ln_fwd(x: &[f32], p: &[f32], y: &mut [f32], mean: &mut [f32], rstd: &mut [f32], d: u32, off0: u32, eps: f32, rows_pm: u32, stride: u32) {
    let r = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let u = UNIT_POS_X;
    let base = r * d;
    let off = off0 + (r / rows_pm) * stride;
    let mut part = 0.0f32;
    let mut j = u;
    while j < d {
        part += x[(base + j) as usize];
        j += ROW_DIM;
    }
    let mu = row_reduce(part, u, false) / f32::cast_from(d);
    let mut sq = 0.0f32;
    j = u;
    while j < d {
        let c = x[(base + j) as usize] - mu;
        sq += c * c;
        j += ROW_DIM;
    }
    let std = f32::sqrt(row_reduce(sq, u, false) / f32::cast_from(d) + eps);
    j = u;
    while j < d {
        let xhat = (x[(base + j) as usize] - mu) / std;
        y[(base + j) as usize] = xhat * p[(off + j) as usize] + p[(off + d + j) as usize];
        j += ROW_DIM;
    }
    if u == 0 {
        mean[r as usize] = mu;
        rstd[r as usize] = 1.0 / std;
    }
}

/// dx = rstd * (g - mean(g) - xhat * mean(g * xhat)), g = dy * gamma.
#[allow(clippy::collapsible_if)]
#[cube(launch)]
fn k_ln_bwd_dx(dy: &[f32], x: &[f32], p: &[f32], mean: &[f32], rstd: &[f32], dx: &mut [f32], d: u32, off0: u32, #[comptime] accumulate: bool, rows_pm: u32, stride: u32) {
    let r = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let u = UNIT_POS_X;
    let base = r * d;
    let off = off0 + (r / rows_pm) * stride;
    let mu = mean[r as usize];
    let rs = rstd[r as usize];
    let mut p1 = 0.0f32;
    let mut p2 = 0.0f32;
    let mut j = u;
    while j < d {
        let g = dy[(base + j) as usize] * p[(off + j) as usize];
        p1 += g;
        p2 += g * (x[(base + j) as usize] - mu) * rs;
        j += ROW_DIM;
    }
    let s1 = row_reduce(p1, u, false);
    let s2 = row_reduce(p2, u, false);
    let inv_d = 1.0 / f32::cast_from(d);
    j = u;
    while j < d {
        let i = (base + j) as usize;
        let g = dy[i] * p[(off + j) as usize];
        let xhat = (x[i] - mu) * rs;
        let mut v = rs * (g - s1 * inv_d - xhat * s2 * inv_d);
        if accumulate {
            v += dx[i];
        }
        dx[i] = v;
        j += ROW_DIM;
    }
}

/// Per column j: dgamma += sum_r dy * xhat, dbeta += sum_r dy, over
/// model CUBE_POS_Y's `rows` rows, into its parameters.
#[cube(launch)]
fn k_ln_bwd_params(dy: &[f32], x: &[f32], mean: &[f32], rstd: &[f32], g: &mut [f32], rows: u32, d: u32, off0: u32, stride: u32) {
    let tx = UNIT_POS_X;
    let lane = UNIT_POS_Y;
    let j = CUBE_POS_X * COL_W + tx;
    let r0 = CUBE_POS_Y * rows;
    let off = off0 + CUBE_POS_Y * stride;
    let mut pg = 0.0f32;
    let mut pb = 0.0f32;
    if j < d {
        let mut r = lane;
        // Four rows' loads are issued before their sums (same add order
        // as the plain loop): the loop is load-latency bound, with only 16
        // cubes per model.
        while r + 3 * COL_LANES < rows {
            let (r1, r2, r3) = (r0 + r + COL_LANES, r0 + r + 2 * COL_LANES, r0 + r + 3 * COL_LANES);
            let r0r = r0 + r;
            let (i0, i1, i2, i3) = ((r0r * d + j) as usize, (r1 * d + j) as usize, (r2 * d + j) as usize, (r3 * d + j) as usize);
            let (d0, d1, d2, d3) = (dy[i0], dy[i1], dy[i2], dy[i3]);
            let (x0, x1, x2, x3) = (x[i0], x[i1], x[i2], x[i3]);
            let (m0, m1, m2, m3) = (mean[r0r as usize], mean[r1 as usize], mean[r2 as usize], mean[r3 as usize]);
            let (s0, s1, s2, s3) = (rstd[r0r as usize], rstd[r1 as usize], rstd[r2 as usize], rstd[r3 as usize]);
            pg += d0 * (x0 - m0) * s0;
            pb += d0;
            pg += d1 * (x1 - m1) * s1;
            pb += d1;
            pg += d2 * (x2 - m2) * s2;
            pb += d2;
            pg += d3 * (x3 - m3) * s3;
            pb += d3;
            r += 4 * COL_LANES;
        }
        while r < rows {
            let i = ((r0 + r) * d + j) as usize;
            pg += dy[i] * (x[i] - mean[(r0 + r) as usize]) * rstd[(r0 + r) as usize];
            pb += dy[i];
            r += COL_LANES;
        }
    }
    let sg = lane_reduce(pg, tx, lane);
    let sb = lane_reduce(pb, tx, lane);
    if lane == 0 && j < d {
        g[(off + j) as usize] += sg;
        g[(off + d + j) as usize] += sb;
    }
}

#[allow(clippy::collapsible_if)]
#[cube(launch)]
fn k_softmax_fwd(x: &[f32], y: &mut [f32], n: u32, t: u32, scale: f32, #[comptime] causal: bool, #[comptime] one: bool) {
    let r = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let u = UNIT_POS_X;
    let base = r * n;
    let mut limit = n;
    if causal {
        limit = r % t + 1;
    }
    // column 0 is always visible, so every unit's max starts from it
    let mut pm = scale * x[base as usize];
    let mut j = u;
    while j < limit {
        pm = f32::max(pm, scale * x[(base + j) as usize]);
        j += ROW_DIM;
    }
    let mut m = row_reduce(pm, u, true);
    if one {
        if m < 0.0 {
            m = 0.0;
        }
    }
    let mut ps = 0.0f32;
    j = u;
    while j < limit {
        ps += f32::exp(scale * x[(base + j) as usize] - m);
        j += ROW_DIM;
    }
    let mut s = row_reduce(ps, u, false);
    if one {
        s += f32::exp(-m);
    }
    j = u;
    while j < n {
        let mut v = 0.0f32;
        if j < limit {
            v = f32::exp(scale * x[(base + j) as usize] - m) / s;
        }
        y[(base + j) as usize] = v;
        j += ROW_DIM;
    }
}

#[cube(launch)]
fn k_softmax_bwd(dy: &[f32], y: &[f32], dx: &mut [f32], n: u32, scale: f32) {
    let r = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
    let u = UNIT_POS_X;
    let base = r * n;
    let mut pd = 0.0f32;
    let mut j = u;
    while j < n {
        pd += dy[(base + j) as usize] * y[(base + j) as usize];
        j += ROW_DIM;
    }
    let dot = row_reduce(pd, u, false);
    j = u;
    while j < n {
        let i = (base + j) as usize;
        dx[i] = scale * y[i] * (dy[i] - dot);
        j += ROW_DIM;
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_step::{read, upload};
    use crate::nn::{LayerNorm, Rng};
    use crate::tape::Tape;
    use crate::tensor::NdArray;

    fn gauss(rng: &mut Rng, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.next_gaussian()).collect()
    }

    /// Max |got - want| relative to want's largest magnitude.
    fn rel_err(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(1e-6f32, |s, v| s.max(v.abs()));
        got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0, f32::max) / scale
    }

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
    }

    /// Central differences of loss(x) = sum(f(x) * r) through the GPU
    /// forward `f`, at a spread of coordinates, against the GPU backward's
    /// `dx`. Step 1e-2 with an f64 host-side loss: roundoff and truncation
    /// both stay near 1e-4, so 1e-2 relative to dx's scale is a real check.
    fn finite_diff(x: &[f32], r: &[f32], dx: &[f32], f: impl Fn(&[f32]) -> Vec<f32>, what: &str) {
        let h = 1e-2f32;
        let scale = dx.iter().fold(1e-6f32, |s, v| s.max(v.abs()));
        for i in (0..x.len()).step_by(x.len() / 23 + 1) {
            let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
            xp[i] += h;
            xm[i] -= h;
            let num = ((dot(&f(&xp), r) - dot(&f(&xm), r)) / (2.0 * h as f64)) as f32;
            assert!((num - dx[i]).abs() <= 1e-2 * scale, "{what}: finite difference at {i}: {num} vs gpu {}", dx[i]);
        }
    }

    /// Needs the discrete GPU. LayerNorm forward, dx (written, then
    /// accumulated) and gamma/beta gradients against the CPU tape's
    /// composed LayerNorm, with non-trivial gamma/beta at an offset in the
    /// flat buffer; then dx by finite differences through the GPU forward.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn layer_norm_matches_cpu_tape() {
        // eps 1e-5, as LayerNorm::new; d = 200 spans several passes of a
        // row cube, rows = 37 isn't a multiple of the column lanes
        // rows = 203 runs the params kernel's 4-row unrolled loop and its tail
        for (rows, d, off) in [(37, 24, 5), (37, 200, 3), (203, 24, 5)] {
            layer_norm_case(rows, d, off);
        }
    }

    fn layer_norm_case(rows: usize, d: usize, off: usize) {
        let mut rng = Rng::new(7);
        let mut ln = LayerNorm::new(d);
        ln.gamma.data = (0..d).map(|j| 1.0 + 0.1 * ((j * 7) % 5) as f32).collect();
        ln.beta.data = (0..d).map(|j| 0.05 * ((j * 3) % 4) as f32 - 0.1).collect();
        let (x, r) = (gauss(&mut rng, rows * d), gauss(&mut rng, rows * d));
        let mut tape = Tape::new();
        let xv = tape.leaf(NdArray::new(x.clone(), vec![rows, d]));
        let out = ln.forward(&mut tape, xv);
        let rv = tape.leaf(NdArray::new(r.clone(), vec![rows, d]));
        let prod = tape.mul(out.y, rv);
        let loss = tape.sum(prod);
        tape.backward(loss);

        let mut flat = vec![9.0f32; off];
        flat.extend(ln.to_flat());
        flat.push(9.0);
        let (xh, rh, ph) = (upload(&x), upload(&r), upload(&flat));
        let fwd = layer_norm(&xh, rows, d, &ph, off, 1e-5, Models::ONE);
        let y = read(&fwd.y);
        assert!(rel_err(&y, &tape.value(out.y).data) < 1e-5, "d {d} forward: {}", rel_err(&y, &tape.value(out.y).data));

        let grads = upload(&vec![0.5f32; flat.len()]);
        let dx = upload(&vec![0.0f32; rows * d]);
        layer_norm_backward(&rh, &xh, &fwd, rows, d, &ph, &grads, off, &dx, false, Models::ONE);
        let got_dx = read(&dx);
        let want_dx = &tape.grad(xv).unwrap().data;
        assert!(rel_err(&got_dx, want_dx) < 1e-4, "dx: {}", rel_err(&got_dx, want_dx));
        let g = read(&grads);
        let want_g: Vec<f32> = tape.grad(out.gamma).unwrap().data.iter().map(|v| v + 0.5).collect();
        let want_b: Vec<f32> = tape.grad(out.beta).unwrap().data.iter().map(|v| v + 0.5).collect();
        assert!(rel_err(&g[off..off + d], &want_g) < 1e-4, "dgamma: {}", rel_err(&g[off..off + d], &want_g));
        assert!(rel_err(&g[off + d..off + 2 * d], &want_b) < 1e-4, "dbeta");
        assert!(g[..off].iter().chain(&g[off + 2 * d..]).all(|&v| v == 0.5), "wrote outside gamma/beta");

        layer_norm_backward(&rh, &xh, &fwd, rows, d, &ph, &grads, off, &dx, true, Models::ONE);
        let twice: Vec<f32> = want_dx.iter().map(|v| 2.0 * v).collect();
        assert!(rel_err(&read(&dx), &twice) < 1e-4, "dx accumulate");

        finite_diff(&x, &r, &got_dx, |xs| read(&layer_norm(&upload(xs), rows, d, &ph, off, 1e-5, Models::ONE).y), "layer norm");
    }

    /// Needs the discrete GPU. Softmax (plain and softmax1) over causal
    /// attention scores [B·H·T, T] with the 1/sqrt(d_k) scale, against the
    /// CPU tape's scale + (-inf mask) + softmax/softmax1; then its backward
    /// by finite differences. softmax1 gets large logits too, where its
    /// max(row max, 0) shift matters.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn softmax_matches_cpu_tape() {
        // t = 70: rows longer than a row cube
        for t in [7, 70] {
            softmax_case(t);
        }
    }

    fn softmax_case(t: usize) {
        let heads = 6;
        let (rows, scale) = (heads * t, 0.25f32);
        let mut rng = Rng::new(9);
        let mut mask = vec![0.0f32; rows * t];
        for row in 0..rows {
            for j in row % t + 1..t {
                mask[row * t + j] = f32::NEG_INFINITY;
            }
        }
        for (one, amp) in [(false, 1.0f32), (true, 1.0), (true, 40.0)] {
            let x: Vec<f32> = gauss(&mut rng, rows * t).iter().map(|v| v * amp).collect();
            let r = gauss(&mut rng, rows * t);
            let mut tape = Tape::new();
            let xv = tape.leaf(NdArray::new(x.clone(), vec![rows, t]));
            let scaled = tape.scale(xv, scale);
            let mv = tape.leaf(NdArray::new(mask.clone(), vec![rows, t]));
            let masked = tape.add(scaled, mv);
            let y = if one { tape.softmax1(masked) } else { tape.softmax(masked) };
            let rv = tape.leaf(NdArray::new(r.clone(), vec![rows, t]));
            let prod = tape.mul(y, rv);
            let loss = tape.sum(prod);
            tape.backward(loss);

            let what = format!("softmax t={t} one={one} amp={amp}");
            let yh = softmax(&upload(&x), rows, t, t, scale, true, one);
            let got = read(&yh);
            assert!(rel_err(&got, &tape.value(y).data) < 1e-5, "{what} forward: {}", rel_err(&got, &tape.value(y).data));
            let dx = read(&softmax_backward(&upload(&r), &yh, rows, t, scale));
            let want_dx = &tape.grad(xv).unwrap().data;
            assert!(rel_err(&dx, want_dx) < 1e-4, "{what} dx: {}", rel_err(&dx, want_dx));
            if amp == 1.0 {
                finite_diff(&x, &r, &dx, |xs| read(&softmax(&upload(xs), rows, t, t, scale, true, one)), &what);
            }
        }
    }
}
