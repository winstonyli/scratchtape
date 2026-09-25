//! Row-wise kernels (milestone 2): LayerNorm forward and backward, Softmax
//! (plain or softmax1, with the causal mask and score scale folded in) and
//! its backward, and the column sum that gives bias gradients. One unit
//! per row (or per column for the reductions), each summing in a fixed
//! order, so results are deterministic. Parameter gradients always
//! accumulate into the flat gradient buffer, which `zero_grads` clears
//! once per step.
use super::{EW_DIM, buf, client};
use cubecl::prelude::*;
use cubecl::server::Handle;

fn whole(h: &Handle) -> BufferArg {
    buf(h, h.size_in_used() as usize / 4)
}

fn cubes(units: usize) -> CubeCount {
    CubeCount::Static((units as u32).div_ceil(EW_DIM), 1, 1)
}

fn empty(len: usize) -> Handle {
    client().empty(len * 4)
}

/// What LayerNorm's backward needs from its forward.
pub struct LnOut {
    pub y: Handle,
    pub mean: Handle,
    pub rstd: Handle,
}

/// y = (x - mean) / sqrt(var + eps) * gamma + beta per row of x [rows, d],
/// biased variance, as `LayerNorm::forward_shared`. gamma and beta sit at
/// `off` and `off + d` in `params` (`LayerNorm::to_flat` order).
pub fn layer_norm(x: &Handle, rows: usize, d: usize, params: &Handle, off: usize, eps: f32) -> LnOut {
    let out = LnOut { y: empty(rows * d), mean: empty(rows), rstd: empty(rows) };
    k_ln_fwd::launch(client(), cubes(rows), CubeDim::new_1d(EW_DIM), whole(x), whole(params), whole(&out.y), whole(&out.mean), whole(&out.rstd), rows as u32, d as u32, off as u32, eps);
    out
}

/// dx (+)= LayerNorm's input gradient, and gamma/beta's gradients
/// accumulate into `grads` at `off` / `off + d`. Two launches.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward(dy: &Handle, x: &Handle, fwd: &LnOut, rows: usize, d: usize, params: &Handle, grads: &Handle, off: usize, dx: &Handle, accumulate: bool) {
    let c = client();
    k_ln_bwd_dx::launch(c, cubes(rows), CubeDim::new_1d(EW_DIM), whole(dy), whole(x), whole(params), whole(&fwd.mean), whole(&fwd.rstd), whole(dx), rows as u32, d as u32, off as u32, accumulate);
    k_ln_bwd_params::launch(c, cubes(d), CubeDim::new_1d(EW_DIM), whole(dy), whole(x), whole(&fwd.mean), whole(&fwd.rstd), whole(grads), rows as u32, d as u32, off as u32);
}

/// Row-wise softmax of `scale * x` over x [rows, n]. `causal`: row r is
/// query position r % t and sees only columns 0..=r % t (the rest are
/// exactly 0, as the CPU's -inf mask gives). `one`: softmax1, a phantom
/// logit 0 in the denominator, shifted by max(row max, 0) as
/// `Tape::softmax1`.
pub fn softmax(x: &Handle, rows: usize, n: usize, t: usize, scale: f32, causal: bool, one: bool) -> Handle {
    let y = empty(rows * n);
    k_softmax_fwd::launch(client(), cubes(rows), CubeDim::new_1d(EW_DIM), whole(x), whole(&y), rows as u32, n as u32, t as u32, scale, causal, one);
    y
}

/// dx = scale * y * (dy - sum(dy * y)) per row: the input gradient of
/// `softmax` for either variant (softmax1 has the same Jacobian in terms
/// of its output; masked entries have y = 0, so they get 0).
pub fn softmax_backward(dy: &Handle, y: &Handle, rows: usize, n: usize, scale: f32) -> Handle {
    let dx = empty(rows * n);
    k_softmax_bwd::launch(client(), cubes(rows), CubeDim::new_1d(EW_DIM), whole(dy), whole(y), whole(&dx), rows as u32, n as u32, scale);
    dx
}

/// out[off + j] += sum over rows of x[r, j], for x [rows, n]: a bias
/// gradient.
pub fn col_sum(x: &Handle, rows: usize, n: usize, out: &Handle, off: usize) {
    k_col_sum::launch(client(), cubes(n), CubeDim::new_1d(EW_DIM), whole(x), whole(out), rows as u32, n as u32, off as u32);
}

#[cube(launch)]
fn k_ln_fwd(x: &[f32], p: &[f32], y: &mut [f32], mean: &mut [f32], rstd: &mut [f32], rows: u32, d: u32, off: u32, eps: f32) {
    let r = ABSOLUTE_POS as u32;
    if r < rows {
        let base = r * d;
        let mut s = 0.0f32;
        for j in 0..d {
            s += x[(base + j) as usize];
        }
        let mu = s / f32::cast_from(d);
        let mut sq = 0.0f32;
        for j in 0..d {
            let c = x[(base + j) as usize] - mu;
            sq += c * c;
        }
        let std = f32::sqrt(sq / f32::cast_from(d) + eps);
        for j in 0..d {
            let xhat = (x[(base + j) as usize] - mu) / std;
            y[(base + j) as usize] = xhat * p[(off + j) as usize] + p[(off + d + j) as usize];
        }
        mean[r as usize] = mu;
        rstd[r as usize] = 1.0 / std;
    }
}

/// dx = rstd * (g - mean(g) - xhat * mean(g * xhat)), g = dy * gamma.
#[allow(clippy::collapsible_if)]
#[cube(launch)]
fn k_ln_bwd_dx(dy: &[f32], x: &[f32], p: &[f32], mean: &[f32], rstd: &[f32], dx: &mut [f32], rows: u32, d: u32, off: u32, #[comptime] accumulate: bool) {
    let r = ABSOLUTE_POS as u32;
    if r < rows {
        let base = r * d;
        let mu = mean[r as usize];
        let rs = rstd[r as usize];
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        for j in 0..d {
            let g = dy[(base + j) as usize] * p[(off + j) as usize];
            s1 += g;
            s2 += g * (x[(base + j) as usize] - mu) * rs;
        }
        let inv_d = 1.0 / f32::cast_from(d);
        for j in 0..d {
            let i = (base + j) as usize;
            let g = dy[i] * p[(off + j) as usize];
            let xhat = (x[i] - mu) * rs;
            let mut v = rs * (g - s1 * inv_d - xhat * s2 * inv_d);
            if accumulate {
                v += dx[i];
            }
            dx[i] = v;
        }
    }
}

/// Per column j: dgamma += sum_r dy * xhat, dbeta += sum_r dy.
#[cube(launch)]
fn k_ln_bwd_params(dy: &[f32], x: &[f32], mean: &[f32], rstd: &[f32], g: &mut [f32], rows: u32, d: u32, off: u32) {
    let j = ABSOLUTE_POS as u32;
    if j < d {
        let mut sg = 0.0f32;
        let mut sb = 0.0f32;
        for r in 0..rows {
            let i = (r * d + j) as usize;
            sg += dy[i] * (x[i] - mean[r as usize]) * rstd[r as usize];
            sb += dy[i];
        }
        g[(off + j) as usize] += sg;
        g[(off + d + j) as usize] += sb;
    }
}

#[allow(clippy::collapsible_if)]
#[cube(launch)]
fn k_softmax_fwd(x: &[f32], y: &mut [f32], rows: u32, n: u32, t: u32, scale: f32, #[comptime] causal: bool, #[comptime] one: bool) {
    let r = ABSOLUTE_POS as u32;
    if r < rows {
        let base = r * n;
        let mut limit = n;
        if causal {
            limit = r % t + 1;
        }
        let mut m = scale * x[base as usize];
        for j in 1..limit {
            let v = scale * x[(base + j) as usize];
            if v > m {
                m = v;
            }
        }
        if one {
            if m < 0.0 {
                m = 0.0;
            }
        }
        let mut s = 0.0f32;
        for j in 0..limit {
            s += f32::exp(scale * x[(base + j) as usize] - m);
        }
        if one {
            s += f32::exp(-m);
        }
        for j in 0..n {
            let mut v = 0.0f32;
            if j < limit {
                v = f32::exp(scale * x[(base + j) as usize] - m) / s;
            }
            y[(base + j) as usize] = v;
        }
    }
}

#[cube(launch)]
fn k_softmax_bwd(dy: &[f32], y: &[f32], dx: &mut [f32], rows: u32, n: u32, scale: f32) {
    let r = ABSOLUTE_POS as u32;
    if r < rows {
        let base = r * n;
        let mut dot = 0.0f32;
        for j in 0..n {
            dot += dy[(base + j) as usize] * y[(base + j) as usize];
        }
        for j in 0..n {
            let i = (base + j) as usize;
            dx[i] = scale * y[i] * (dy[i] - dot);
        }
    }
}

#[cube(launch)]
fn k_col_sum(x: &[f32], out: &mut [f32], rows: u32, n: u32, off: u32) {
    let j = ABSOLUTE_POS as u32;
    if j < n {
        let mut s = 0.0f32;
        for r in 0..rows {
            s += x[(r * n + j) as usize];
        }
        out[(off + j) as usize] += s;
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
        let (rows, d, off) = (37, 24, 5); // eps 1e-5, as LayerNorm::new
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
        let fwd = layer_norm(&xh, rows, d, &ph, off, 1e-5);
        let y = read(&fwd.y);
        assert!(rel_err(&y, &tape.value(out.y).data) < 1e-5, "forward: {}", rel_err(&y, &tape.value(out.y).data));

        let grads = upload(&vec![0.5f32; flat.len()]);
        let dx = upload(&vec![0.0f32; rows * d]);
        layer_norm_backward(&rh, &xh, &fwd, rows, d, &ph, &grads, off, &dx, false);
        let got_dx = read(&dx);
        let want_dx = &tape.grad(xv).unwrap().data;
        assert!(rel_err(&got_dx, want_dx) < 1e-4, "dx: {}", rel_err(&got_dx, want_dx));
        let g = read(&grads);
        let want_g: Vec<f32> = tape.grad(out.gamma).unwrap().data.iter().map(|v| v + 0.5).collect();
        let want_b: Vec<f32> = tape.grad(out.beta).unwrap().data.iter().map(|v| v + 0.5).collect();
        assert!(rel_err(&g[off..off + d], &want_g) < 1e-4, "dgamma: {}", rel_err(&g[off..off + d], &want_g));
        assert!(rel_err(&g[off + d..off + 2 * d], &want_b) < 1e-4, "dbeta");
        assert!(g[..off].iter().chain(&g[off + 2 * d..]).all(|&v| v == 0.5), "wrote outside gamma/beta");

        layer_norm_backward(&rh, &xh, &fwd, rows, d, &ph, &grads, off, &dx, true);
        let twice: Vec<f32> = want_dx.iter().map(|v| 2.0 * v).collect();
        assert!(rel_err(&read(&dx), &twice) < 1e-4, "dx accumulate");

        finite_diff(&x, &r, &got_dx, |xs| read(&layer_norm(&upload(xs), rows, d, &ph, off, 1e-5).y), "layer norm");
    }

    /// Needs the discrete GPU. Softmax (plain and softmax1) over causal
    /// attention scores [B·H·T, T] with the 1/sqrt(d_k) scale, against the
    /// CPU tape's scale + (-inf mask) + softmax/softmax1; then its backward
    /// by finite differences. softmax1 gets large logits too, where its
    /// max(row max, 0) shift matters.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn softmax_matches_cpu_tape() {
        let (heads, t) = (6, 7);
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

            let what = format!("softmax one={one} amp={amp}");
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

    /// Needs the discrete GPU. Bias gradient = column sum, accumulated at an
    /// offset.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn col_sum_accumulates_bias_gradient() {
        let (rows, n, off) = (70, 33, 4);
        let mut rng = Rng::new(2);
        let x = gauss(&mut rng, rows * n);
        let out = upload(&vec![1.0f32; off + n + 2]);
        col_sum(&upload(&x), rows, n, &out, off);
        let got = read(&out);
        let want: Vec<f32> = (0..n).map(|j| 1.0 + (0..rows).map(|r| x[r * n + j]).sum::<f32>()).collect();
        assert!(rel_err(&got[off..off + n], &want) < 1e-5);
        assert!(got[..off].iter().chain(&got[off + n..]).all(|&v| v == 1.0));
    }
}
