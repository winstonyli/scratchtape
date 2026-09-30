//! The two ops indexed by token id (milestone 2): Embed (token + position,
//! summed) and CrossEntropy (fused softmax + NLL, mean over rows). Their
//! reductions are ordered loops, not atomics, so they're deterministic.
use super::{EW_DIM, Models, buf, client, cubes};
use cubecl::prelude::*;
use cubecl::server::Handle;

fn whole(h: &Handle) -> BufferArg {
    buf(h, h.size_in_used() as usize / 4)
}

/// Token ids or targets, as the u32 buffer the kernels index with.
pub fn upload_ids(ids: &[usize]) -> Handle {
    let v: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    client().create_from_slice(u32::as_bytes(&v))
}

/// y [rows, d] = token table row ids[r] + position table row r % t. The
/// tables sit in `params` at `tok_off` [vocab, d] and `pos_off` [t, d]
/// (each model's own, under `m`).
#[allow(clippy::too_many_arguments)]
pub fn embed(ids: &Handle, rows: usize, d: usize, t: usize, params: &Handle, tok_off: usize, pos_off: usize, m: Models) -> Handle {
    let y = client().empty(rows * d * 4);
    super::count_launch();
    k_embed::launch(client(), cubes(rows * d), CubeDim::new_1d(EW_DIM), whole(ids), whole(params), whole(&y), rows as u32, d as u32, t as u32, tok_off as u32, pos_off as u32, (rows / m.k) as u32, m.stride as u32);
    y
}

/// Accumulates both tables' gradients into `grads`. Two launches: one
/// unit per token-table element, summing its model's rows that used that
/// token, and one per position-table element, summing over its model's
/// batch.
#[allow(clippy::too_many_arguments)]
pub fn embed_backward(dy: &Handle, ids: &Handle, rows: usize, d: usize, t: usize, vocab: usize, grads: &Handle, tok_off: usize, pos_off: usize, m: Models) {
    let c = client();
    let (rows_pm, stride) = ((rows / m.k) as u32, m.stride as u32);
    super::count_launch();
    k_embed_bwd_tok::launch(c, cubes(m.k * vocab * d), CubeDim::new_1d(EW_DIM), whole(dy), whole(ids), whole(grads), rows_pm, d as u32, vocab as u32, tok_off as u32, m.k as u32, stride);
    super::count_launch();
    k_embed_bwd_pos::launch(c, cubes(m.k * t * d), CubeDim::new_1d(EW_DIM), whole(dy), whole(grads), rows_pm, d as u32, t as u32, pos_off as u32, m.k as u32, stride);
}

/// What CrossEntropy's backward needs, and the loss itself (one f32 per
/// model, the only thing a training step reads back).
#[derive(Clone)]
pub struct CeOut {
    pub loss: Handle,
    pub lse: Handle,
    /// Each row's own loss (for per-row scores, e.g. an ensemble's).
    pub row_loss: Handle,
}

/// Mean over each model's rows of -log softmax(logits)[target], one f32
/// per model (`models` equal slices of the rows). Computed as
/// logsumexp - logit, which differs from `Tape::cross_entropy`'s
/// -log(p + 1e-9) only when p is within a few orders of 1e-9. Two
/// launches: per-row loss, then an ordered sum by one unit per model.
pub fn cross_entropy(logits: &Handle, targets: &Handle, rows: usize, vocab: usize, models: usize) -> CeOut {
    let c = client();
    let out = CeOut { loss: c.empty(models * 4), lse: c.empty(rows * 4), row_loss: c.empty(rows * 4) };
    super::count_launch();
    k_ce_rows::launch(c, cubes(rows), CubeDim::new_1d(EW_DIM), whole(logits), whole(targets), whole(&out.lse), whole(&out.row_loss), rows as u32, vocab as u32);
    super::count_launch();
    k_mean::launch(c, CubeCount::Static(1, 1, 1), CubeDim::new_1d(models as u32), whole(&out.row_loss), whole(&out.loss), (rows / models) as u32);
    out
}

/// dlogits = (softmax(logits) - onehot(target)) / rows per model: the
/// gradient of each model's mean loss. One launch.
///
/// `distill` > 0 (fused models only) adds deep mutual learning's term
/// (Zhang et al. 2018): distill · KL(q ‖ p) per row, q the other models'
/// mean probabilities on the same row, held constant. Its gradient is
/// distill · (p - q), so row r of model m gets
/// (p - onehot + distill · (p - q)) / rows.
#[allow(clippy::too_many_arguments)]
pub fn cross_entropy_backward(logits: &Handle, targets: &Handle, fwd: &CeOut, rows: usize, vocab: usize, models: usize, distill: f32) -> Handle {
    assert!(distill == 0.0 || models > 1, "distillation needs peers");
    let dl = client().empty(rows * vocab * 4);
    super::count_launch();
    k_ce_bwd::launch(client(), cubes(rows * vocab), CubeDim::new_1d(EW_DIM), whole(logits), whole(targets), whole(&fwd.lse), whole(&dl), rows as u32, vocab as u32, (rows / models) as u32, models as u32, distill, distill != 0.0);
    dl
}

#[cube(launch)]
fn k_embed(ids: &[u32], p: &[f32], y: &mut [f32], rows: u32, d: u32, t: u32, tok_off: u32, pos_off: u32, rows_pm: u32, stride: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < rows * d {
        let r = i / d;
        let j = i % d;
        let base = (r / rows_pm) * stride;
        y[i as usize] = p[(base + tok_off + ids[r as usize] * d + j) as usize] + p[(base + pos_off + (r % t) * d + j) as usize];
    }
}

/// One unit per (model, table element); `rows` is per model.
#[cube(launch)]
fn k_embed_bwd_tok(dy: &[f32], ids: &[u32], g: &mut [f32], rows: u32, d: u32, vocab: u32, tok_off: u32, models: u32, stride: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < models * vocab * d {
        let model = i / (vocab * d);
        let e = i % (vocab * d);
        let v = e / d;
        let j = e % d;
        let r0 = model * rows;
        let mut s = 0.0f32;
        for r in 0..rows {
            if ids[(r0 + r) as usize] == v {
                s += dy[((r0 + r) * d + j) as usize];
            }
        }
        g[(model * stride + tok_off + e) as usize] += s;
    }
}

#[cube(launch)]
fn k_embed_bwd_pos(dy: &[f32], g: &mut [f32], rows: u32, d: u32, t: u32, pos_off: u32, models: u32, stride: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < models * t * d {
        let model = i / (t * d);
        let e = i % (t * d);
        let tt = e / d;
        let j = e % d;
        let r0 = model * rows;
        let mut s = 0.0f32;
        for b in 0..rows / t {
            s += dy[((r0 + b * t + tt) * d + j) as usize];
        }
        g[(model * stride + pos_off + e) as usize] += s;
    }
}

#[cube(launch)]
fn k_ce_rows(x: &[f32], targets: &[u32], lse: &mut [f32], loss: &mut [f32], rows: u32, vocab: u32) {
    let r = ABSOLUTE_POS as u32;
    if r < rows {
        let base = r * vocab;
        let mut m = x[base as usize];
        for j in 1..vocab {
            let v = x[(base + j) as usize];
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0f32;
        for j in 0..vocab {
            s += f32::exp(x[(base + j) as usize] - m);
        }
        let l = m + f32::ln(s);
        lse[r as usize] = l;
        loss[r as usize] = l - x[(base + targets[r as usize]) as usize];
    }
}

#[cube(launch)]
fn k_mean(x: &[f32], out: &mut [f32], n: u32) {
    // unit m: the mean of x[m·n .. m·n + n]
    let m = ABSOLUTE_POS as u32;
    let mut s = 0.0f32;
    for i in 0..n {
        s += x[(m * n + i) as usize];
    }
    out[m as usize] = s / f32::cast_from(n);
}

/// `per`: rows per model, the mean's divisor.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_ce_bwd(x: &[f32], targets: &[u32], lse: &[f32], dl: &mut [f32], rows: u32, vocab: u32, per: u32, models: u32, alpha: f32, #[comptime] distill: bool) {
    let i = ABSOLUTE_POS as u32;
    if i < rows * vocab {
        let r = i / vocab;
        let p = f32::exp(x[i as usize] - lse[r as usize]);
        let mut v = p;
        if i % vocab == targets[r as usize] {
            v -= 1.0;
        }
        if distill {
            // q: the other models' mean probability of this column, same row
            let (c, rr) = (i % vocab, r % per);
            let mut q = 0.0f32;
            for j in 0..models {
                let o = j * per + rr;
                if o != r {
                    q += f32::exp(x[(o * vocab + c) as usize] - lse[o as usize]);
                }
            }
            v += alpha * (p - q / f32::cast_from(models - 1));
        }
        dl[i as usize] = v / f32::cast_from(per);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_step::{read, upload};
    use crate::nn::{Embedding, Rng};
    use crate::tape::Tape;
    use crate::tensor::NdArray;

    fn rel_err(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(1e-6f32, |s, v| s.max(v.abs()));
        got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0, f32::max) / scale
    }

    /// Needs the discrete GPU. Token + position embedding against the CPU
    /// tape (two `Embedding::forward`s and an add), with repeated and
    /// unused token ids, the tables at offsets in one flat buffer, and
    /// gradients accumulated onto a non-zero buffer. It only moves and
    /// sums values, so forward is exact and gradients are close.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn embed_matches_cpu_tape() {
        let (vocab, d, t, batch) = (11, 6, 5, 3);
        let rows = batch * t;
        let mut rng = Rng::new(8);
        let (tok, pos) = (Embedding::new(&mut rng, vocab, d), Embedding::new(&mut rng, t, d));
        let ids: Vec<usize> = (0..rows).map(|r| (r * 7) % 9).collect(); // 9 and 10 unused
        let r: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        let positions: Vec<usize> = (0..batch).flat_map(|_| 0..t).collect();
        let mut tape = Tape::new();
        let (to, po) = (tok.forward(&mut tape, &ids), pos.forward(&mut tape, &positions));
        let y = tape.add(to.y, po.y);
        let rv = tape.leaf(NdArray::new(r.clone(), vec![rows, d]));
        let prod = tape.mul(y, rv);
        let loss = tape.sum(prod);
        tape.backward(loss);

        let (tok_off, pos_off) = (3, 3 + vocab * d + 2);
        let mut flat = vec![0.0f32; pos_off + t * d + 1];
        flat[tok_off..tok_off + vocab * d].copy_from_slice(&tok.to_flat());
        flat[pos_off..pos_off + t * d].copy_from_slice(&pos.to_flat());
        let (ph, idh) = (upload(&flat), upload_ids(&ids));
        let yh = embed(&idh, rows, d, t, &ph, tok_off, pos_off, Models::ONE);
        assert_eq!(read(&yh), tape.value(y).data, "forward");

        let grads = upload(&vec![0.25f32; flat.len()]);
        embed_backward(&upload(&r), &idh, rows, d, t, vocab, &grads, tok_off, pos_off, Models::ONE);
        let g = read(&grads);
        let want_tok: Vec<f32> = tape.grad(to.table).unwrap().data.iter().map(|v| v + 0.25).collect();
        let want_pos: Vec<f32> = tape.grad(po.table).unwrap().data.iter().map(|v| v + 0.25).collect();
        assert!(rel_err(&g[tok_off..tok_off + vocab * d], &want_tok) < 1e-5, "token table gradient");
        assert!(rel_err(&g[pos_off..pos_off + t * d], &want_pos) < 1e-5, "position table gradient");
        assert!(g[..tok_off].iter().chain(&g[tok_off + vocab * d..pos_off]).chain(&g[pos_off + t * d..]).all(|&v| v == 0.25), "wrote outside the tables");
    }

    /// Needs the discrete GPU. Loss and dlogits against
    /// `Tape::cross_entropy` and an f64 closed form, then dlogits by
    /// central differences through the GPU loss.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn cross_entropy_matches_cpu_tape() {
        let (rows, vocab) = (19, 13);
        let mut rng = Rng::new(6);
        let x: Vec<f32> = (0..rows * vocab).map(|_| 2.0 * rng.next_gaussian()).collect();
        let targets: Vec<usize> = (0..rows).map(|r| (r * 5) % vocab).collect();
        let mut tape = Tape::new();
        let xv = tape.leaf(NdArray::new(x.clone(), vec![rows, vocab]));
        let loss = tape.cross_entropy(xv, &targets);
        tape.backward(loss);

        let th = upload_ids(&targets);
        let gpu_loss = |xs: &[f32]| read(&cross_entropy(&upload(xs), &th, rows, vocab, 1).loss)[0];
        let xh = upload(&x);
        let fwd = cross_entropy(&xh, &th, rows, vocab, 1);
        let got = read(&fwd.loss)[0];
        let want = tape.value(loss).data[0];
        assert!((got - want).abs() <= 1e-5 * want.abs(), "loss {got} vs {want}");
        let dl = read(&cross_entropy_backward(&xh, &th, &fwd, rows, vocab, 1, 0.0));
        // f64 closed form (softmax - onehot) / rows, the ground truth both
        // are measured against.
        let mut exact = vec![0.0f32; rows * vocab];
        for r in 0..rows {
            let row = &x[r * vocab..(r + 1) * vocab];
            let m = row.iter().fold(f64::MIN, |m, &v| m.max(v as f64));
            let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
            for j in 0..vocab {
                let p = (row[j] as f64 - m).exp() / s;
                exact[r * vocab + j] = ((p - if j == targets[r] { 1.0 } else { 0.0 }) / rows as f64) as f32;
            }
        }
        let want_dl = &tape.grad(xv).unwrap().data;
        assert!(rel_err(&dl, &exact) < 1e-5, "dlogits vs f64: {}", rel_err(&dl, &exact));
        // The tape's -log(p + 1e-9) scales its gradient by p_t / (p_t + 1e-9).
        // These logits reach p_t = 8.3e-6, so the tape sits 1.2e-4 from the
        // closed form (measured; the GPU is 2.8e-7 from it). At the step's
        // scale (p ~ 1/256 at init) the gap is ~3e-7.
        assert!(rel_err(&dl, want_dl) < 2e-4, "dlogits vs tape: {}", rel_err(&dl, want_dl));

        let (h, scale) = (1e-2f32, dl.iter().fold(0.0f32, |s, v| s.max(v.abs())));
        for i in (0..x.len()).step_by(11) {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[i] += h;
            xm[i] -= h;
            let num = (gpu_loss(&xp) - gpu_loss(&xm)) / (2.0 * h);
            assert!((num - dl[i]).abs() <= 2e-2 * scale, "finite difference at {i}: {num} vs gpu {}", dl[i]);
        }
    }

    /// Needs the discrete GPU. Fused models' CE: one mean per model, and
    /// the backward with mutual distillation against an f64 closed form,
    /// (p - onehot + a (p - q)) / rows_pm with q the peers' mean
    /// probabilities. Checked at a 0 (plain, per model) and a 0.7.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn fused_cross_entropy_distills() {
        let (k, per, vocab) = (3, 5, 7);
        let rows = k * per;
        let mut rng = Rng::new(4);
        let x: Vec<f32> = (0..rows * vocab).map(|_| 2.0 * rng.next_gaussian()).collect();
        let targets: Vec<usize> = (0..rows).map(|r| (r * 3) % vocab).collect();
        let p = |r: usize| -> Vec<f64> {
            let row = &x[r * vocab..(r + 1) * vocab];
            let m = row.iter().fold(f64::MIN, |m, &v| m.max(v as f64));
            let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
            row.iter().map(|&v| (v as f64 - m).exp() / s).collect()
        };
        let (xh, th) = (upload(&x), upload_ids(&targets));
        let fwd = cross_entropy(&xh, &th, rows, vocab, k);
        let loss = read(&fwd.loss);
        for (m, &got) in loss[..k].iter().enumerate() {
            let want: f64 = (m * per..(m + 1) * per).map(|r| -p(r)[targets[r]].ln()).sum::<f64>() / per as f64;
            assert!((got as f64 - want).abs() < 1e-5 * want, "model {m} loss {got} vs {want}");
        }
        for a in [0.0f32, 0.7] {
            let dl = read(&cross_entropy_backward(&xh, &th, &fwd, rows, vocab, k, a));
            for r in 0..rows {
                let (m, rr) = (r / per, r % per);
                let pr = p(r);
                for c in 0..vocab {
                    let q: f64 = (0..k).filter(|&j| j != m).map(|j| p(j * per + rr)[c]).sum::<f64>() / (k - 1) as f64;
                    let want = (pr[c] - if c == targets[r] { 1.0 } else { 0.0 } + a as f64 * (pr[c] - q)) / per as f64;
                    assert!((dl[r * vocab + c] as f64 - want).abs() < 1e-6, "a {a} row {r} col {c}: {} vs {want}", dl[r * vocab + c]);
                }
            }
        }
    }
}
