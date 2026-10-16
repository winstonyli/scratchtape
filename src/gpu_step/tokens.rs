//! The two ops indexed by token id (milestone 2): Embed (token + position,
//! summed) and CrossEntropy (fused softmax + NLL, mean over rows). Their
//! reductions are ordered loops, not atomics, so they're deterministic.
use super::tape::dropout_bits;
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
    k_embed::launch(
        client(),
        cubes(rows * d),
        CubeDim::new_1d(EW_DIM),
        whole(ids),
        whole(params),
        whole(&y),
        rows as u32,
        d as u32,
        t as u32,
        tok_off as u32,
        pos_off as u32,
        (rows / m.k) as u32,
        m.stride as u32,
    );
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
    k_embed_bwd_tok::launch(
        c,
        CubeCount::Static((vocab * d).div_ceil(EW_DIM as usize) as u32, m.k as u32, 1),
        CubeDim::new_1d(EW_DIM),
        whole(dy),
        whole(ids),
        whole(grads),
        rows_pm,
        d as u32,
        vocab as u32,
        tok_off as u32,
        stride,
    );
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
    k_ce_bwd::launch(
        client(),
        cubes(rows * vocab),
        CubeDim::new_1d(EW_DIM),
        whole(logits),
        whole(targets),
        whole(&fwd.lse),
        whole(&dl),
        rows as u32,
        vocab as u32,
        (rows / models) as u32,
        models as u32,
        distill,
        distill != 0.0,
    );
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

/// Ids staged in shared memory per pass of `k_embed_bwd_tok`.
const ID_CHUNK: u32 = 512;

/// One unit per (model, table element) (cube y = model, `rows` per model).
/// Every unit scans all the model's ids in row order; the cube stages each
/// chunk of ids in shared memory, so the scan is not a chain of dependent
/// global loads (it was 0.43 ms for 65k units x 2048 rows).
#[cube(launch)]
fn k_embed_bwd_tok(dy: &[f32], ids: &[u32], g: &mut [f32], rows: u32, d: u32, vocab: u32, tok_off: u32, stride: u32) {
    let mut sh = Shared::<[u32]>::new_slice(512usize);
    let model = CUBE_POS_Y;
    let e = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let live = e < vocab * d;
    let v = e / d;
    let j = e % d;
    let r0 = model * rows;
    let mut s = 0.0f32;
    let mut c0 = 0u32;
    while c0 < rows {
        let n = u32::min(ID_CHUNK, rows - c0);
        let mut q = UNIT_POS_X;
        while q < n {
            sh[q as usize] = ids[(r0 + c0 + q) as usize];
            q += CUBE_DIM_X;
        }
        sync_cube();
        if live {
            for q in 0..n {
                if sh[q as usize] == v {
                    s += dy[((r0 + c0 + q) * d + j) as usize];
                }
            }
        }
        sync_cube();
        c0 += ID_CHUNK;
    }
    if live {
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

/// What `memory_cross_entropy`'s backward needs (the plain `CeOut` plus the numerator's log-mass per row).
#[derive(Clone)]
pub struct MemCeOut {
    pub ce: CeOut,
    pub lnum: Handle,
}

/// Where a memory-augmented row sits: `book` ids and absolute byte `pos` per row, windows of `t` rows, groups of `gm` rows
/// (the memory is the group's rows), `thr`/`seed` the local-memory drop (`mem_visible`).
pub struct MemMeta<'a> {
    pub book: &'a Handle,
    pub pos: &'a Handle,
    pub t: usize,
    pub gm: usize,
    pub scale: f32,
    pub thr: u32,
    pub seed: u32,
}

/// TRIME's loss (Zhong et al. 2022): -ln of the target's mass in softmax over [logits | scale·scores[r, j] for visible j], where
/// column j is a memory entry whose token is targets[j] and the target's mass is its logit plus every visible entry with the same
/// token. `scores` [rows, gm]: row r's similarities to its group's gm rows. One model only.
pub fn memory_cross_entropy(logits: &Handle, scores: &Handle, targets: &Handle, m: &MemMeta, rows: usize, vocab: usize) -> MemCeOut {
    let c = client();
    let out = MemCeOut { ce: CeOut { loss: c.empty(4), lse: c.empty(rows * 4), row_loss: c.empty(rows * 4) }, lnum: c.empty(rows * 4) };
    super::count_launch();
    k_mce_rows::launch(
        c,
        cubes(rows),
        CubeDim::new_1d(EW_DIM),
        whole(logits),
        whole(scores),
        whole(targets),
        whole(m.book),
        whole(m.pos),
        whole(&out.ce.lse),
        whole(&out.lnum),
        whole(&out.ce.row_loss),
        rows as u32,
        vocab as u32,
        m.gm as u32,
        m.t as u32,
        m.scale,
        m.thr,
        m.seed,
    );
    super::count_launch();
    k_mean::launch(c, CubeCount::Static(1, 1, 1), CubeDim::new_1d(1), whole(&out.ce.row_loss), whole(&out.ce.loss), rows as u32);
    out
}

/// (dlogits, dscores) of the mean loss: dlogits = (p - onehot(target)·w) / rows, with w the target logit's share of the
/// numerator; dscores[r, j] = scale · (p_j - [targets[j] = targets[r]]·share_j) / rows for visible j, else 0.
pub fn memory_cross_entropy_backward(logits: &Handle, scores: &Handle, targets: &Handle, m: &MemMeta, fwd: &MemCeOut, rows: usize, vocab: usize) -> (Handle, Handle) {
    let c = client();
    let (dl, ds) = (c.empty(rows * vocab * 4), c.empty(rows * m.gm * 4));
    super::count_launch();
    k_mce_bwd_logits::launch(c, cubes(rows * vocab), CubeDim::new_1d(EW_DIM), whole(logits), whole(targets), whole(&fwd.ce.lse), whole(&fwd.lnum), whole(&dl), rows as u32, vocab as u32);
    super::count_launch();
    k_mce_bwd_scores::launch(
        c,
        cubes(rows * m.gm),
        CubeDim::new_1d(EW_DIM),
        whole(scores),
        whole(targets),
        whole(m.book),
        whole(m.pos),
        whole(&fwd.ce.lse),
        whole(&fwd.lnum),
        whole(&ds),
        rows as u32,
        m.gm as u32,
        m.t as u32,
        m.scale,
        m.thr,
        m.seed,
    );
    (dl, ds)
}

/// 1 if memory row `j` is visible to row `r`: another book always is; the same book if the entry is earlier in the text
/// (so its token, the byte after it, is no later than r's own input), except that for the same window the whole local memory
/// is dropped with probability thr / 2³² (TRIME drops local memory so the model learns to use the long-term and external one).
#[cube]
fn mem_visible(r: u32, j: u32, book: &[u32], pos: &[u32], t: u32, thr: u32, seed: u32) -> u32 {
    let mut v = 0u32;
    if book[j as usize] != book[r as usize] {
        v = 1u32;
    } else if pos[j as usize] < pos[r as usize] {
        v = 1u32;
        if j / t == r / t && dropout_bits(r, seed) < thr {
            v = 0u32;
        }
    }
    v
}

#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_mce_rows(
    x: &[f32],
    s: &[f32],
    targets: &[u32],
    book: &[u32],
    pos: &[u32],
    lse: &mut [f32],
    lnum: &mut [f32],
    loss: &mut [f32],
    rows: u32,
    vocab: u32,
    gm: u32,
    t: u32,
    scale: f32,
    thr: u32,
    seed: u32,
) {
    let r = ABSOLUTE_POS as u32;
    if r < rows {
        let base = r * vocab;
        let g0 = (r / gm) * gm;
        let tg = targets[r as usize];
        let mut m = x[base as usize];
        for c in 1..vocab {
            let v = x[(base + c) as usize];
            if v > m {
                m = v;
            }
        }
        for j in 0..gm {
            if mem_visible(r, g0 + j, book, pos, t, thr, seed) == 1u32 {
                let v = s[(r * gm + j) as usize] * scale;
                if v > m {
                    m = v;
                }
            }
        }
        let mut z = 0.0f32;
        for c in 0..vocab {
            z += f32::exp(x[(base + c) as usize] - m);
        }
        let mut n = f32::exp(x[(base + tg) as usize] - m);
        for j in 0..gm {
            if mem_visible(r, g0 + j, book, pos, t, thr, seed) == 1u32 {
                let e = f32::exp(s[(r * gm + j) as usize] * scale - m);
                z += e;
                if targets[(g0 + j) as usize] == tg {
                    n += e;
                }
            }
        }
        let l = m + f32::ln(z);
        let ln = m + f32::ln(n);
        lse[r as usize] = l;
        lnum[r as usize] = ln;
        loss[r as usize] = l - ln;
    }
}

#[cube(launch)]
fn k_mce_bwd_logits(x: &[f32], targets: &[u32], lse: &[f32], lnum: &[f32], dl: &mut [f32], rows: u32, vocab: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < rows * vocab {
        let r = i / vocab;
        let mut v = f32::exp(x[i as usize] - lse[r as usize]);
        if i % vocab == targets[r as usize] {
            v -= f32::exp(x[i as usize] - lnum[r as usize]);
        }
        dl[i as usize] = v / f32::cast_from(rows);
    }
}

#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_mce_bwd_scores(s: &[f32], targets: &[u32], book: &[u32], pos: &[u32], lse: &[f32], lnum: &[f32], ds: &mut [f32], rows: u32, gm: u32, t: u32, scale: f32, thr: u32, seed: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < rows * gm {
        let r = i / gm;
        let jr = (r / gm) * gm + i % gm;
        let mut v = 0.0f32;
        if mem_visible(r, jr, book, pos, t, thr, seed) == 1u32 {
            let sc = s[i as usize] * scale;
            v = f32::exp(sc - lse[r as usize]);
            if targets[jr as usize] == targets[r as usize] {
                v -= f32::exp(sc - lnum[r as usize]);
            }
            v = v * scale / f32::cast_from(rows);
        }
        ds[i as usize] = v;
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
        // 750 rows (two id chunks) and 330 table elements (two cubes)
        let (vocab, d, t, batch) = (11, 30, 5, 150);
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

    /// f64 reference of `memory_cross_entropy`'s mean loss, written from the definition.
    #[allow(clippy::too_many_arguments)]
    fn mce_reference(x: &[f64], s: &[f64], targets: &[usize], book: &[usize], pos: &[usize], t: usize, gm: usize, scale: f64, vocab: usize, visible: &dyn Fn(usize, usize) -> bool) -> f64 {
        let rows = targets.len();
        let _ = (book, pos, t);
        let mut total = 0.0;
        for r in 0..rows {
            let g0 = r / gm * gm;
            let mut z: f64 = (0..vocab).map(|c| x[r * vocab + c].exp()).sum();
            let mut num = x[r * vocab + targets[r]].exp();
            for j in 0..gm {
                if visible(r, g0 + j) {
                    let e = (s[r * gm + j] * scale).exp();
                    z += e;
                    if targets[g0 + j] == targets[r] {
                        num += e;
                    }
                }
            }
            total += z.ln() - num.ln();
        }
        total / rows as f64
    }

    /// Needs the discrete GPU. TRIME's memory loss and its two gradients against an f64 reference (forward, and central
    /// differences through the reference for dlogits and dscores), and with no memory visible, plain cross-entropy exactly.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn memory_cross_entropy_matches_reference() {
        let (t, gm, groups, vocab) = (4, 12, 2, 9);
        let rows = gm * groups;
        let mut rng = Rng::new(12);
        let x: Vec<f32> = (0..rows * vocab).map(|_| rng.next_gaussian()).collect();
        let s: Vec<f32> = (0..rows * gm).map(|_| 2.0 * rng.next_gaussian()).collect();
        // three windows per group: windows 0 and 1 of book 0 (consecutive), window 2 of book 1
        let book: Vec<usize> = (0..rows).map(|r| if (r % gm) / t == 2 { 1 } else { 0 }).collect();
        let pos: Vec<usize> = (0..rows).map(|r| 100 * (r / gm) + (r % gm) % (2 * t) + if (r % gm) / t == 2 { 7 } else { 0 }).collect();
        let targets: Vec<usize> = (0..rows).map(|r| (r * 5 + r / 3) % vocab).collect();
        let (bh, ph, th, xh, sh) = (upload_ids(&book), upload_ids(&pos), upload_ids(&targets), upload(&x), upload(&s));
        let scale = 0.35f32;
        // drop = 0 (thr 0): local memory kept. Visible: other book, or same book and earlier position.
        let visible = |r: usize, j: usize| book[j] != book[r] || pos[j] < pos[r];
        let meta = MemMeta { book: &bh, pos: &ph, t, gm, scale, thr: 0, seed: 1 };
        let fwd = memory_cross_entropy(&xh, &sh, &th, &meta, rows, vocab);
        let (xd, sd): (Vec<f64>, Vec<f64>) = (x.iter().map(|&v| v as f64).collect(), s.iter().map(|&v| v as f64).collect());
        let f = |xs: &[f64], ss: &[f64]| mce_reference(xs, ss, &targets, &book, &pos, t, gm, scale as f64, vocab, &visible);
        let want = f(&xd, &sd);
        let got = read(&fwd.ce.loss)[0] as f64;
        assert!((got - want).abs() < 1e-5 * want.abs(), "loss {got} vs {want}");
        let (dl, ds) = memory_cross_entropy_backward(&xh, &sh, &th, &meta, &fwd, rows, vocab);
        let (dl, ds) = (read(&dl), read(&ds));
        let h = 1e-5;
        for i in (0..x.len()).step_by(7) {
            let (mut p, mut m) = (xd.clone(), xd.clone());
            p[i] += h;
            m[i] -= h;
            let num = (f(&p, &sd) - f(&m, &sd)) / (2.0 * h);
            assert!((num - dl[i] as f64).abs() < 1e-5, "dlogits[{i}]: {num} vs {}", dl[i]);
        }
        let mut nonzero = 0;
        for i in (0..s.len()).step_by(5) {
            let (mut p, mut m) = (sd.clone(), sd.clone());
            p[i] += h;
            m[i] -= h;
            let num = (f(&xd, &p) - f(&xd, &m)) / (2.0 * h);
            assert!((num - ds[i] as f64).abs() < 1e-5, "dscores[{i}]: {num} vs {}", ds[i]);
            nonzero += (ds[i] != 0.0) as usize;
        }
        assert!(nonzero > 5, "the test should exercise visible entries");

        // Nothing visible (one book, equal positions): plain cross-entropy.
        let (one, same) = (upload_ids(&vec![0; rows]), upload_ids(&vec![5; rows]));
        let none = MemMeta { book: &one, pos: &same, t, gm, scale, thr: 0, seed: 1 };
        let m0 = memory_cross_entropy(&xh, &sh, &th, &none, rows, vocab);
        let plain = cross_entropy(&xh, &th, rows, vocab, 1);
        assert_eq!(read(&m0.ce.loss), read(&plain.loss), "no memory: loss is plain CE");
        let (dl0, ds0) = memory_cross_entropy_backward(&xh, &sh, &th, &none, &m0, rows, vocab);
        let dplain = read(&cross_entropy_backward(&xh, &th, &plain, rows, vocab, 1, 0.0));
        assert!(rel_err(&read(&dl0), &dplain) < 1e-6, "no memory: dlogits is plain CE's");
        assert!(read(&ds0).iter().all(|&v| v == 0.0));
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
