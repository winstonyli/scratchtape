//! The device tape (milestones 3 and 4): the design's option C. Each op is
//! a coarse, layer-level step whose forward and backward are milestone-2
//! kernels; the tape records ops as they run, and `backward` walks it in
//! reverse, as `crate::tape::Tape` does.
//! - Values live on the device; nothing is read back except what a caller
//!   asks for (the loss).
//! - Parameter gradients accumulate into the flat gradient buffer
//!   (`DeviceParams::grads`, zeroed once per step).
//! - Activation gradients are allocated on first contribution. A residual
//!   passes its gradient through by aliasing the buffer: in reverse order
//!   every consumer of that buffer has already run, so later accumulation
//!   into it can't corrupt anything still needed.
use super::heads::{merge_heads, split_heads};
use super::matmul::{Epilogue, MatRef, matmul};
use super::rows::{LnOut, col_sum, layer_norm, layer_norm_backward, softmax, softmax_backward};
use super::tokens::{CeOut, cross_entropy, cross_entropy_backward, embed, embed_backward, upload_ids};
use super::{DeviceParams, EW_DIM, buf, client, k_fill};
use cubecl::prelude::*;
use cubecl::server::Handle;

/// LayerNorm's eps, as `nn::LayerNorm::new`.
const LN_EPS: f32 = 1e-5;

/// A node on the device tape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DVar(usize);

#[derive(Clone)]
enum Op {
    Embed { ids: Handle, t: usize, vocab: usize, tok_off: usize, pos_off: usize },
    LayerNorm { x: usize, fwd: LnOut, off: usize },
    /// y = act(x @ W + b) (+ residual). W [inp, out] at w_off, b right after.
    Linear { x: usize, inp: usize, w_off: usize, relu: bool, residual: Option<usize> },
    /// Columns col0.. of x [B·T, cols] as [B·H·T, width].
    SplitHeads { x: usize, cols: usize, col0: usize, heads: usize, width: usize, batch: usize, t: usize },
    MergeHeads { x: usize, heads: usize, width: usize, batch: usize, t: usize },
    /// c[z] = a[z] @ b[z] (b stored transposed if trans_b), z < batch.
    BatchedMatmul { a: usize, b: usize, batch: usize, m: usize, k: usize, n: usize, trans_b: bool },
    Softmax { x: usize, scale: f32 },
    CrossEntropy { logits: usize, targets: Handle, fwd: CeOut, vocab: usize },
}

struct Node {
    value: Handle,
    rows: usize,
    cols: usize,
    grad: Option<Handle>,
    op: Op,
}

pub struct DeviceTape<'p> {
    params: &'p DeviceParams,
    nodes: Vec<Node>,
}

impl<'p> DeviceTape<'p> {
    pub fn new(params: &'p DeviceParams) -> Self {
        Self { params, nodes: Vec::new() }
    }

    pub fn value(&self, v: DVar) -> &Handle {
        &self.nodes[v.0].value
    }

    /// Number of recorded ops.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn push(&mut self, value: Handle, rows: usize, cols: usize, op: Op) -> DVar {
        self.nodes.push(Node { value, rows, cols, grad: None, op });
        DVar(self.nodes.len() - 1)
    }

    fn shape(&self, v: DVar) -> (usize, usize) {
        (self.nodes[v.0].rows, self.nodes[v.0].cols)
    }

    /// Token + position embedding of `ids` (B·T rows); tables at `tok_off`
    /// [vocab, d] and `pos_off` [t, d].
    pub fn embed(&mut self, ids: &[usize], d: usize, t: usize, vocab: usize, tok_off: usize, pos_off: usize) -> DVar {
        let rows = ids.len();
        let ids = upload_ids(ids);
        let y = embed(&ids, rows, d, t, &self.params.params, tok_off, pos_off);
        self.push(y, rows, d, Op::Embed { ids, t, vocab, tok_off, pos_off })
    }

    /// LayerNorm with gamma/beta at `off` (`LayerNorm::to_flat` order).
    pub fn layer_norm(&mut self, x: DVar, off: usize) -> DVar {
        let (rows, d) = self.shape(x);
        let fwd = layer_norm(self.value(x), rows, d, &self.params.params, off, LN_EPS);
        let y = fwd.y.clone();
        self.push(y, rows, d, Op::LayerNorm { x: x.0, fwd, off })
    }

    /// act(x @ W + b) (+ residual), W [inp, out] at `w_off` and b after it
    /// (`Linear::to_flat` order). One launch.
    pub fn linear(&mut self, x: DVar, w_off: usize, out: usize, relu: bool, residual: Option<DVar>) -> DVar {
        let (rows, inp) = self.shape(x);
        let y = client().empty(rows * out * 4);
        let p = &self.params.params;
        let epi = Epilogue { bias: Some((p, w_off + inp * out)), relu, residual: residual.map(|r| (self.value(r), 0)), ..Default::default() };
        let w = MatRef { h: p, off: w_off, stride: 0, trans: false };
        matmul(MatRef::new(self.value(x)), w, MatRef::new(&y), 1, rows, inp, out, epi);
        self.push(y, rows, out, Op::Linear { x: x.0, inp, w_off, relu, residual: residual.map(|r| r.0) })
    }

    pub fn split_heads(&mut self, x: DVar, col0: usize, heads: usize, width: usize, batch: usize) -> DVar {
        let (rows, cols) = self.shape(x);
        let t = rows / batch;
        let y = split_heads(self.value(x), cols, col0, heads, width, batch, t);
        self.push(y, rows * heads, width, Op::SplitHeads { x: x.0, cols, col0, heads, width, batch, t })
    }

    pub fn merge_heads(&mut self, x: DVar, heads: usize, batch: usize) -> DVar {
        let (rows, width) = self.shape(x);
        let t = rows / (batch * heads);
        let y = client().empty(rows * width * 4);
        merge_heads(self.value(x), &y, heads * width, 0, heads, width, batch, t, false);
        self.push(y, batch * t, heads * width, Op::MergeHeads { x: x.0, heads, width, batch, t })
    }

    /// a and b hold `batch` stacked matrices: a [batch·m, k], b [batch·k, n]
    /// or, with trans_b, [batch·n, k]. Out [batch·m, n].
    pub fn batched_matmul(&mut self, a: DVar, b: DVar, batch: usize, trans_b: bool) -> DVar {
        let ((ar, k), (br, bc)) = (self.shape(a), self.shape(b));
        let m = ar / batch;
        let n = if trans_b { br / batch } else { bc };
        let y = client().empty(batch * m * n * 4);
        let bref = MatRef { h: self.value(b), off: 0, stride: k * n, trans: trans_b };
        matmul(MatRef { h: self.value(a), off: 0, stride: m * k, trans: false }, bref, MatRef { h: &y, off: 0, stride: m * n, trans: false }, batch, m, k, n, Epilogue::default());
        self.push(y, batch * m, n, Op::BatchedMatmul { a: a.0, b: b.0, batch, m, k, n, trans_b })
    }

    /// Causal softmax (or softmax1) of `scale * x` over attention scores
    /// [B·H·T, T].
    pub fn causal_softmax(&mut self, x: DVar, scale: f32, one: bool) -> DVar {
        let (rows, n) = self.shape(x);
        let y = softmax(self.value(x), rows, n, n, scale, true, one);
        self.push(y, rows, n, Op::Softmax { x: x.0, scale })
    }

    /// Mean cross-entropy; the value is one f32.
    pub fn cross_entropy(&mut self, logits: DVar, targets: &[usize]) -> DVar {
        let (rows, vocab) = self.shape(logits);
        let targets = upload_ids(targets);
        let fwd = cross_entropy(self.value(logits), &targets, rows, vocab);
        let loss = fwd.loss.clone();
        self.push(loss, 1, 1, Op::CrossEntropy { logits: logits.0, targets, fwd, vocab })
    }

    /// Where a gradient for node v goes: its existing buffer (accumulate),
    /// or a new one. `full_write`: the caller writes every element, so a
    /// new buffer needn't be zeroed first.
    fn grad_slot(&mut self, v: usize, full_write: bool) -> (Handle, bool) {
        if let Some(g) = &self.nodes[v].grad {
            return (g.clone(), true);
        }
        let len = self.nodes[v].rows * self.nodes[v].cols;
        let g = client().empty(len * 4);
        if !full_write {
            super::count_launch();
            k_fill::launch(client(), CubeCount::Static((len as u32).div_ceil(EW_DIM), 1, 1), CubeDim::new_1d(EW_DIM), buf(&g, len), 0.0f32, len as u32);
        }
        self.nodes[v].grad = Some(g.clone());
        (g, !full_write)
    }

    /// Adds a finished gradient buffer into node v's: aliases it if v has
    /// none yet (see the module doc), otherwise one add launch.
    fn add_grad(&mut self, v: usize, g: Handle) {
        match &self.nodes[v].grad {
            None => self.nodes[v].grad = Some(g),
            Some(dst) => {
                let len = self.nodes[v].rows * self.nodes[v].cols;
                super::count_launch();
                k_add_into::launch(client(), CubeCount::Static((len as u32).div_ceil(EW_DIM), 1, 1), CubeDim::new_1d(EW_DIM), buf(dst, len), buf(&g, len), len as u32);
            }
        }
    }

    /// Backpropagates from `loss` (a cross_entropy node, seed 1).
    /// Parameter gradients accumulate into the params' gradient buffer.
    pub fn backward(&mut self, loss: DVar) {
        assert!(matches!(self.nodes[loss.0].op, Op::CrossEntropy { .. }), "backward starts at a cross_entropy node");
        let grads = self.params.grads.clone();
        let p = self.params.params.clone();
        for i in (0..=loss.0).rev() {
            let (op, rows, cols) = (self.nodes[i].op.clone(), self.nodes[i].rows, self.nodes[i].cols);
            let dy = match (&op, &self.nodes[i].grad) {
                (Op::CrossEntropy { .. }, _) => None,
                (_, Some(g)) => Some(g.clone()),
                (_, None) => continue, // nothing downstream used it
            };
            match op {
                Op::CrossEntropy { logits, targets, fwd, vocab } => {
                    let r = self.nodes[logits].rows;
                    let dl = cross_entropy_backward(&self.nodes[logits].value, &targets, &fwd, r, vocab);
                    self.add_grad(logits, dl);
                }
                Op::Embed { ids, t, vocab, tok_off, pos_off } => {
                    embed_backward(&dy.unwrap(), &ids, rows, cols, t, vocab, &grads, tok_off, pos_off);
                }
                Op::LayerNorm { x, fwd, off } => {
                    let (dx, acc) = self.grad_slot(x, true);
                    layer_norm_backward(&dy.unwrap(), &self.nodes[x].value, &fwd, rows, cols, &p, &grads, off, &dx, acc);
                }
                Op::Linear { x, inp, w_off, relu, residual } => {
                    let mut dz = dy.unwrap();
                    if relu {
                        let masked = client().empty(rows * cols * 4);
                        let len = rows * cols;
                        super::count_launch();
                        k_relu_mask::launch(client(), CubeCount::Static((len as u32).div_ceil(EW_DIM), 1, 1), CubeDim::new_1d(EW_DIM), buf(&dz, len), buf(&self.nodes[i].value, len), buf(&masked, len), len as u32);
                        dz = masked;
                    }
                    let out = cols;
                    let xv = self.nodes[x].value.clone();
                    // dW += xᵀ dz, db += colsum(dz)
                    matmul(MatRef { h: &xv, off: 0, stride: 0, trans: true }, MatRef::new(&dz), MatRef { h: &grads, off: w_off, stride: 0, trans: false }, 1, inp, rows, out, Epilogue { accumulate: true, ..Default::default() });
                    col_sum(&dz, rows, out, &grads, w_off + inp * out);
                    // dx (+)= dz Wᵀ
                    let (dx, acc) = self.grad_slot(x, true);
                    matmul(MatRef::new(&dz), MatRef { h: &p, off: w_off, stride: 0, trans: true }, MatRef::new(&dx), 1, rows, out, inp, Epilogue { accumulate: acc, ..Default::default() });
                    if let Some(r) = residual {
                        self.add_grad(r, dz);
                    }
                }
                Op::SplitHeads { x, cols: xc, col0, heads, width, batch, t } => {
                    let (dx, acc) = self.grad_slot(x, false);
                    merge_heads(&dy.unwrap(), &dx, xc, col0, heads, width, batch, t, acc);
                }
                Op::MergeHeads { x, heads, width, batch, t } => {
                    let dx = split_heads(&dy.unwrap(), heads * width, 0, heads, width, batch, t);
                    self.add_grad(x, dx);
                }
                Op::BatchedMatmul { a, b, batch, m, k, n, trans_b } => {
                    let dc = dy.unwrap();
                    let (av, bv) = (self.nodes[a].value.clone(), self.nodes[b].value.clone());
                    let dcr = MatRef { h: &dc, off: 0, stride: m * n, trans: false };
                    let (da, acc_a) = self.grad_slot(a, true);
                    let (db, acc_b) = self.grad_slot(b, true);
                    let da_ref = MatRef { h: &da, off: 0, stride: m * k, trans: false };
                    let db_ref = MatRef { h: &db, off: 0, stride: k * n, trans: false };
                    if trans_b {
                        // c = a bᵀ, b [n, k]: da = dc b, db = dcᵀ a
                        matmul(dcr, MatRef { h: &bv, off: 0, stride: n * k, trans: false }, da_ref, batch, m, n, k, Epilogue { accumulate: acc_a, ..Default::default() });
                        matmul(MatRef { trans: true, ..dcr }, MatRef { h: &av, off: 0, stride: m * k, trans: false }, db_ref, batch, n, m, k, Epilogue { accumulate: acc_b, ..Default::default() });
                    } else {
                        // c = a b, b [k, n]: da = dc bᵀ, db = aᵀ dc
                        matmul(dcr, MatRef { h: &bv, off: 0, stride: k * n, trans: true }, da_ref, batch, m, n, k, Epilogue { accumulate: acc_a, ..Default::default() });
                        matmul(MatRef { h: &av, off: 0, stride: m * k, trans: true }, dcr, db_ref, batch, k, m, n, Epilogue { accumulate: acc_b, ..Default::default() });
                    }
                }
                Op::Softmax { x, scale } => {
                    let dx = softmax_backward(&dy.unwrap(), &self.nodes[i].value, rows, cols, scale);
                    self.add_grad(x, dx);
                }
            }
        }
    }
}

/// Where each parameter tensor sits in the flat buffer `gpu_step::pack`
/// builds, for a plain tiny_lm.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub vocab: usize,
    pub d: usize,
    pub heads: usize,
    pub d_ff: usize,
    pub t: usize,
    pub n_blocks: usize,
    pub softmax1: bool,
}

impl Config {
    pub fn tok_off(&self) -> usize {
        0
    }
    pub fn pos_off(&self) -> usize {
        self.vocab * self.d
    }
    fn block_len(&self) -> usize {
        let (d, f) = (self.d, self.d_ff);
        2 * d + (d * 3 * d + 3 * d) + (d * d + d) + 2 * d + (d * f + f) + (f * d + d)
    }
    pub fn block_off(&self, i: usize) -> usize {
        self.pos_off() + self.t * self.d + i * self.block_len()
    }
    pub fn final_ln_off(&self) -> usize {
        self.block_off(self.n_blocks)
    }
    pub fn proj_off(&self) -> usize {
        self.final_ln_off() + 2 * self.d
    }
    /// Total parameter count; equals `pack(..).len()`.
    pub fn len(&self) -> usize {
        self.proj_off() + self.d * self.vocab + self.vocab
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One transformer block, as `TransformerBlock::forward_full` without the
/// extras (QK-norm, gate, sinks are out of scope for v1). `off` is the
/// block's `to_flat` start. 13 ops.
pub fn block_forward(tape: &mut DeviceTape, cfg: &Config, x: DVar, off: usize, batch: usize) -> DVar {
    let (d, h, f) = (cfg.d, cfg.heads, cfg.d_ff);
    let dk = d / h;
    let qkv_off = off + 2 * d;
    let out_off = qkv_off + d * 3 * d + 3 * d;
    let ln2_off = out_off + d * d + d;
    let ffn1_off = ln2_off + 2 * d;
    let ffn2_off = ffn1_off + d * f + f;
    let ln1 = tape.layer_norm(x, off);
    let qkv = tape.linear(ln1, qkv_off, 3 * d, false, None);
    let q = tape.split_heads(qkv, 0, h, dk, batch);
    let k = tape.split_heads(qkv, d, h, dk, batch);
    let v = tape.split_heads(qkv, 2 * d, h, dk, batch);
    let scores = tape.batched_matmul(q, k, batch * h, true);
    let weights = tape.causal_softmax(scores, 1.0 / (dk as f32).sqrt(), cfg.softmax1);
    let heads = tape.batched_matmul(weights, v, batch * h, false);
    let merged = tape.merge_heads(heads, h, batch);
    let x1 = tape.linear(merged, out_off, d, false, Some(x));
    let ln2 = tape.layer_norm(x1, ln2_off);
    let hidden = tape.linear(ln2, ffn1_off, f, true, None);
    tape.linear(hidden, ffn2_off, d, false, Some(x1))
}

/// The whole model: embedding, blocks, final LayerNorm, output projection,
/// cross-entropy. Returns (each block's output, logits, loss).
pub fn model_forward(tape: &mut DeviceTape, cfg: &Config, ids: &[usize], targets: &[usize], batch: usize) -> (Vec<DVar>, DVar, DVar) {
    let mut x = tape.embed(ids, cfg.d, cfg.t, cfg.vocab, cfg.tok_off(), cfg.pos_off());
    let mut outs = Vec::with_capacity(cfg.n_blocks);
    for i in 0..cfg.n_blocks {
        x = block_forward(tape, cfg, x, cfg.block_off(i), batch);
        outs.push(x);
    }
    let ln = tape.layer_norm(x, cfg.final_ln_off());
    let logits = tape.linear(ln, cfg.proj_off(), cfg.vocab, false, None);
    let loss = tape.cross_entropy(logits, targets);
    (outs, logits, loss)
}

#[cube(launch)]
fn k_add_into(dst: &mut [f32], src: &[f32], len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        dst[i] += src[i];
    }
}

/// dz = dy where y > 0, else 0: ReLU's backward from its output.
#[cube(launch)]
fn k_relu_mask(dy: &[f32], y: &[f32], dz: &mut [f32], len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        let mut v = 0.0f32;
        if y[i] > 0.0 {
            v = dy[i];
        }
        dz[i] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_step::{pack, read};
    use crate::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
    use crate::tape::{Tape, Var};

    /// Relative to each parameter tensor's (or activation's) own scale.
    fn rel_err(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(1e-12f32, |s, v| s.max(v.abs()));
        got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0, f32::max) / scale
    }

    /// Needs the discrete GPU. Milestones 3 and 4: one training step's
    /// forward and backward on the device tape against the CPU tape, same
    /// init and batch, plain softmax and softmax1. Every block's output,
    /// the logits and the loss must match to 1e-4 relative (milestone 3),
    /// and so must every parameter tensor's gradient, each against its own
    /// scale (milestone 4). Small ragged shapes first, then the real
    /// tiny_lm step (batch 8, d 128, 8 heads, T 64, 4 blocks).
    ///
    /// At real size about one FFN pre-activation a step lands within
    /// rounding of 0, and the two tapes' ReLUs can disagree on it; that
    /// flips one hidden unit's whole gradient (milestone 5). So each config
    /// takes the first seed from 12 whose ReLUs agree everywhere, and every
    /// disagreement on the seeds it skips must be such a tie.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn device_step_matches_cpu_tape() {
        let small = Config { vocab: 21, d: 16, heads: 4, d_ff: 24, t: 6, n_blocks: 2, softmax1: false };
        let real = Config { vocab: 256, d: 128, heads: 8, d_ff: 256, t: 64, n_blocks: 4, softmax1: false };
        for (cfg, batch) in [(small, 3), (Config { softmax1: true, ..small }, 3), (real, 8), (Config { softmax1: true, ..real }, 8)] {
            let seed = (12..20).find(|&seed| step_case(cfg, batch, seed));
            assert!(seed.is_some(), "{cfg:?}: no seed in 12..20 without a ReLU tie");
        }
    }

    /// One step_case of device_step_matches_cpu_tape. False (having checked
    /// the forward) if the ReLUs disagree anywhere.
    fn step_case(cfg: Config, batch: usize, seed: u64) -> bool {
        {
            let what = format!("{cfg:?} batch {batch} seed {seed}");
            let mut rng = Rng::new(seed);
            let tok = Embedding::new(&mut rng, cfg.vocab, cfg.d);
            let pos = Embedding::new(&mut rng, cfg.t, cfg.d);
            let blocks: Vec<_> = (0..cfg.n_blocks).map(|_| TransformerBlock::new(&mut rng, cfg.d, cfg.heads, cfg.d_ff)).collect();
            let mut final_ln = LayerNorm::new(cfg.d);
            // Non-trivial final LayerNorm, so gamma/beta mistakes show.
            final_ln.gamma.data = (0..cfg.d).map(|j| 1.0 + 0.05 * (j % 7) as f32).collect();
            final_ln.beta.data = (0..cfg.d).map(|j| 0.02 * (j % 5) as f32 - 0.04).collect();
            let proj = Linear::new(&mut rng, cfg.d, cfg.vocab);
            let rows = batch * cfg.t;
            let ids: Vec<usize> = (0..rows).map(|_| (rng.next_gaussian().abs() * 1e4) as usize % cfg.vocab).collect();
            let targets: Vec<usize> = (0..rows).map(|_| (rng.next_gaussian().abs() * 1e4) as usize % cfg.vocab).collect();

            // CPU reference, as tiny_lm's forward.
            let mut tape = Tape::new();
            let positions: Vec<usize> = (0..batch).flat_map(|_| 0..cfg.t).collect();
            let (to, po) = (tok.forward(&mut tape, &ids), pos.forward(&mut tape, &positions));
            let mut x = tape.add(to.y, po.y);
            let mut bouts = Vec::new();
            for b in &blocks {
                let o = b.forward_full(&mut tape, x, batch, cfg.softmax1);
                x = o.y;
                bouts.push(o);
            }
            let lo = final_ln.forward(&mut tape, x);
            let po2 = proj.forward(&mut tape, lo.y);
            let loss = tape.cross_entropy(po2.y, &targets);
            tape.backward(loss);

            let flat = pack(&tok, &pos, &blocks, &final_ln, &proj);
            assert_eq!(flat.len(), cfg.len(), "{what}: Config's layout must match pack");
            let dev = DeviceParams::upload(&flat);
            let mut dt = DeviceTape::new(&dev);
            let (douts, dlogits, dloss) = model_forward(&mut dt, &cfg, &ids, &targets, batch);
            for (i, (d, c)) in douts.iter().zip(&bouts).enumerate() {
                let e = rel_err(&read(dt.value(*d)), &tape.value(c.y).data);
                assert!(e < 1e-4, "{what}: block {i} output off by {e}");
            }
            let e = rel_err(&read(dt.value(dlogits)), &tape.value(po2.y).data);
            assert!(e < 1e-4, "{what}: logits off by {e}");
            let (gl, cl) = (read(dt.value(dloss))[0], tape.value(loss).data[0]);
            assert!((gl - cl).abs() <= 1e-4 * cl.abs(), "{what}: loss {gl} vs {cl}");

            // ReLU ties: the GPU's FFN hidden (post-ReLU) against the CPU's
            // pre-activation, block by block
            let relus = (0..dt.nodes.len()).filter(|&n| matches!(dt.nodes[n].op, Op::Linear { relu: true, .. }));
            let mut ties = 0;
            for (n, o) in relus.zip(&bouts) {
                let (h, z) = (read(&dt.nodes[n].value), &tape.value(o.ffn1_out.y).data);
                for (hv, zv) in h.iter().zip(z) {
                    if (*hv > 0.0) != (*zv > 0.0) {
                        assert!(zv.abs() < 1e-5, "{what}: ReLUs disagree on pre-activation {zv}, not a rounding tie");
                        ties += 1;
                    }
                }
            }
            if ties > 0 {
                eprintln!("{what}: {ties} ReLU tie(s); next seed");
                return false;
            }

            dt.backward(dloss);
            let g = dev.read(&dev.grads);
            let mut want: Vec<(String, Var)> = vec![("token table".into(), to.table), ("position table".into(), po.table)];
            for (i, o) in bouts.iter().enumerate() {
                for (name, v) in [("ln1 gamma", o.ln1_out.gamma), ("ln1 beta", o.ln1_out.beta), ("qkv w", o.qkv_out.w), ("qkv b", o.qkv_out.b), ("out_proj w", o.out_proj_out.w), ("out_proj b", o.out_proj_out.b), ("ln2 gamma", o.ln2_out.gamma), ("ln2 beta", o.ln2_out.beta), ("ffn1 w", o.ffn1_out.w), ("ffn1 b", o.ffn1_out.b), ("ffn2 w", o.ffn2_out.w), ("ffn2 b", o.ffn2_out.b)] {
                    want.push((format!("block {i} {name}"), v));
                }
            }
            want.extend([("final ln gamma".into(), lo.gamma), ("final ln beta".into(), lo.beta), ("proj w".into(), po2.w), ("proj b".into(), po2.b)]);
            let mut off = 0;
            let mut worst = (0.0f32, String::new());
            for (name, v) in &want {
                let w = &tape.grad(*v).unwrap().data;
                let e = rel_err(&g[off..off + w.len()], w);
                if e > worst.0 {
                    worst = (e, name.clone());
                }
                assert!(e < 1e-4, "{what}: {name} gradient off by {e}");
                off += w.len();
            }
            assert_eq!(off, g.len(), "{what}: gradient layout");
            eprintln!("{what}: {} device ops; worst gradient {:.1e} ({})", dt.len(), worst.0, worst.1);
        }
        true
    }

    /// Needs the discrete GPU. Milestone 5 at unit scale: several SGD
    /// steps on both tapes from the same init and batches (small ragged
    /// model); after every step the loss and every parameter must still
    /// agree. Catches anything that carries over between steps (gradient
    /// zeroing, in-place updates, buffer reuse), which the one-step test
    /// can't. Small on purpose: at the real size (512 rows, 4 blocks) one
    /// of the ~524k FFN pre-activations lands within float rounding of 0
    /// about once a step, so the two tapes' ReLUs disagree on it (step 1
    /// at seed 13: +7.0e-7 on the CPU, <= 0 on the GPU). That unit's whole
    /// gradient flips, the parameters part by ~1e-4, and from then on
    /// dozens of units flip per step. Neither tape is wrong; at that size
    /// the comparison is the training outcome (gpu_train_check).
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn device_training_tracks_cpu_over_steps() {
        use crate::optim::Sgd;
        let cfg = Config { vocab: 21, d: 16, heads: 4, d_ff: 24, t: 6, n_blocks: 2, softmax1: false };
        let (batch, lr) = (3, 0.3);
        let mut rng = Rng::new(13);
        let mut tok = Embedding::new(&mut rng, cfg.vocab, cfg.d);
        let mut pos = Embedding::new(&mut rng, cfg.t, cfg.d);
        let mut blocks: Vec<_> = (0..cfg.n_blocks).map(|_| TransformerBlock::new(&mut rng, cfg.d, cfg.heads, cfg.d_ff)).collect();
        let mut final_ln = LayerNorm::new(cfg.d);
        let mut proj = Linear::new(&mut rng, cfg.d, cfg.vocab);
        let dev = DeviceParams::upload(&pack(&tok, &pos, &blocks, &final_ln, &proj));
        let opt = Sgd { lr };
        let rows = batch * cfg.t;
        for step in 0..6 {
            let ids: Vec<usize> = (0..rows).map(|_| (rng.next_gaussian().abs() * 1e4) as usize % cfg.vocab).collect();
            let targets: Vec<usize> = (0..rows).map(|_| (rng.next_gaussian().abs() * 1e4) as usize % cfg.vocab).collect();
            dev.zero_grads();
            let mut dt = DeviceTape::new(&dev);
            let (_, _, dloss) = model_forward(&mut dt, &cfg, &ids, &targets, batch);
            dt.backward(dloss);
            dev.sgd(lr);
            let gl = read(dt.value(dloss))[0];

            let mut tape = Tape::new();
            let positions: Vec<usize> = (0..batch).flat_map(|_| 0..cfg.t).collect();
            let (to, po) = (tok.forward(&mut tape, &ids), pos.forward(&mut tape, &positions));
            let mut x = tape.add(to.y, po.y);
            let mut bouts = Vec::new();
            for b in &blocks {
                let o = b.forward_full(&mut tape, x, batch, false);
                x = o.y;
                bouts.push(o);
            }
            let lo = final_ln.forward(&mut tape, x);
            let po2 = proj.forward(&mut tape, lo.y);
            let loss = tape.cross_entropy(po2.y, &targets);
            tape.backward(loss);
            tok.apply_grad(&tape, &to, &opt);
            pos.apply_grad(&tape, &po, &opt);
            for (b, o) in blocks.iter_mut().zip(&bouts) {
                b.apply_grad(&tape, o, &opt);
            }
            final_ln.apply_grad(&tape, &lo, &opt);
            proj.apply_grad(&tape, &po2, &opt);
            let cl = tape.value(loss).data[0];

            let e = rel_err(&dev.read(&dev.params), &pack(&tok, &pos, &blocks, &final_ln, &proj));
            eprintln!("step {step}: loss cpu {cl:.6} gpu {gl:.6}; parameters off by {e:.1e}");
            assert!((gl - cl).abs() <= 1e-4 * cl.abs(), "step {step}: loss {gl} vs {cl}");
            assert!(e < 1e-4, "step {step}: parameters off by {e}");
        }
    }
}
