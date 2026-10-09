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
//! - Fused models (`DeviceParams::upload_models`): every node holds K
//!   models' rows, model m's the m-th slice. Linear runs as a K-batch
//!   matmul over the models' weights; ops without parameters don't care.
use super::matmul::{Epilogue, MatRef, matmul};
use super::rows::{LnOut, layer_norm, layer_norm_backward, softmax, softmax_backward};
use super::tokens::{CeOut, cross_entropy, cross_entropy_backward, embed, embed_backward, upload_ids};
use super::{DeviceParams, EW_DIM, buf, client, cubes, k_fill};
use cubecl::prelude::*;
use cubecl::server::Handle;

/// LayerNorm's eps, as `nn::LayerNorm::new`.
const LN_EPS: f32 = 1e-5;

/// A node on the device tape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DVar(usize);

#[derive(Clone)]
enum Op {
    Embed {
        ids: Handle,
        t: usize,
        vocab: usize,
        tok_off: usize,
        pos_off: usize,
    },
    LayerNorm {
        x: usize,
        fwd: LnOut,
        off: usize,
    },
    /// y = act(x @ W + b) (+ residual). W [inp, out] at w_off, b right after.
    Linear {
        x: usize,
        inp: usize,
        w_off: usize,
        relu: bool,
        residual: Option<usize>,
    },
    /// c[z] = a[z] @ b[z] (b stored transposed if trans_b), z < batch,
    /// each operand and c laid out in its node as its `Layout` says.
    BatchedMatmul {
        a: usize,
        b: usize,
        lay: [Layout; 3],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_b: bool,
    },
    Softmax {
        x: usize,
        scale: f32,
    },
    Dropout {
        x: usize,
        rate: f32,
        seed: u32,
    },
    CrossEntropy {
        logits: usize,
        targets: Handle,
        fwd: CeOut,
        vocab: usize,
    },
}

/// Where a batched matmul's matrices sit in a node's [rows, cols] buffer.
#[derive(Clone, Copy, Debug)]
pub enum Layout {
    /// Stacked: matrix z is rows z·r .. z·r + r.
    Stacked,
    /// Attention heads in place (`MatRef::heads`): matrix z = b·heads + h
    /// is columns col0 + h·width .. + width of rows b·t .. b·t + t.
    Heads { col0: usize, heads: usize, width: usize },
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
    /// Dropout rate and this step's seed; None: `dropout` passes through.
    dropout: Option<(f32, u32)>,
    /// Mutual distillation's weight between fused models (`with_distill`).
    distill: f32,
}

impl<'p> DeviceTape<'p> {
    pub fn new(params: &'p DeviceParams) -> Self {
        Self { params, nodes: Vec::new(), dropout: None, distill: 0.0 }
    }

    /// A tape whose `dropout` ops zero each element with probability
    /// `rate` (and scale the rest by 1 / (1 - rate)). `seed` should change
    /// every step; the masks are a hash of it, the op and the element.
    pub fn with_dropout(params: &'p DeviceParams, rate: f32, seed: u32) -> Self {
        assert!((0.0..1.0).contains(&rate));
        Self { params, nodes: Vec::new(), dropout: Some((rate, seed)), distill: 0.0 }
    }

    /// Fused models learn from each other too: each one's cross_entropy
    /// gradient gains `alpha` · KL(peers' mean ‖ its own) per row (deep
    /// mutual learning; `tokens::cross_entropy_backward`). The loss values
    /// stay plain CE.
    pub fn with_distill(self, alpha: f32) -> Self {
        Self { distill: alpha, ..self }
    }

    pub fn value(&self, v: DVar) -> &Handle {
        &self.nodes[v.0].value
    }

    /// A cross_entropy node's per-row losses ([rows], every model's).
    pub fn row_losses(&self, loss: DVar) -> &Handle {
        match &self.nodes[loss.0].op {
            Op::CrossEntropy { fwd, .. } => &fwd.row_loss,
            _ => panic!("row_losses of a non-cross_entropy node"),
        }
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
        let y = embed(&ids, rows, d, t, &self.params.params, tok_off, pos_off, self.params.models);
        self.push(y, rows, d, Op::Embed { ids, t, vocab, tok_off, pos_off })
    }

    /// LayerNorm with gamma/beta at `off` (`LayerNorm::to_flat` order).
    pub fn layer_norm(&mut self, x: DVar, off: usize) -> DVar {
        let (rows, d) = self.shape(x);
        let fwd = layer_norm(self.value(x), rows, d, &self.params.params, off, LN_EPS, self.params.models);
        let y = fwd.y.clone();
        self.push(y, rows, d, Op::LayerNorm { x: x.0, fwd, off })
    }

    /// act(x @ W + b) (+ residual), W [inp, out] at `w_off` and b after it
    /// (`Linear::to_flat` order). One launch; fused models are its batch.
    pub fn linear(&mut self, x: DVar, w_off: usize, out: usize, relu: bool, residual: Option<DVar>) -> DVar {
        let (rows, inp) = self.shape(x);
        let y = client().empty(rows * out * 4);
        let (p, m) = (&self.params.params, self.params.models);
        let rpm = rows / m.k;
        let epi = Epilogue { bias: Some((p, w_off + inp * out)), bias_stride: m.stride, relu, residual: residual.map(|r| (self.value(r), 0)), ..Default::default() };
        let w = MatRef { off: w_off, stride: m.stride, trans: false, ..MatRef::new(p) };
        matmul(MatRef { stride: rpm * inp, ..MatRef::new(self.value(x)) }, w, MatRef { stride: rpm * out, ..MatRef::new(&y) }, m.k, rpm, inp, out, epi);
        self.push(y, rows, out, Op::Linear { x: x.0, inp, w_off, relu, residual: residual.map(|r| r.0) })
    }

    /// out[z] = a[z] @ b[z] (b[z] used transposed if trans_b) for z in
    /// 0..batch, each matrix found in its node through its `Layout`; out is
    /// a new node, laid out as `out` says (Heads needs col0 0: the node is
    /// exactly the heads' columns). One launch.
    #[allow(clippy::too_many_arguments)]
    pub fn batched_matmul(&mut self, a: DVar, a_lay: Layout, b: DVar, b_lay: Layout, batch: usize, trans_b: bool, out: Layout) -> DVar {
        let ((m, k), (br, bc)) = (self.mat_shape(a.0, a_lay, batch), self.mat_shape(b.0, b_lay, batch));
        let n = if trans_b { br } else { bc };
        assert_eq!(k, if trans_b { bc } else { br }, "inner dimensions");
        let (rows, cols) = match out {
            Layout::Stacked => (batch * m, n),
            Layout::Heads { col0, heads, width } => {
                assert!(col0 == 0 && width == n);
                (batch / heads * m, heads * width)
            }
        };
        let y = client().empty(rows * cols * 4);
        let (av, bv) = (self.value(a), self.value(b));
        let (ac, bcols) = (self.nodes[a.0].cols, self.nodes[b.0].cols);
        matmul(view(av, a_lay, ac, m, k, false), view(bv, b_lay, bcols, br, bc, trans_b), view(&y, out, cols, m, n, false), batch, m, k, n, Epilogue::default());
        self.push(y, rows, cols, Op::BatchedMatmul { a: a.0, b: b.0, lay: [a_lay, b_lay, out], batch, m, k, n, trans_b })
    }

    /// The [r, c] of each of node v's `batch` matrices under `lay`.
    fn mat_shape(&self, v: usize, lay: Layout, batch: usize) -> (usize, usize) {
        let (rows, cols) = (self.nodes[v].rows, self.nodes[v].cols);
        match lay {
            Layout::Stacked => (rows / batch, cols),
            Layout::Heads { heads, width, .. } => (rows / (batch / heads), width),
        }
    }

    /// Causal softmax (or softmax1) of `scale * x` over attention scores
    /// [B·H·T, T].
    pub fn causal_softmax(&mut self, x: DVar, scale: f32, one: bool) -> DVar {
        let (rows, n) = self.shape(x);
        let y = softmax(self.value(x), rows, n, n, scale, true, one);
        self.push(y, rows, n, Op::Softmax { x: x.0, scale })
    }

    /// Mean cross-entropy; the value is one f32 per model.
    pub fn cross_entropy(&mut self, logits: DVar, targets: &[usize]) -> DVar {
        let (rows, vocab) = self.shape(logits);
        let targets = upload_ids(targets);
        let k = self.params.models.k;
        let fwd = cross_entropy(self.value(logits), &targets, rows, vocab, k);
        let loss = fwd.loss.clone();
        self.push(loss, 1, k, Op::CrossEntropy { logits: logits.0, targets, fwd, vocab })
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
            k_fill::launch(client(), cubes(len), CubeDim::new_1d(EW_DIM), buf(&g, len), 0.0f32, len as u32);
        }
        self.nodes[v].grad = Some(g.clone());
        (g, !full_write)
    }

    /// Dropout on x if the tape has a rate (`with_dropout`), else x.
    pub fn dropout(&mut self, x: DVar) -> DVar {
        let Some((rate, seed)) = self.dropout else { return x };
        // A distinct seed per node: the node index, mixed in.
        let seed = seed ^ (self.nodes.len() as u32).wrapping_mul(0x9e37_79b9);
        let (rows, cols) = self.shape(x);
        let y = dropout(self.value(x), rows * cols, rate, seed);
        self.push(y, rows, cols, Op::Dropout { x: x.0, rate, seed })
    }

    /// Adds a finished gradient buffer into node v's: aliases it if v has
    /// none yet (see the module doc), otherwise one add launch.
    fn add_grad(&mut self, v: usize, g: Handle) {
        match &self.nodes[v].grad {
            None => self.nodes[v].grad = Some(g),
            Some(dst) => {
                let len = self.nodes[v].rows * self.nodes[v].cols;
                super::count_launch();
                k_add_into::launch(client(), cubes(len), CubeDim::new_1d(EW_DIM), buf(dst, len), buf(&g, len), len as u32);
            }
        }
    }

    /// Backpropagates from `loss` (a cross_entropy node, seed 1).
    /// Parameter gradients accumulate into the params' gradient buffer.
    pub fn backward(&mut self, loss: DVar) {
        assert!(matches!(self.nodes[loss.0].op, Op::CrossEntropy { .. }), "backward starts at a cross_entropy node");
        let grads = self.params.grads.clone();
        let p = self.params.params.clone();
        let m = self.params.models;
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
                    let dl = cross_entropy_backward(&self.nodes[logits].value, &targets, &fwd, r, vocab, m.k, self.distill);
                    self.add_grad(logits, dl);
                }
                Op::Embed { ids, t, vocab, tok_off, pos_off } => {
                    embed_backward(&dy.unwrap(), &ids, rows, cols, t, vocab, &grads, tok_off, pos_off, m);
                }
                Op::LayerNorm { x, fwd, off } => {
                    let (dx, acc) = self.grad_slot(x, true);
                    layer_norm_backward(&dy.unwrap(), &self.nodes[x].value, &fwd, rows, cols, &p, &grads, off, &dx, acc, m);
                }
                Op::Linear { x, inp, w_off, relu, residual } => {
                    let mut dz = dy.unwrap();
                    if relu {
                        let masked = client().empty(rows * cols * 4);
                        let len = rows * cols;
                        super::count_launch();
                        k_relu_mask::launch(client(), cubes(len), CubeDim::new_1d(EW_DIM), buf(&dz, len), buf(&self.nodes[i].value, len), buf(&masked, len), len as u32);
                        dz = masked;
                    }
                    let out = cols;
                    let xv = self.nodes[x].value.clone();
                    // Per model (the matmuls' batch): dW += xᵀ dz, db +=
                    // colsum(dz) (inside the dW matmul), dx (+)= dz Wᵀ
                    let rpm = rows / m.k;
                    let (xs, dzs) = (MatRef { stride: rpm * inp, ..MatRef::new(&xv) }, MatRef { stride: rpm * out, ..MatRef::new(&dz) });
                    matmul(
                        MatRef { trans: true, ..xs },
                        dzs,
                        MatRef { off: w_off, stride: m.stride, ..MatRef::new(&grads) },
                        m.k,
                        inp,
                        rpm,
                        out,
                        Epilogue { accumulate: true, col_sum: Some((w_off + inp * out, m.stride)), ..Default::default() },
                    );
                    let (dx, acc) = self.grad_slot(x, true);
                    matmul(
                        dzs,
                        MatRef { off: w_off, stride: m.stride, trans: true, ..MatRef::new(&p) },
                        MatRef { stride: rpm * inp, ..MatRef::new(&dx) },
                        m.k,
                        rpm,
                        out,
                        inp,
                        Epilogue { accumulate: acc, ..Default::default() },
                    );
                    if let Some(r) = residual {
                        self.add_grad(r, dz);
                    }
                }
                Op::BatchedMatmul { a, b, lay: [a_lay, b_lay, c_lay], batch, m, k, n, trans_b } => {
                    let dc = dy.unwrap();
                    let (av, bv) = (self.nodes[a].value.clone(), self.nodes[b].value.clone());
                    let (ac, bcols) = (self.nodes[a].cols, self.nodes[b].cols);
                    let dcr = view(&dc, c_lay, cols, m, n, false);
                    // A head view covers only some of its node's columns, so
                    // that gradient starts zeroed and is accumulated into.
                    let partial = |l: Layout| matches!(l, Layout::Heads { .. });
                    let (da, acc_a) = self.grad_slot(a, !partial(a_lay));
                    let (db, acc_b) = self.grad_slot(b, !partial(b_lay));
                    let (br, bc) = if trans_b { (n, k) } else { (k, n) };
                    let db_ref = view(&db, b_lay, bcols, br, bc, false);
                    let (av_ref, bv_ref) = (view(&av, a_lay, ac, m, k, false), view(&bv, b_lay, bcols, br, bc, false));
                    let da_ref = view(&da, a_lay, ac, m, k, false);
                    if trans_b {
                        // c = a bᵀ, b [n, k]: da = dc b, db = dcᵀ a
                        matmul(dcr, bv_ref, da_ref, batch, m, n, k, Epilogue { accumulate: acc_a, ..Default::default() });
                        matmul(MatRef { trans: true, ..dcr }, av_ref, db_ref, batch, n, m, k, Epilogue { accumulate: acc_b, ..Default::default() });
                    } else {
                        // c = a b, b [k, n]: da = dc bᵀ, db = aᵀ dc
                        matmul(dcr, MatRef { trans: true, ..bv_ref }, da_ref, batch, m, n, k, Epilogue { accumulate: acc_a, ..Default::default() });
                        matmul(MatRef { trans: true, ..av_ref }, dcr, db_ref, batch, k, m, n, Epilogue { accumulate: acc_b, ..Default::default() });
                    }
                }
                Op::Softmax { x, scale } => {
                    let dx = softmax_backward(&dy.unwrap(), &self.nodes[i].value, rows, cols, scale);
                    self.add_grad(x, dx);
                }
                Op::Dropout { x, rate, seed } => {
                    // The forward's mask and scale, regenerated from the seed.
                    let dx = dropout(&dy.unwrap(), rows * cols, rate, seed);
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
    /// Where a block's tensors start, given the block's own start `off`
    /// (`TransformerBlock::to_flat` order: ln1, qkv, out_proj, ln2, ffn1,
    /// ffn2).
    fn block_offsets(&self, off: usize) -> BlockOffsets {
        let (d, f) = (self.d, self.d_ff);
        let qkv = off + 2 * d;
        let out = qkv + d * 3 * d + 3 * d;
        let ln2 = out + d * d + d;
        let ffn1 = ln2 + 2 * d;
        let ffn2 = ffn1 + d * f + f;
        BlockOffsets { ln1: off, qkv, out, ln2, ffn1, ffn2, end: ffn2 + f * d + d }
    }
    fn block_len(&self) -> usize {
        self.block_offsets(0).end
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
    /// Weight decay's usual mask, in `pack` order: 1 for the embedding
    /// tables and Linear weights, 0 for LayerNorm gains and shifts and
    /// for biases.
    pub fn decay_mask(&self) -> Vec<f32> {
        let (d, f, v) = (self.d, self.d_ff, self.vocab);
        let mut mask = vec![1.0; (v + self.t) * d];
        // (weights, then parameters exempt), per piece; LayerNorm is (0, 2d).
        let block = [(0, 2 * d), (d * 3 * d, 3 * d), (d * d, d), (0, 2 * d), (d * f, f), (f * d, d)];
        let pieces = (0..self.n_blocks).flat_map(|_| block).chain([(0, 2 * d), (d * v, v)]);
        for (weights, exempt) in pieces {
            mask.extend(std::iter::repeat_n(1.0, weights));
            mask.extend(std::iter::repeat_n(0.0, exempt));
        }
        assert_eq!(mask.len(), self.len());
        mask
    }
}

/// Start of each tensor of one block in the flat buffer (`Config::block_offsets`).
struct BlockOffsets {
    ln1: usize,
    qkv: usize,
    out: usize,
    ln2: usize,
    ffn1: usize,
    ffn2: usize,
    end: usize,
}

/// One transformer block, as `TransformerBlock::forward_full` without the
/// extras (QK-norm, gate, sinks are out of scope for v1). `off` is the
/// block's `to_flat` start. 9 ops: attention reads Q, K and V as head
/// views of the QKV output and writes its heads straight into the merged
/// layout.
pub fn block_forward(tape: &mut DeviceTape, cfg: &Config, x: DVar, off: usize, batch: usize) -> DVar {
    let (d, h, f) = (cfg.d, cfg.heads, cfg.d_ff);
    let dk = d / h;
    let o = cfg.block_offsets(off);
    let ln1 = tape.layer_norm(x, o.ln1);
    let qkv = tape.linear(ln1, o.qkv, 3 * d, false, None);
    let head = |col0| Layout::Heads { col0, heads: h, width: dk };
    let scores = tape.batched_matmul(qkv, head(0), qkv, head(d), batch * h, true, Layout::Stacked);
    let weights = tape.causal_softmax(scores, 1.0 / (dk as f32).sqrt(), cfg.softmax1);
    let merged = tape.batched_matmul(weights, Layout::Stacked, qkv, head(2 * d), batch * h, false, head(0));
    let merged = tape.dropout(merged);
    let x1 = tape.linear(merged, o.out, d, false, Some(x));
    let ln2 = tape.layer_norm(x1, o.ln2);
    let hidden = tape.linear(ln2, o.ffn1, f, true, None);
    let hidden = tape.dropout(hidden);
    tape.linear(hidden, o.ffn2, d, false, Some(x1))
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

/// Node matrix z, under `lay`, of a buffer h shaped like its node (width
/// `cols`, each matrix stored [r, c]) as a matmul operand, used transposed
/// if `trans`.
fn view(h: &Handle, lay: Layout, cols: usize, r: usize, c: usize, trans: bool) -> MatRef<'_> {
    match lay {
        Layout::Stacked => MatRef { stride: r * c, trans, ..MatRef::new(h) },
        Layout::Heads { col0, heads, width } => MatRef { trans, ..MatRef::heads(h, col0, cols, r, heads, width) },
    }
}

#[cube(launch)]
fn k_add_into(dst: &mut [f32], src: &[f32], len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        dst[i] += src[i];
    }
}

/// y = x / (1 - rate) where hash(seed, i) >= rate·2³², else 0; one
/// launch. Applied to dy with the same seed, it is dropout's backward.
pub fn dropout(x: &Handle, len: usize, rate: f32, seed: u32) -> Handle {
    let y = client().empty(len * 4);
    let threshold = (rate as f64 * 4294967296.0) as u32;
    super::count_launch();
    k_dropout::launch(client(), cubes(len), CubeDim::new_1d(EW_DIM), buf(x, len), buf(&y, len), seed, threshold, 1.0 / (1.0 - rate), len as u32);
    y
}

/// `dropout`'s kernel. The mask bits are lowbias32 (Wellons) of
/// i + seed·golden ratio: an independent-looking 32-bit value per (seed, i).
#[cube(launch)]
fn k_dropout(x: &[f32], y: &mut [f32], seed: u32, threshold: u32, scale: f32, len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        let mut h = (i as u32) + seed * 0x9e37_79b9u32;
        h ^= h >> 16;
        h *= 0x7feb_352du32;
        h ^= h >> 15;
        h *= 0x846c_a68bu32;
        h ^= h >> 16;
        let mut v = 0.0f32;
        if h >= threshold {
            v = x[i] * scale;
        }
        y[i] = v;
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

    /// decay_mask against a fresh model's `pack`: there, LayerNorm gains
    /// are 1 and its shifts and all biases 0, while weights and
    /// embeddings are random. So the exempt parameters must be exactly
    /// the ones equal to 0 or 1, with one gain per LayerNorm unit.
    #[test]
    fn decay_mask_exempts_layer_norm_and_biases() {
        let cfg = Config { vocab: 21, d: 16, heads: 4, d_ff: 24, t: 6, n_blocks: 2, softmax1: false };
        let mut rng = Rng::new(5);
        let tok = Embedding::new(&mut rng, cfg.vocab, cfg.d);
        let pos = Embedding::new(&mut rng, cfg.t, cfg.d);
        let blocks: Vec<_> = (0..cfg.n_blocks).map(|_| TransformerBlock::new(&mut rng, cfg.d, cfg.heads, cfg.d_ff)).collect();
        let flat = pack(&tok, &pos, &blocks, &LayerNorm::new(cfg.d), &Linear::new(&mut rng, cfg.d, cfg.vocab));
        let mask = cfg.decay_mask();
        assert_eq!(mask.len(), flat.len());
        for (i, (p, m)) in flat.iter().zip(&mask).enumerate() {
            assert_eq!(*m == 0.0, *p == 0.0 || *p == 1.0, "parameter {i} = {p}, mask {m}");
        }
        let gains = flat.iter().zip(&mask).filter(|(p, m)| **m == 0.0 && **p == 1.0).count();
        assert_eq!(gains, (2 * cfg.n_blocks + 1) * cfg.d);
    }

    /// Dropout's mask: about `rate` of the elements zeroed and the rest
    /// scaled exactly; the same elements for dy (the backward pass) as for
    /// x under one seed, and different ones under another seed.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn dropout_masks_match_between_passes() {
        let (n, rate) = (100_000, 0.1f32);
        let x: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 1e-3).collect();
        let dy: Vec<f32> = (0..n).map(|i| -2.0 - i as f32 * 1e-4).collect();
        let pass = |v: &[f32], seed| read(&dropout(&crate::gpu_step::upload_f32(v), n, rate, seed));
        let (y, dx, other) = (pass(&x, 7), pass(&dy, 7), pass(&x, 8));
        let kept = y.iter().filter(|&&v| v != 0.0).count();
        assert!((kept as f32 / n as f32 - (1.0 - rate)).abs() < 0.005, "kept {kept} of {n}");
        let scale = 1.0 / (1.0 - rate);
        for i in 0..n {
            assert_eq!(y[i] == 0.0, dx[i] == 0.0, "masks differ at {i}");
            assert!(y[i] == 0.0 || (y[i] == x[i] * scale && dx[i] == dy[i] * scale), "scale at {i}");
        }
        let same = (0..n).filter(|&i| (y[i] == 0.0) == (other[i] == 0.0)).count();
        assert!(same < n * 19 / 20, "seed 8's mask matches seed 7's on {same} of {n}");
    }

    /// Backward through dropout, end to end: the loss's finite difference
    /// along the gradient matches the gradient's norm, with the masks held
    /// fixed by the seed (small config, rate 0.3). Dropout must also
    /// change the loss.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn dropout_gradient_matches_finite_difference() {
        let cfg = Config { vocab: 21, d: 16, heads: 4, d_ff: 24, t: 6, n_blocks: 2, softmax1: false };
        let batch = 3;
        let mut rng = Rng::new(9);
        let tok = Embedding::new(&mut rng, cfg.vocab, cfg.d);
        let pos = Embedding::new(&mut rng, cfg.t, cfg.d);
        let blocks: Vec<_> = (0..cfg.n_blocks).map(|_| TransformerBlock::new(&mut rng, cfg.d, cfg.heads, cfg.d_ff)).collect();
        let flat = pack(&tok, &pos, &blocks, &LayerNorm::new(cfg.d), &Linear::new(&mut rng, cfg.d, cfg.vocab));
        let mut ids = || (0..batch * cfg.t).map(|_| (rng.next_f32() * cfg.vocab as f32) as usize % cfg.vocab).collect::<Vec<_>>();
        let (input, target) = (ids(), ids());
        let loss_at = |p: &[f32], rate: Option<f32>, grad: bool| {
            let dev = DeviceParams::upload(p);
            let mut dt = match rate {
                Some(r) => DeviceTape::with_dropout(&dev, r, 11),
                None => DeviceTape::new(&dev),
            };
            let (_, _, l) = model_forward(&mut dt, &cfg, &input, &target, batch);
            if grad {
                dt.backward(l);
            }
            (read(dt.value(l))[0], dev.read(&dev.grads))
        };
        let (loss, g) = loss_at(&flat, Some(0.3), true);
        assert!((loss - loss_at(&flat, None, false).0).abs() > 1e-3, "dropout didn't change the loss");
        let norm = g.iter().map(|v| v * v).sum::<f32>().sqrt();
        // The error shrinks as eps² down to 3e-4 (curvature), then float
        // noise takes over; no dropout behaves the same.
        let eps = 3e-4;
        let moved = |s: f32| flat.iter().zip(&g).map(|(p, gi)| p + s * eps * gi / norm).collect::<Vec<_>>();
        let fd = (loss_at(&moved(1.0), Some(0.3), false).0 - loss_at(&moved(-1.0), Some(0.3), false).0) / (2.0 * eps);
        assert!((fd - norm).abs() < 1e-2 * norm, "finite difference {fd} vs |grad| {norm}");
    }

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
                for (name, v) in [
                    ("ln1 gamma", o.ln1_out.gamma),
                    ("ln1 beta", o.ln1_out.beta),
                    ("qkv w", o.qkv_out.w),
                    ("qkv b", o.qkv_out.b),
                    ("out_proj w", o.out_proj_out.w),
                    ("out_proj b", o.out_proj_out.b),
                    ("ln2 gamma", o.ln2_out.gamma),
                    ("ln2 beta", o.ln2_out.beta),
                    ("ffn1 w", o.ffn1_out.w),
                    ("ffn1 b", o.ffn1_out.b),
                    ("ffn2 w", o.ffn2_out.w),
                    ("ffn2 b", o.ffn2_out.b),
                ] {
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

    /// Needs the discrete GPU. Horizontal fusion: K models, each its own
    /// init and batch, trained in the same launches, against each trained
    /// alone. The forward uses the same tiles per model, so step 0's
    /// losses must match exactly; gradients may differ by the order
    /// split-k sums in (single models only), so parameters get 1e-5. The
    /// small ragged config runs three SGD steps. The real one runs the
    /// training shape, K 4 × batch 32, whose 65536 attention rows fold the
    /// row kernels' grid into y, for one step only: its step 1 hit a ReLU
    /// tie (model 3: 12499 parameters off by up to 1.2e-4, all at or
    /// upstream of block 1's FFN1 unit 130), as device_training_tracks_cpu_over_steps
    /// describes.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn fused_models_match_separate() {
        let small = Config { vocab: 21, d: 16, heads: 4, d_ff: 24, t: 6, n_blocks: 2, softmax1: false };
        let real = Config { vocab: 256, d: 128, heads: 8, d_ff: 256, t: 64, n_blocks: 4, softmax1: false };
        for (cfg, k, batch, steps) in [(small, 3, 3, 3), (real, 4, 32, 1)] {
            let flats: Vec<Vec<f32>> = (0..k as u64)
                .map(|s| {
                    let mut rng = Rng::new(40 + s);
                    let tok = Embedding::new(&mut rng, cfg.vocab, cfg.d);
                    let pos = Embedding::new(&mut rng, cfg.t, cfg.d);
                    let blocks: Vec<_> = (0..cfg.n_blocks).map(|_| TransformerBlock::new(&mut rng, cfg.d, cfg.heads, cfg.d_ff)).collect();
                    pack(&tok, &pos, &blocks, &LayerNorm::new(cfg.d), &Linear::new(&mut rng, cfg.d, cfg.vocab))
                })
                .collect();
            let alone: Vec<DeviceParams> = flats.iter().map(|f| DeviceParams::upload(f)).collect();
            let fused = DeviceParams::upload_models(&flats.concat(), k);
            let mut rng = Rng::new(7);
            let rows = batch * cfg.t;
            for step in 0..steps {
                let mut ids = || (0..k * rows).map(|_| (rng.next_gaussian().abs() * 1e4) as usize % cfg.vocab).collect::<Vec<_>>();
                let (ids, targets) = (ids(), ids());
                fused.zero_grads();
                let mut dt = DeviceTape::new(&fused);
                let (_, _, l) = model_forward(&mut dt, &cfg, &ids, &targets, k * batch);
                dt.backward(l);
                fused.sgd(0.3);
                let fused_loss = read(dt.value(l));
                let fused_params = fused.read(&fused.params);
                for (m, dev) in alone.iter().enumerate() {
                    let slice = |v: &[usize]| v[m * rows..(m + 1) * rows].to_vec();
                    dev.zero_grads();
                    let mut dt = DeviceTape::new(dev);
                    let (_, _, l) = model_forward(&mut dt, &cfg, &slice(&ids), &slice(&targets), batch);
                    dt.backward(l);
                    dev.sgd(0.3);
                    let what = format!("{cfg:?} step {step} model {m}");
                    let loss = read(dt.value(l))[0];
                    // Step 0's forward shares every input bit for bit.
                    if step == 0 {
                        assert_eq!(fused_loss[m], loss, "{what}: loss");
                    }
                    assert!((fused_loss[m] - loss).abs() <= 1e-5 * loss, "{what}: loss {} vs {loss}", fused_loss[m]);
                    let e = rel_err(&fused_params[m * cfg.len()..(m + 1) * cfg.len()], &dev.read(&dev.params));
                    assert!(e < 1e-5, "{what}: parameters off by {e}");
                }
            }
        }
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
