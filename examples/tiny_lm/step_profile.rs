// Where does a training_recipe_check.rs step spend its time? Splits one
// step into forward / backward / apply_grad, and reports how much of it is
// NdArray::matmul (tensor::MATMUL_NANOS). Decides whether multithreading
// matmul (Amdahl: only the matmul share speeds up) or a GPU port is worth it.
//   step_profile [batch=8] [steps=20]
// Same model and init as training_recipe_check.rs (plain softmax).
//
// Result 2026-09-24 (CPU shared with a 17-core job, so absolute times are
// inflated; the shares are the point): 888 tape nodes forward.
//   batch 8: fwd 313 ms + bwd 444 ms + update 0.3 ms = 758 ms, matmul 52%
//   batch 1: fwd  38 ms + bwd  56 ms + update 0.3 ms =  94 ms, matmul 53%
// Amdahl: even infinitely fast matmul only halves a step; 8-way matmul
// threading alone gives ~1.8x. The other ~48% (elementwise, softmax,
// layernorm, transpose, broadcast, allocation) is ~365 ms at batch 8 -
// slow for a few million elements, so it's its own target.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};
use scratchtape::tensor::MATMUL_NANOS;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{ForwardOut, apply_grad, encode_bytes, sample_window};

const D_MODEL: usize = 128;
const N_HEADS: usize = 8;
const D_FF: usize = 256;
const SEQ_LEN: usize = 64;
const N_BLOCKS: usize = 4;
const VOCAB: usize = 256;

struct Model {
    token_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    final_ln: LayerNorm,
    output_proj: Linear,
}

fn forward(tape: &mut Tape, m: &Model, input_ids: &[usize], batch: usize) -> (Var, ForwardOut) {
    let positions: Vec<usize> = (0..batch).flat_map(|_| 0..SEQ_LEN).collect();
    let tok_out = m.token_emb.forward(tape, input_ids);
    let pos_out = m.pos_emb.forward(tape, &positions);
    let mut x = tape.add(tok_out.y, pos_out.y);
    let mut block_outs = Vec::with_capacity(m.blocks.len());
    for block in &m.blocks {
        let out = block.forward_full(tape, x, batch, false);
        x = out.y;
        block_outs.push(out);
    }
    let ln_out = m.final_ln.forward(tape, x);
    let proj_out = m.output_proj.forward(tape, ln_out.y);
    (proj_out.y, ForwardOut { tok_out, pos_out, block_outs, ln_out, proj_out })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let batch: usize = args.get(1).map(|a| a.parse().unwrap()).unwrap_or(8);
    let steps: usize = args.get(2).map(|a| a.parse().unwrap()).unwrap_or(20);
    let full = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let train = &full[..(full.len() as f32 * 0.9) as usize];
    let mut rng = Rng::new(1);
    let mut m = Model {
        token_emb: Embedding::new(&mut rng, VOCAB, D_MODEL),
        pos_emb: Embedding::new(&mut rng, SEQ_LEN, D_MODEL),
        blocks: (0..N_BLOCKS).map(|_| TransformerBlock::new(&mut rng, D_MODEL, N_HEADS, D_FF)).collect(),
        final_ln: LayerNorm::new(D_MODEL),
        output_proj: Linear::new(&mut rng, D_MODEL, VOCAB),
    };
    let opt = Sgd { lr: 0.3 };
    let (mut t_fwd, mut t_bwd, mut t_upd, mut mm_fwd, mut mm_bwd, mut nodes) = (0.0, 0.0, 0.0, 0u64, 0u64, 0);
    for _ in 0..steps {
        let mut input = Vec::with_capacity(batch * SEQ_LEN);
        let mut target = Vec::with_capacity(batch * SEQ_LEN);
        for _ in 0..batch {
            let (i, t) = sample_window(&mut rng, train, SEQ_LEN);
            input.extend(i);
            target.extend(t);
        }
        let mm0 = MATMUL_NANOS.load(Ordering::Relaxed);
        let t0 = Instant::now();
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &m, &input, batch);
        let loss = tape.cross_entropy(logits, &target);
        let t1 = Instant::now();
        let mm1 = MATMUL_NANOS.load(Ordering::Relaxed);
        tape.backward(loss);
        let t2 = Instant::now();
        let mm2 = MATMUL_NANOS.load(Ordering::Relaxed);
        apply_grad(&tape, &out, &mut m.token_emb, &mut m.pos_emb, &mut m.blocks, &mut m.final_ln, &mut m.output_proj, &opt);
        let t3 = Instant::now();
        t_fwd += (t1 - t0).as_secs_f64();
        t_bwd += (t2 - t1).as_secs_f64();
        t_upd += (t3 - t2).as_secs_f64();
        mm_fwd += mm1 - mm0;
        mm_bwd += mm2 - mm1;
        nodes = tape.len();
    }
    let n = steps as f64;
    let (f, b, u) = (t_fwd / n * 1e3, t_bwd / n * 1e3, t_upd / n * 1e3);
    let (mf, mb) = (mm_fwd as f64 / n / 1e6, mm_bwd as f64 / n / 1e6);
    println!("batch {batch}, {steps} steps, {nodes} tape nodes; ms per step:");
    println!("  forward   {f:8.1}  (matmul {mf:7.1}, {:.0}%)", 100.0 * mf / f);
    println!("  backward  {b:8.1}  (matmul {mb:7.1}, {:.0}%)", 100.0 * mb / b);
    println!("  update    {u:8.1}");
    println!("  total     {:8.1}  (matmul {:.0}%)", f + b + u, 100.0 * (mf + mb) / (f + b + u));
}
