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
//
// Launch census (`step_profile 8 1 census`, 2026-09-24): kernel launches one
// step would make on a device-resident tape, under the model in `census`.
//   layout                    unfused   elementwise-fused   ms at ~10 us/launch
//   8 heads as ops (before)   2372      1281                23.7 / 12.8
//   heads batched (estimate)   776       385                 7.8 /  3.9
//   heads batched (actual)     728       361                 7.3 /  3.6
// The estimate ran 1 head of width 128 in place of 8 batched heads. Per-head
// ops dominated before: 173 matmuls, against 33 batched. Batching heads cut
// launches 3x, fusion would cut another 2x. The remaining broadcast reduces
// are bias gradients, which could fold into matmul epilogues. Launch cost
// only: kernel time comes on top.
//
// After batched heads (2026-09-24, idle CPU): 280 tape nodes forward.
//   batch 8: fwd 145 ms + bwd 204 ms = 349 ms, matmul 37%
// training_recipe_check trains ~1.35x faster than per-head (the estimate
// bounded it at <=1.8x).
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use scratchtape::tensor::MATMUL_NANOS;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{Model, apply_grad, encode_bytes, sample_window};

const D_MODEL: usize = 128;
const N_HEADS: usize = 8;
const D_FF: usize = 256;
const SEQ_LEN: usize = 64;
const N_BLOCKS: usize = 4;
const VOCAB: usize = 256;

/// Elementwise ops a GPU backend could fuse into one kernel.
fn is_ew(op: &str) -> bool {
    matches!(op, "Add" | "Sub" | "Mul" | "Relu" | "Scale" | "Exp" | "Div" | "Sqrt" | "Log")
}

/// Kernel launches one step would make on a device-resident tape, unfused
/// and with elementwise chains fused. Model (assumptions, not measured):
/// - forward: one launch per non-leaf node. Fused: an elementwise node is
///   inlined into its consumer when it has exactly one consumer and that
///   consumer is elementwise with the same element count, or is a reduction
///   (Sum, SumLastAxis, MaxLastAxis). Inlined nodes form a group with one
///   launch.
/// - backward: one kernel per gradient a node sends to a parent (Concat:
///   one per part). Plus one reduce kernel when the parent is smaller
///   (broadcast) and one accumulate kernel per extra consumer. Fused: a
///   whole group's backward is one kernel writing every outgoing gradient,
///   accumulating in place (+=), so extra-consumer adds disappear;
///   broadcast reduces stay; matmuls keep 2.
/// - update: one launch per parameter leaf, or one multi-tensor launch.
/// Gradients sent into leaves are counted separately, because constant
/// leaves (one-hot targets, eps) need none on a real device tape.
fn census(tape: &Tape) {
    let ops = tape.ops();
    let mut consumers = vec![0usize; ops.len()];
    for (_, parents, _) in &ops {
        for &p in parents {
            consumers[p] += 1;
        }
    }
    let mut hist: std::collections::BTreeMap<&str, usize> = Default::default();
    for (op, _, _) in &ops {
        *hist.entry(op).or_default() += 1;
    }
    // inlined[i]: node i is fused into its (single) consumer's kernel.
    let mut inlined = vec![false; ops.len()];
    for (op, parents, elems) in &ops {
        let reduction = matches!(*op, "Sum" | "SumLastAxis" | "MaxLastAxis");
        for &p in parents {
            let (pop, _, pelems) = &ops[p];
            if is_ew(pop) && consumers[p] == 1 && (reduction || (is_ew(op) && pelems == elems)) {
                inlined[p] = true;
            }
        }
    }
    let nonleaf = ops.iter().filter(|(op, _, _)| *op != "Leaf").count();
    let fwd_fused = ops.iter().enumerate().filter(|(i, (op, _, _))| *op != "Leaf" && !inlined[*i]).count();
    let (mut bwd, mut bwd_leaf, mut reduce, mut accum, mut bwd_fused) = (0, 0, 0, 0, 0);
    for (i, (op, parents, elems)) in ops.iter().enumerate() {
        if *op == "Leaf" {
            continue;
        }
        let mut group_has_launch = false;
        for &p in parents {
            let into_leaf = ops[p].0 == "Leaf";
            if into_leaf {
                bwd_leaf += 1
            } else {
                bwd += 1
            }
            if ops[p].2 < *elems && is_ew(op) {
                reduce += 1;
            }
            if matches!(*op, "MatMul" | "BatchedMatMul") {
                bwd_fused += 1;
            } else if !inlined[i] {
                group_has_launch = true;
            }
        }
        if group_has_launch {
            bwd_fused += 1;
        }
    }
    for &c in &consumers {
        accum += c.saturating_sub(1);
    }
    let params = ops.iter().enumerate().filter(|(i, (op, _, _))| *op == "Leaf" && consumers[*i] > 0).count();
    println!("census: {} nodes; op counts {:?}", ops.len(), hist);
    println!("  forward  launches: {nonleaf} unfused, {fwd_fused} with elementwise fusion");
    println!(
        "  backward launches: {} grad kernels ({} of them into leaves) + {reduce} broadcast reduces + {accum} accumulates = {} unfused",
        bwd + bwd_leaf,
        bwd_leaf,
        bwd + bwd_leaf + reduce + accum
    );
    println!("                     ~{} fused (+{reduce} reduces)", bwd_fused);
    println!("  update   launches: {params} leaves in use, or 1 multi-tensor");
    let unfused = nonleaf + bwd + bwd_leaf + reduce + accum + params;
    let fused = fwd_fused + bwd_fused + reduce + 1;
    println!("  step total: ~{unfused} unfused, ~{fused} fused; at ~10 us/launch {:.1} vs {:.1} ms", unfused as f64 * 0.01, fused as f64 * 0.01);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let batch: usize = args.get(1).map(|a| a.parse().unwrap()).unwrap_or(8);
    let steps: usize = args.get(2).map(|a| a.parse().unwrap()).unwrap_or(20);
    if args.get(3).map(String::as_str) == Some("census") {
        let full = encode_bytes(include_str!("../../data/aesops_fables.txt"));
        let mut rng = Rng::new(1);
        let m = Model {
            token_emb: Embedding::new(&mut rng, VOCAB, D_MODEL),
            pos_emb: Embedding::new(&mut rng, SEQ_LEN, D_MODEL),
            blocks: (0..N_BLOCKS).map(|_| TransformerBlock::new(&mut rng, D_MODEL, N_HEADS, D_FF)).collect(),
            final_ln: LayerNorm::new(D_MODEL),
            output_proj: Linear::new(&mut rng, D_MODEL, VOCAB),
        };
        let (mut input, mut target) = (vec![], vec![]);
        for _ in 0..batch {
            let (i, t) = sample_window(&mut rng, &full, SEQ_LEN);
            input.extend(i);
            target.extend(t);
        }
        let mut tape = Tape::with_capacity(2000);
        let (logits, _) = m.forward(&mut tape, &input, batch, false);
        tape.cross_entropy(logits, &target);
        census(&tape);
        return;
    }
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
        let (logits, out) = m.forward(&mut tape, &input, batch, false);
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
