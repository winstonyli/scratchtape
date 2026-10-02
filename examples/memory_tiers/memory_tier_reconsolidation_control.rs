// Follow-up to memory_tier_reconsolidation.rs's explicit open question:
// that file found A-replay's gradient shrinking sharply from early to
// late phase 2 in final_ln/output_proj (0.39x, 0.43x) while B-fresh
// stayed flat or grew - the one signature matching real
// reconsolidation's "retrieval windows narrow with repetition" - but
// couldn't rule out the mundane alternative: B-training had already
// pulled the shared weights away from A by retrieval time, so some of
// that shrinkage could just be ordinary convergence as the corrective
// gradient needed on A settles down, nothing retrieval-specific about
// it.
//
// This is the no-interference control that isolates it: same phase 1
// (train on corpus A, same seed, same steps - byte-for-byte identical
// starting point), but phase 2 continues training on corpus A ALONE,
// no corpus B, no replay. If A's own gradient decays late-vs-early on
// a similar schedule here, the earlier experiment's decay was just
// continued optimization, not reconsolidation. If it doesn't decay
// here (flat or grows, the way B-fresh did in the interference run),
// that decay really was specific to retrieval-under-interference.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::collections::HashMap;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{ForwardOut, apply_grad, encode_bytes, forward, sample_window};

/// Same six-segment breakdown as memory_tier_reconsolidation.rs's
/// segment_sq_grads - copied rather than shared, same reasoning as
/// that file's own header comment about Fisher's accumulate not
/// fitting this call shape (and now a second sibling file, not a
/// library-worthy abstraction either).
fn segment_sq_grads(tape: &Tape, out: &ForwardOut, n_blocks: usize) -> Vec<(String, f32)> {
    let sq = |g: Option<&scratchtape::tensor::NdArray>| g.map(|g| g.data.iter().map(|x| x * x).sum::<f32>()).unwrap_or(0.0);
    let mut result = vec![
        ("token_emb".to_string(), sq(tape.grad(out.tok_out.table))),
        ("pos_emb".to_string(), sq(tape.grad(out.pos_out.table))),
        ("final_ln".to_string(), sq(tape.grad(out.ln_out.gamma)) + sq(tape.grad(out.ln_out.beta))),
        ("output_proj".to_string(), sq(tape.grad(out.proj_out.w)) + sq(tape.grad(out.proj_out.b))),
    ];
    for i in 0..n_blocks {
        let b = &out.block_outs[i];
        let mut total = sq(tape.grad(b.ln1_out.gamma)) + sq(tape.grad(b.ln1_out.beta));
        total += sq(tape.grad(b.qkv_out.w)) + sq(tape.grad(b.qkv_out.b));
        total += sq(tape.grad(b.out_proj_out.w)) + sq(tape.grad(b.out_proj_out.b));
        total += sq(tape.grad(b.ln2_out.gamma)) + sq(tape.grad(b.ln2_out.beta));
        total += sq(tape.grad(b.ffn1_out.w)) + sq(tape.grad(b.ffn1_out.b));
        total += sq(tape.grad(b.ffn2_out.w)) + sq(tape.grad(b.ffn2_out.b));
        result.push((format!("block{i}"), total));
    }
    result
}

#[derive(Default)]
struct Bucket {
    sums: HashMap<String, f32>,
    counts: HashMap<String, usize>,
}

impl Bucket {
    fn add(&mut self, grads: &[(String, f32)]) {
        for (name, g) in grads {
            *self.sums.entry(name.clone()).or_insert(0.0) += g;
            *self.counts.entry(name.clone()).or_insert(0) += 1;
        }
    }

    fn avg(&self, name: &str) -> f32 {
        let n = *self.counts.get(name).unwrap_or(&0);
        if n == 0 {
            return 0.0;
        }
        self.sums.get(name).unwrap_or(&0.0) / n as f32
    }
}

fn main() {
    let corpus_a = encode_bytes(include_str!("../../data/aesops_fables.txt"));

    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    let steps_per_phase = 4000;
    let segments = ["token_emb", "pos_emb", "block0", "block1", "block2", "block3", "final_ln", "output_proj"];

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> = (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    let start_time = Instant::now();

    println!("phase 1: training {steps_per_phase} steps on corpus A (fables) - same seed/steps as memory_tier_reconsolidation.rs");
    for _ in 0..steps_per_phase {
        let (input, target) = sample_window(&mut rng, &corpus_a, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
    println!("  phase 1 done ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

    println!("\nphase 2 (control): continuing {steps_per_phase} steps on corpus A alone - no corpus B, no replay");
    let mut early = Bucket::default();
    let mut late = Bucket::default();
    let half = steps_per_phase / 2;
    for step in 0..steps_per_phase {
        let (input, target) = sample_window(&mut rng, &corpus_a, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        let grads = segment_sq_grads(&tape, &out, n_blocks);
        if step < half {
            early.add(&grads);
        } else {
            late.add(&grads);
        }
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
    println!("  phase 2 done ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

    println!("\nper-segment mean squared gradient, continued-A-only - early | late | ratio (late/early):");
    for seg in segments {
        let e = early.avg(seg);
        let l = late.avg(seg);
        let ratio = if e > 0.0 { l / e } else { f32::NAN };
        println!("  {seg}: {e:.3e} | {l:.3e} | {ratio:.3}");
    }

    println!("\ntotal training time: {:.1}s", start_time.elapsed().as_secs_f32());
}
