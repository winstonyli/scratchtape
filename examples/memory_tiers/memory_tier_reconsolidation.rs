// Tests a specific prediction the "memory tiers as cognition tiers"
// analogy raises but nothing in this project's line of consolidation
// experiments has checked: reconsolidation. In memory neuroscience,
// retrieving a consolidated memory briefly returns it to a labile,
// re-encoding-like state before it re-stabilizes - it isn't a passive
// readout. The `memory_tier_multigen_diverse*` files already replay old
// windows during later-phase training; this asks what a replay STEP
// actually looks like from the gradient's point of view: does training
// on a retrieved corpus-A window produce a gradient closer to "actively
// learning" (labile) or to "already converged, nothing left to update"
// (stable, frozen)? And does that answer change with repetition, the
// way real reconsolidation windows narrow the more a memory is
// retrieved?
//
// Minimal 2-phase design, not the 4-corpus multigen setup - this is one
// targeted question, not another retention/interference sweep:
//   phase 1: train on corpus A only (Aesop's Fables), no replay. The
//     last 500 steps' per-segment squared-gradient average is the
//     "actively learning A, but nearly converged" baseline.
//   phase 2: train on corpus B (Sherlock Holmes) with corpus-A replay
//     at the same replay_prob=0.15 memory_tier_sweep.rs already found
//     sits solidly in the "any replay works" plateau. Every step's
//     per-segment squared gradient is bucketed four ways: A-replay
//     early/late (first half vs second half of phase 2, by step index)
//     and B-fresh early/late, the same split applied to both so a
//     decay specific to retrieval can be told apart from ordinary
//     within-phase convergence (which would shrink B-fresh's gradient
//     too, for a completely mundane reason).
//
// Segment breakdown (token_emb, pos_emb, block0-3, final_ln,
// output_proj) copied from memory_tier_multigen_diverse_consolidate_fisher.rs's
// train_with_fisher - same six-way split, reused because it's the
// established granularity for "which part of the network is this
// sensitive in", not reinvented here. Not reusing that file's `Fisher`
// struct itself: its accumulate/total shape is a single running phase-
// long average, and this question is about a time series across a
// phase (early vs late), a genuinely different shape - so the
// accumulation here is a small local `Bucket`, not a shared type with
// one consumer on each side.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::collections::HashMap;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{ForwardOut, apply_grad, curate, encode_bytes, forward, sample_window};

/// Same six segments memory_tier_multigen_diverse_consolidate_fisher.rs
/// scores - see that file's train_with_fisher for why blocks are one
/// segment each rather than per-sublayer.
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

    fn n(&self, name: &str) -> usize {
        *self.counts.get(name).unwrap_or(&0)
    }
}

fn main() {
    let corpus_a = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let corpus_b = encode_bytes(include_str!("../../data/sherlock_holmes.txt"));

    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    let steps_per_phase = 4000;
    let replay_prob = 0.15;
    let budget = 8;
    let baseline_window = 500;
    let segments = ["token_emb", "pos_emb", "block0", "block1", "block2", "block3", "final_ln", "output_proj"];

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> = (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    let start_time = Instant::now();

    println!("phase 1: training {steps_per_phase} steps on corpus A (fables), no replay");
    let mut baseline = Bucket::default();
    for step in 0..steps_per_phase {
        let (input, target) = sample_window(&mut rng, &corpus_a, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        if step >= steps_per_phase - baseline_window {
            baseline.add(&segment_sq_grads(&tape, &out, n_blocks));
        }
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
    println!("  phase 1 done ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

    let mut snapshot_rng = Rng::new(99);
    let replay_buffer = curate(Vec::new(), "A(fables)", &corpus_a, seq_len, budget, &mut snapshot_rng);

    println!("\nphase 2: training {steps_per_phase} steps on corpus B (holmes), replaying corpus A at p={replay_prob}");
    let mut a_replay_early = Bucket::default();
    let mut a_replay_late = Bucket::default();
    let mut b_fresh_early = Bucket::default();
    let mut b_fresh_late = Bucket::default();
    let half = steps_per_phase / 2;
    for step in 0..steps_per_phase {
        let is_replay = !replay_buffer.is_empty() && rng.next_f32() < replay_prob;
        let (input, target) = if is_replay {
            let idx = (rng.next_f32() * replay_buffer.len() as f32) as usize;
            let (_, input, target) = &replay_buffer[idx];
            (input.clone(), target.clone())
        } else {
            sample_window(&mut rng, &corpus_b, seq_len)
        };
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        let grads = segment_sq_grads(&tape, &out, n_blocks);
        match (is_replay, step < half) {
            (true, true) => a_replay_early.add(&grads),
            (true, false) => a_replay_late.add(&grads),
            (false, true) => b_fresh_early.add(&grads),
            (false, false) => b_fresh_late.add(&grads),
        }
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
    println!(
        "  phase 2 done ({:.1}s elapsed) - {} A-replay steps ({} early, {} late), {} B-fresh steps",
        start_time.elapsed().as_secs_f32(),
        a_replay_early.n(segments[0]) + a_replay_late.n(segments[0]),
        a_replay_early.n(segments[0]),
        a_replay_late.n(segments[0]),
        b_fresh_early.n(segments[0]) + b_fresh_late.n(segments[0]),
    );

    println!("\nper-segment mean squared gradient - phase1-A-baseline (last {baseline_window} steps) | A-replay early | A-replay late | B-fresh early | B-fresh late:");
    for seg in segments {
        println!(
            "  {seg}: {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e}",
            baseline.avg(seg),
            a_replay_early.avg(seg),
            a_replay_late.avg(seg),
            b_fresh_early.avg(seg),
            b_fresh_late.avg(seg),
        );
    }

    println!("\nrelative decay (late/early, <1.0 = shrinking over phase 2):");
    for seg in segments {
        let a_ratio = if a_replay_early.avg(seg) > 0.0 { a_replay_late.avg(seg) / a_replay_early.avg(seg) } else { f32::NAN };
        let b_ratio = if b_fresh_early.avg(seg) > 0.0 { b_fresh_late.avg(seg) / b_fresh_early.avg(seg) } else { f32::NAN };
        println!("  {seg}: A-replay {a_ratio:.3}, B-fresh {b_ratio:.3}");
    }

    println!("\ntotal training time: {:.1}s", start_time.elapsed().as_secs_f32());
}
