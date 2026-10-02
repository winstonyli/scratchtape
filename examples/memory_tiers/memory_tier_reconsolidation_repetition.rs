// Sharpens memory_tier_reconsolidation.rs's early/late phase-2 split
// into what "reconsolidation windows narrow with repetition" actually
// claims: a real dose-response curve against how many times each
// SPECIFIC replayed window has itself been retrieved, not just how far
// into phase 2 the step happens to fall. The early/late version already
// answered the qualitative question (final_ln/output_proj show
// retrieval-specific decay, isolated from ordinary convergence by
// memory_tier_reconsolidation_control.rs) - this checks the shape: does
// the decay track repetition count specifically (consistent with real
// reconsolidation, where each retrieval progressively re-stabilizes a
// memory), or was the early/late split just a coarser view of
// something that tracks elapsed phase-2 time instead?
//
// Same 2-phase setup as memory_tier_reconsolidation.rs (train on corpus
// A, then corpus B with corpus-A replay at p=0.15), but each replay
// event is now keyed by which of the 8 buffer windows was drawn and how
// many times THAT window has been replayed so far (1st, 2nd, 3rd, ...),
// pooled across all 8 windows into one curve per repetition count (capped
// at 20, everything beyond folded into one overflow bucket - the buffer
// gets replayed ~75 times on average over a phase, so resolution stays
// good through the interesting early repetitions without a very long
// thin tail). The phase-2 step index at which each replay happens is
// tracked alongside (as a pseudo-segment through the same Bucket type,
// not a new type) so a rep-count effect can be told apart from a
// step-index effect by eye, even though this file doesn't re-run the
// no-interference control again - that control's finding (the growth
// trend in token_emb/pos_emb/block0-3 is generic, only final_ln/
// output_proj's decay is retrieval-specific) is taken as already
// established and only final_ln/output_proj/block0/token_emb are
// printed here, not the full 8-segment breakdown.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::collections::HashMap;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{ForwardOut, apply_grad, curate, encode_bytes, forward, sample_window};

/// Same six-segment breakdown as memory_tier_reconsolidation.rs -
/// copied for the same reason that file gave for not sharing Fisher's
/// accumulate shape: a third sibling with its own call shape (bucketed
/// by repetition count, not phase time), not a library-worthy
/// abstraction.
fn segment_sq_grads(tape: &Tape, out: &ForwardOut, n_blocks: usize) -> Vec<(String, f32)> {
    let sq = |g: Option<&scratchtape::tensor::NdArray>| g.map(|g| g.data.iter().map(|x| x * x).sum::<f32>()).unwrap_or(0.0);
    let mut result = vec![
        ("token_emb".to_string(), sq(tape.grad(out.tok_out.table))),
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
    let rep_cap = 20;
    let segments = ["token_emb", "block0", "block1", "block2", "block3", "final_ln", "output_proj"];

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> = (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    let start_time = Instant::now();

    println!("phase 1: training {steps_per_phase} steps on corpus A (fables), no replay");
    for _ in 0..steps_per_phase {
        let (input, target) = sample_window(&mut rng, &corpus_a, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
    println!("  phase 1 done ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

    let mut snapshot_rng = Rng::new(99);
    let replay_buffer = curate(Vec::new(), "A(fables)", &corpus_a, seq_len, budget, &mut snapshot_rng);
    let mut rep_counts = vec![0usize; replay_buffer.len()];

    println!("\nphase 2: training {steps_per_phase} steps on corpus B (holmes), replaying corpus A at p={replay_prob}, tracking per-window repetition count");
    let mut by_rep: HashMap<usize, Bucket> = HashMap::new();
    let mut total_replays = 0usize;
    for step in 0..steps_per_phase {
        let is_replay = !replay_buffer.is_empty() && rng.next_f32() < replay_prob;
        if !is_replay {
            let (input, target) = sample_window(&mut rng, &corpus_b, seq_len);
            let mut tape = Tape::with_capacity(2000);
            let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
            let loss = tape.cross_entropy(logits, &target);
            tape.backward(loss);
            apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
            continue;
        }
        let idx = (rng.next_f32() * replay_buffer.len() as f32) as usize;
        rep_counts[idx] += 1;
        let rep = rep_counts[idx].min(rep_cap + 1); // rep_cap+1 = overflow bucket
        total_replays += 1;
        let (_, input, target) = &replay_buffer[idx];
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, input);
        let loss = tape.cross_entropy(logits, target);
        tape.backward(loss);
        let mut grads = segment_sq_grads(&tape, &out, n_blocks);
        grads.push(("phase2_step".to_string(), step as f32));
        by_rep.entry(rep).or_default().add(&grads);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
    println!("  phase 2 done ({:.1}s elapsed) - {total_replays} total replays across {} buffer windows", start_time.elapsed().as_secs_f32(), replay_buffer.len());

    println!("\nmean squared gradient by repetition count of the specific replayed window (rep, n, avg phase2 step, then per-segment):");
    println!("  rep | n | avg_step | {}", segments.join(" | "));
    let mut reps: Vec<usize> = by_rep.keys().copied().collect();
    reps.sort_unstable();
    for rep in reps {
        let bucket = &by_rep[&rep];
        let n = bucket.n("token_emb");
        let avg_step = bucket.avg("phase2_step");
        let rep_label = if rep > rep_cap { format!("{}+", rep_cap + 1) } else { rep.to_string() };
        let cells: Vec<String> = segments.iter().map(|s| format!("{:.2e}", bucket.avg(s))).collect();
        println!("  {rep_label} | {n} | {avg_step:.0} | {}", cells.join(" | "));
    }

    println!("\ntotal training time: {:.1}s", start_time.elapsed().as_secs_f32());
}
