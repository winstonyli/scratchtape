// Another brain analog the memory-tier line raises but hadn't tested:
// Josselyn & Frankland's neurogenesis-forgetting hypothesis (Akers et
// al. 2014) - new hippocampal neurons integrating into an existing
// circuit are proposed to CAUSE forgetting of memories that circuit
// already held, independent of the passage of time or how much else
// gets learned. It's the leading explanation for infantile amnesia
// (neurogenesis rate is highest in infancy) and for why experimentally
// suppressing/boosting adult neurogenesis respectively slows/speeds
// forgetting. Adult neurogenesis specifically happens in the dentate
// gyrus, an early/shallow stage of the hippocampal circuit - which is
// why this reinitializes block0 (the earliest transformer block), not
// an arbitrary one: it's also the segment `memory_tier_multigen_diverse_consolidate_fisher.rs`
// and `memory_tier_reconsolidation.rs` independently found to be the
// most Fisher-sensitive / highest-raw-gradient part of this
// architecture, so it's the segment where "new growth" should matter
// most if the analogy holds at all.
//
// Design: train once on corpus A (phase 1), snapshot the weights, then
// fork into two phase-2 branches that see the IDENTICAL corpus-B
// training sequence (same rng seed) for the same step count - the only
// difference is that one branch gets block0 reinitialized to fresh
// random weights before phase 2 starts, mimicking new neurons born
// into the existing circuit. If reinitializing genuinely causes EXTRA
// forgetting of corpus A beyond what the same additional training
// alone produces, that's the analog holding up; if both branches
// forget A about equally, reinitializing one block didn't matter more
// than continuing to train it would have anyway.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{apply_grad, encode_bytes, eval_loss, flatten_all, forward, reconstruct, sample_window};

fn train(
    rng: &mut Rng,
    token_emb: &mut Embedding,
    pos_emb: &mut Embedding,
    blocks: &mut [TransformerBlock],
    final_ln: &mut LayerNorm,
    output_proj: &mut Linear,
    corpus: &[usize],
    seq_len: usize,
    steps: usize,
    opt: &Sgd,
) {
    for _ in 0..steps {
        let (input, target) = sample_window(rng, corpus, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, token_emb, pos_emb, blocks, final_ln, output_proj, opt);
    }
}

fn main() {
    let corpus_a = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let corpus_b = encode_bytes(include_str!("../../data/sherlock_holmes.txt"));

    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    let steps_per_phase = 4000;
    let opt = Sgd { lr: 0.3 };

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> = (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);

    let start_time = Instant::now();
    println!("phase 1: training {steps_per_phase} steps on corpus A (fables)");
    train(&mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &corpus_a, seq_len, steps_per_phase, &opt);
    let loss_a_after_phase1 = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, seq_len);
    println!("  loss on A right after phase 1: {loss_a_after_phase1:.3} ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

    let snapshot = flatten_all(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj);

    // Control branch: continue training the snapshot on corpus B,
    // untouched - ordinary continual learning, no reinitialization.
    let (mut c_token_emb, mut c_pos_emb, mut c_blocks, mut c_final_ln, mut c_output_proj) = reconstruct(&snapshot, vocab_size, d_model, seq_len, n_blocks, n_heads, d_ff);
    println!("\nphase 2 (control): continuing on corpus B, block0 untouched");
    let mut phase2_rng = Rng::new(2);
    train(&mut phase2_rng, &mut c_token_emb, &mut c_pos_emb, &mut c_blocks, &mut c_final_ln, &mut c_output_proj, &corpus_b, seq_len, steps_per_phase, &opt);
    let control_loss_a = eval_loss(&c_token_emb, &c_pos_emb, &c_blocks, &c_final_ln, &c_output_proj, &corpus_a, seq_len);
    let control_loss_b = eval_loss(&c_token_emb, &c_pos_emb, &c_blocks, &c_final_ln, &c_output_proj, &corpus_b, seq_len);
    println!("  control: loss on A = {control_loss_a:.3}, loss on B = {control_loss_b:.3} ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

    // Neurogenesis branch: identical snapshot, but block0 gets replaced
    // with fresh random weights before phase 2 - new neurons integrating
    // into an existing circuit - then trains on the SAME corpus-B
    // sequence (same rng seed) for the same step count, so the only
    // difference from the control is the reinitialization event itself.
    let (mut n_token_emb, mut n_pos_emb, mut n_blocks, mut n_final_ln, mut n_output_proj) = reconstruct(&snapshot, vocab_size, d_model, seq_len, n_blocks, n_heads, d_ff);
    let mut neurogenesis_rng = Rng::new(777);
    n_blocks[0] = TransformerBlock::new(&mut neurogenesis_rng, d_model, n_heads, d_ff);
    println!("\nphase 2 (neurogenesis): continuing on corpus B, block0 reinitialized to fresh random weights first");
    let mut phase2_rng = Rng::new(2);
    train(&mut phase2_rng, &mut n_token_emb, &mut n_pos_emb, &mut n_blocks, &mut n_final_ln, &mut n_output_proj, &corpus_b, seq_len, steps_per_phase, &opt);
    let neuro_loss_a = eval_loss(&n_token_emb, &n_pos_emb, &n_blocks, &n_final_ln, &n_output_proj, &corpus_a, seq_len);
    let neuro_loss_b = eval_loss(&n_token_emb, &n_pos_emb, &n_blocks, &n_final_ln, &n_output_proj, &corpus_b, seq_len);
    println!("  neurogenesis: loss on A = {neuro_loss_a:.3}, loss on B = {neuro_loss_b:.3} ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

    println!(
        "\nforgetting of A (loss increase from {loss_a_after_phase1:.3}): control +{:.3}, neurogenesis +{:.3}",
        control_loss_a - loss_a_after_phase1,
        neuro_loss_a - loss_a_after_phase1
    );
    println!("learning B: control {control_loss_b:.3}, neurogenesis {neuro_loss_b:.3}");
    println!("\ntotal training time: {:.1}s", start_time.elapsed().as_secs_f32());
}
