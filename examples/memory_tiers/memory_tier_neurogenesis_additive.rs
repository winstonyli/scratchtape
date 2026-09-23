// Follow-up to memory_tier_neurogenesis.rs's own stated caveat: that
// file reinitialized block0 to test Josselyn & Frankland's neurogenesis-
// forgetting hypothesis, but reinitializing an EXISTING block is a
// harsher "wipe and regrow" than the biological analog - real adult-
// born neurons integrate ALONGSIDE existing, undisturbed synapses, they
// don't erase them. That conflated a real interference effect with "a
// block relearning from scratch is just a worse starting point."
//
// This is the additive version: instead of overwriting block0, phase 2
// APPENDS a fresh 5th block to the end of the stack (block4), leaving
// blocks 0-3 structurally untouched - same weights, same position, same
// role, nothing erased. It's appended at the END rather than the front
// specifically so blocks 0-3's own forward computation is byte-for-byte
// unaffected by the graft: they still see the exact same inputs they
// always did, and produce the exact same outputs, all the way through
// block3. Only what happens strictly downstream of block3 - the new
// block4, then final_ln/output_proj - has to adapt to anything. (A new
// block0 was considered and rejected: prepending would feed every
// existing block an out-of-distribution input from an untrained new
// first block, which is a different and arguably worse disruption than
// the one being tested here, not a cleaner one.)
//
// Both branches leave every parameter trainable during phase 2 - old
// and new alike - matching the biological picture where existing
// synapses keep their own ongoing plasticity rather than being frozen
// once a new neuron arrives. The only structural difference from the
// control is whether block4 exists at all, isolating: does merely
// having new, initially-untrained capacity in the pipeline disrupt
// previously-encoded information, even when nothing is erased and
// everything stays fully trainable?
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
    println!("phase 1: training {steps_per_phase} steps on corpus A (fables), {n_blocks} blocks");
    train(&mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &corpus_a, seq_len, steps_per_phase, &opt);
    let loss_a_after_phase1 = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, seq_len);
    println!("  loss on A right after phase 1: {loss_a_after_phase1:.3} ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

    let snapshot = flatten_all(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj);

    // Control branch: continue training the 4-block snapshot on corpus
    // B, unchanged shape - ordinary continual learning.
    let (mut c_token_emb, mut c_pos_emb, mut c_blocks, mut c_final_ln, mut c_output_proj) =
        reconstruct(&snapshot, vocab_size, d_model, seq_len, n_blocks, n_heads, d_ff);
    println!("\nphase 2 (control): continuing on corpus B, still {n_blocks} blocks");
    let mut phase2_rng = Rng::new(2);
    train(&mut phase2_rng, &mut c_token_emb, &mut c_pos_emb, &mut c_blocks, &mut c_final_ln, &mut c_output_proj, &corpus_b, seq_len, steps_per_phase, &opt);
    let control_loss_a = eval_loss(&c_token_emb, &c_pos_emb, &c_blocks, &c_final_ln, &c_output_proj, &corpus_a, seq_len);
    let control_loss_b = eval_loss(&c_token_emb, &c_pos_emb, &c_blocks, &c_final_ln, &c_output_proj, &corpus_b, seq_len);
    println!("  control: loss on A = {control_loss_a:.3}, loss on B = {control_loss_b:.3} ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

    // Additive branch: identical snapshot, blocks 0-3 untouched, but a
    // fresh block4 is APPENDED (not substituted) before phase 2 - new
    // capacity, nothing erased. Trains on the SAME corpus-B sequence
    // (same rng seed) for the same step count, so the only difference
    // from the control is the presence of this new block.
    let (mut n_token_emb, mut n_pos_emb, mut n_blocks, mut n_final_ln, mut n_output_proj) =
        reconstruct(&snapshot, vocab_size, d_model, seq_len, n_blocks, n_heads, d_ff);
    let mut neurogenesis_rng = Rng::new(777);
    n_blocks.push(TransformerBlock::new(&mut neurogenesis_rng, d_model, n_heads, d_ff));
    println!("\nphase 2 (additive): continuing on corpus B, blocks 0-3 untouched, block4 appended fresh ({} blocks total)", n_blocks.len());
    let mut phase2_rng = Rng::new(2);
    train(&mut phase2_rng, &mut n_token_emb, &mut n_pos_emb, &mut n_blocks, &mut n_final_ln, &mut n_output_proj, &corpus_b, seq_len, steps_per_phase, &opt);
    let additive_loss_a = eval_loss(&n_token_emb, &n_pos_emb, &n_blocks, &n_final_ln, &n_output_proj, &corpus_a, seq_len);
    let additive_loss_b = eval_loss(&n_token_emb, &n_pos_emb, &n_blocks, &n_final_ln, &n_output_proj, &corpus_b, seq_len);
    println!("  additive: loss on A = {additive_loss_a:.3}, loss on B = {additive_loss_b:.3} ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

    println!(
        "\nforgetting of A (loss increase from {loss_a_after_phase1:.3}): control {:+.3}, additive {:+.3}",
        control_loss_a - loss_a_after_phase1,
        additive_loss_a - loss_a_after_phase1
    );
    println!("learning B: control {control_loss_b:.3}, additive {additive_loss_b:.3}");
    println!("\ntotal training time: {:.1}s", start_time.elapsed().as_secs_f32());
}
