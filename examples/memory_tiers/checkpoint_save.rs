use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::fs;

#[path = "../common/mod.rs"]
mod common;
use common::{apply_grad, encode_bytes, eval_loss, forward, sample_window};

const CORPUS_A: &str = "Shall I compare thee to a summer's day?\n\
Thou art more lovely and more temperate.\n\
Rough winds do shake the darling buds of May,\n\
And summer's lease hath all too short a date.";

const CHECKPOINT_PATH: &str = "checkpoint.txt";

// Architecture must match checkpoint_load.rs exactly - shapes aren't
// self-described in the file, so both sides need the identical numbers.
const D_MODEL: usize = 32;
const N_HEADS: usize = 4;
const D_FF: usize = 64;
const SEQ_LEN: usize = 16;
const N_BLOCKS: usize = 2;
const VOCAB_SIZE: usize = 256;

fn main() {
    let corpus_a = encode_bytes(CORPUS_A);
    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, VOCAB_SIZE, D_MODEL);
    let mut pos_emb = Embedding::new(&mut rng, SEQ_LEN, D_MODEL);
    let mut blocks: Vec<TransformerBlock> =
        (0..N_BLOCKS).map(|_| TransformerBlock::new(&mut rng, D_MODEL, N_HEADS, D_FF)).collect();
    let mut final_ln = LayerNorm::new(D_MODEL);
    let mut output_proj = Linear::new(&mut rng, D_MODEL, VOCAB_SIZE);
    let opt = Sgd { lr: 0.3 };

    // Short run (500 steps, not full convergence) - the point here is
    // proving persistence round-trips correctly, not re-demonstrating
    // training convergence already shown in tiny_lm.rs.
    let steps = 500;
    for step in 0..steps {
        let (input, target) = sample_window(&mut rng, &corpus_a, SEQ_LEN);
        let mut tape = Tape::new();
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
        if step % 100 == 0 {
            println!("step {step:>3}: loss = {:.4}", tape.value(loss).data[0]);
        }
    }

    let loss_before_save = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, SEQ_LEN);
    println!("loss on corpus A before saving: {loss_before_save:.6}");

    // Purely mechanical concatenation, fixed order, no headers - the same
    // architecture constants above are what checkpoint_load.rs must also
    // use to reconstruct these shapes correctly.
    let mut flat = token_emb.to_flat();
    flat.extend(pos_emb.to_flat());
    for block in &blocks {
        flat.extend(block.to_flat());
    }
    flat.extend(final_ln.to_flat());
    flat.extend(output_proj.to_flat());

    let text: String = flat.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" ");
    fs::write(CHECKPOINT_PATH, text).expect("failed to write checkpoint");
    println!("saved {} floats to {CHECKPOINT_PATH}", flat.len());
    println!("\nrun `cargo run --release --example checkpoint_load` next - it's a separate process reading this file back.");
}
