use scratchtape::nn::{Embedding, LayerNorm, Linear, TransformerBlock};
use std::fs;

#[path = "../common/mod.rs"]
mod common;
use common::{encode_bytes, eval_loss};

const CORPUS_A: &str = "Shall I compare thee to a summer's day?\n\
Thou art more lovely and more temperate.\n\
Rough winds do shake the darling buds of May,\n\
And summer's lease hath all too short a date.";

const CHECKPOINT_PATH: &str = "checkpoint.txt";

// Must match checkpoint_save.rs exactly - this is a genuinely separate
// process, with no memory of anything checkpoint_save.rs computed. All it
// has is this file and these architecture constants.
const D_MODEL: usize = 32;
const N_HEADS: usize = 4;
const D_FF: usize = 64;
const SEQ_LEN: usize = 16;
const N_BLOCKS: usize = 2;
const VOCAB_SIZE: usize = 256;

// Fields unused here - checkpoint_load only evaluates (no training/
// backward), so it never needs these Vars for apply_grad the way
// checkpoint_save.rs does with the otherwise-identical struct.
#[allow(dead_code)]
fn main() {
    let corpus_a = encode_bytes(CORPUS_A);

    let text = fs::read_to_string(CHECKPOINT_PATH).unwrap_or_else(|_| panic!("couldn't read {CHECKPOINT_PATH} - run checkpoint_save first"));
    let flat: Vec<f32> = text.split_whitespace().map(|s| s.parse().expect("bad float in checkpoint")).collect();
    println!("read {} floats from {CHECKPOINT_PATH}", flat.len());

    // Reconstructs in the identical order checkpoint_save.rs wrote them -
    // purely mechanical, offset threaded through every from_flat call.
    let mut offset = 0usize;
    let token_emb = Embedding::from_flat(&flat, &mut offset, VOCAB_SIZE, D_MODEL);
    let pos_emb = Embedding::from_flat(&flat, &mut offset, SEQ_LEN, D_MODEL);
    let blocks: Vec<TransformerBlock> = (0..N_BLOCKS).map(|_| TransformerBlock::from_flat(&flat, &mut offset, D_MODEL, N_HEADS, D_FF)).collect();
    let final_ln = LayerNorm::from_flat(&flat, &mut offset, D_MODEL);
    let output_proj = Linear::from_flat(&flat, &mut offset, D_MODEL, VOCAB_SIZE);
    assert_eq!(offset, flat.len(), "checkpoint had leftover/missing floats - architecture mismatch");

    let loss_after_load = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, SEQ_LEN);
    println!("loss on corpus A after loading (no retraining): {loss_after_load:.6}");
    println!("\ncompare this exactly against \"loss on corpus A before saving\" printed by checkpoint_save - a genuine cross-process round-trip, not just in-memory continuation.");
}
