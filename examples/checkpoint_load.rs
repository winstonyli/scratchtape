use engine::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, TransformerBlock, TransformerBlockOut};
use engine::tape::{Tape, Var};
use std::fs;

fn encode_bytes(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
}

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
struct ForwardOut {
    tok_out: EmbeddingOut,
    pos_out: EmbeddingOut,
    block_outs: Vec<TransformerBlockOut>,
    ln_out: LayerNormOut,
    proj_out: LinearOut,
}

fn forward(
    tape: &mut Tape,
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    input_ids: &[usize],
) -> (Var, ForwardOut) {
    let positions: Vec<usize> = (0..input_ids.len()).collect();
    let tok_out = token_emb.forward(tape, input_ids);
    let pos_out = pos_emb.forward(tape, &positions);
    let mut x = tape.add(tok_out.y, pos_out.y);

    let mut block_outs = Vec::with_capacity(blocks.len());
    for block in blocks {
        let out = block.forward(tape, x);
        x = out.y;
        block_outs.push(out);
    }

    let ln_out = final_ln.forward(tape, x);
    let proj_out = output_proj.forward(tape, ln_out.y);
    let logits = proj_out.y;
    (logits, ForwardOut { tok_out, pos_out, block_outs, ln_out, proj_out })
}

fn eval_loss(
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    corpus: &[usize],
    seq_len: usize,
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    let mut start = 0;
    while start + seq_len + 1 <= corpus.len() {
        let input = &corpus[start..start + seq_len];
        let target = &corpus[start + 1..start + seq_len + 1];
        let mut tape = Tape::new();
        let (logits, _) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, input);
        let loss = tape.cross_entropy(logits, target);
        total += tape.value(loss).data[0];
        count += 1;
        start += seq_len;
    }
    total / count as f32
}

fn main() {
    let corpus_a = encode_bytes(CORPUS_A);

    let text = fs::read_to_string(CHECKPOINT_PATH)
        .unwrap_or_else(|_| panic!("couldn't read {CHECKPOINT_PATH} - run checkpoint_save first"));
    let flat: Vec<f32> = text.split_whitespace().map(|s| s.parse().expect("bad float in checkpoint")).collect();
    println!("read {} floats from {CHECKPOINT_PATH}", flat.len());

    // Reconstructs in the identical order checkpoint_save.rs wrote them -
    // purely mechanical, offset threaded through every from_flat call.
    let mut offset = 0usize;
    let token_emb = Embedding::from_flat(&flat, &mut offset, VOCAB_SIZE, D_MODEL);
    let pos_emb = Embedding::from_flat(&flat, &mut offset, SEQ_LEN, D_MODEL);
    let blocks: Vec<TransformerBlock> =
        (0..N_BLOCKS).map(|_| TransformerBlock::from_flat(&flat, &mut offset, D_MODEL, N_HEADS, D_FF)).collect();
    let final_ln = LayerNorm::from_flat(&flat, &mut offset, D_MODEL);
    let output_proj = Linear::from_flat(&flat, &mut offset, D_MODEL, VOCAB_SIZE);
    assert_eq!(offset, flat.len(), "checkpoint had leftover/missing floats - architecture mismatch");

    let loss_after_load = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, SEQ_LEN);
    println!("loss on corpus A after loading (no retraining): {loss_after_load:.6}");
    println!("\ncompare this exactly against \"loss on corpus A before saving\" printed by checkpoint_save - a genuine cross-process round-trip, not just in-memory continuation.");
}
