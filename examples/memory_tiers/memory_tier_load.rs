// Second half of the memory-tier integration - see memory_tier_save.rs's
// header comment. This process has no memory of phase 1 at all: it
// reconstructs the phase-1-trained model from memory_tier_checkpoint.txt
// and the retained replay snapshot from memory_tier_replay.txt, then runs
// phase 2 (corpus B) WITH replay - proving persistence and replay compose
// across a genuine process boundary, not just within one running program
// the way catastrophic_forgetting.rs demonstrated them separately.
use scratchtape::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng, TransformerBlock, TransformerBlockOut};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};
use std::fs;

fn encode_bytes(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
}

fn sample_window(rng: &mut Rng, corpus: &[usize], seq_len: usize) -> (Vec<usize>, Vec<usize>) {
    let max_start = corpus.len() - seq_len - 1;
    let start = (rng.next_f32() * max_start as f32) as usize;
    (corpus[start..start + seq_len].to_vec(), corpus[start + 1..start + seq_len + 1].to_vec())
}

// Corpus A is present here only for EVALUATING retained knowledge, the
// same convention catastrophic_forgetting.rs already established - the
// restriction to a tiny snapshot applies to what phase 2 TRAINS on, not
// to what an honest measurement is allowed to check against.
const CORPUS_A: &str = "Shall I compare thee to a summer's day?\n\
Thou art more lovely and more temperate.\n\
Rough winds do shake the darling buds of May,\n\
And summer's lease hath all too short a date.";

// Deliberately different register/vocabulary from CORPUS_A, same choice
// and reasoning as catastrophic_forgetting.rs.
const CORPUS_B: &str = "Twinkle, twinkle, little star,\n\
How I wonder what you are!\n\
Up above the world so high,\n\
Like a diamond in the sky.";

const CHECKPOINT_PATH: &str = "memory_tier_checkpoint.txt";
const REPLAY_PATH: &str = "memory_tier_replay.txt";
const AFTER_REPLAY_CHECKPOINT_PATH: &str = "memory_tier_checkpoint_after_replay.txt";

// Must match memory_tier_save.rs exactly - genuinely separate process,
// same reasoning as checkpoint_load.rs.
const D_MODEL: usize = 32;
const N_HEADS: usize = 4;
const D_FF: usize = 64;
const SEQ_LEN: usize = 16;
const N_BLOCKS: usize = 2;
const VOCAB_SIZE: usize = 256;
const REPLAY_WINDOWS: usize = 8;

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

fn apply_grad(
    tape: &Tape,
    out: &ForwardOut,
    token_emb: &mut Embedding,
    pos_emb: &mut Embedding,
    blocks: &mut [TransformerBlock],
    final_ln: &mut LayerNorm,
    output_proj: &mut Linear,
    opt: &Sgd,
) {
    token_emb.apply_grad(tape, &out.tok_out, opt);
    pos_emb.apply_grad(tape, &out.pos_out, opt);
    for (block, block_out) in blocks.iter_mut().zip(out.block_outs.iter()) {
        block.apply_grad(tape, block_out, opt);
    }
    final_ln.apply_grad(tape, &out.ln_out, opt);
    output_proj.apply_grad(tape, &out.proj_out, opt);
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
    let corpus_b = encode_bytes(CORPUS_B);

    let weights_text = fs::read_to_string(CHECKPOINT_PATH)
        .unwrap_or_else(|_| panic!("couldn't read {CHECKPOINT_PATH} - run memory_tier_save first"));
    let flat: Vec<f32> = weights_text.split_whitespace().map(|s| s.parse().expect("bad float in checkpoint")).collect();
    let mut offset = 0usize;
    let mut token_emb = Embedding::from_flat(&flat, &mut offset, VOCAB_SIZE, D_MODEL);
    let mut pos_emb = Embedding::from_flat(&flat, &mut offset, SEQ_LEN, D_MODEL);
    let mut blocks: Vec<TransformerBlock> =
        (0..N_BLOCKS).map(|_| TransformerBlock::from_flat(&flat, &mut offset, D_MODEL, N_HEADS, D_FF)).collect();
    let mut final_ln = LayerNorm::from_flat(&flat, &mut offset, D_MODEL);
    let mut output_proj = Linear::from_flat(&flat, &mut offset, D_MODEL, VOCAB_SIZE);
    assert_eq!(offset, flat.len(), "checkpoint had leftover/missing floats - architecture mismatch");
    println!("loaded {} floats from {CHECKPOINT_PATH}", flat.len());

    let replay_text = fs::read_to_string(REPLAY_PATH)
        .unwrap_or_else(|_| panic!("couldn't read {REPLAY_PATH} - run memory_tier_save first"));
    let replay_flat: Vec<usize> = replay_text.split_whitespace().map(|s| s.parse().expect("bad token id in replay file")).collect();
    assert_eq!(replay_flat.len(), REPLAY_WINDOWS * 2 * SEQ_LEN, "replay file has the wrong number of tokens - architecture mismatch");
    let replay_buffer: Vec<(Vec<usize>, Vec<usize>)> = (0..REPLAY_WINDOWS)
        .map(|i| {
            let base = i * 2 * SEQ_LEN;
            (replay_flat[base..base + SEQ_LEN].to_vec(), replay_flat[base + SEQ_LEN..base + 2 * SEQ_LEN].to_vec())
        })
        .collect();
    println!("loaded {REPLAY_WINDOWS} replay windows from {REPLAY_PATH}");

    // Round-trip check, same spirit as checkpoint_load.rs's existing one -
    // this should match memory_tier_save.rs's printed "loss on corpus A
    // after phase 1" exactly, confirming the weights survived the
    // save/load boundary before any phase-2 training touches them.
    let loss_a_on_load = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, SEQ_LEN);
    println!("loss on corpus A immediately after loading (no retraining yet): {loss_a_on_load:.6}");
    println!("  compare exactly against memory_tier_save's \"loss on corpus A after phase 1\" - proves the weight round-trip.");

    let opt = Sgd { lr: 0.3 };
    let steps = 2000;
    let replay_prob = 0.15;
    let mut rng = Rng::new(2);
    println!("\nphase 2: training on corpus B WITH replay (15% of steps drawn from the loaded snapshot, {steps} steps)");
    for step in 0..steps {
        let (input, target) = if !replay_buffer.is_empty() && rng.next_f32() < replay_prob {
            let idx = (rng.next_f32() * replay_buffer.len() as f32) as usize;
            replay_buffer[idx].clone()
        } else {
            sample_window(&mut rng, &corpus_b, SEQ_LEN)
        };
        let mut tape = Tape::new();
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
        if step % 500 == 0 {
            println!("  step {step:>4}: loss = {:.4}", tape.value(loss).data[0]);
        }
    }

    let loss_a_after_b_replay = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, SEQ_LEN);
    let loss_b_after_b_replay = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_b, SEQ_LEN);
    println!("\nloss on A after cross-process replay-mitigated phase 2: {loss_a_after_b_replay:.4}");
    println!("loss on B after cross-process replay-mitigated phase 2: {loss_b_after_b_replay:.4}");
    println!("forgetting on A (cross-process, with replay): {:+.4}", loss_a_after_b_replay - loss_a_on_load);
    println!(
        "\ncompare against catastrophic_forgetting.rs's in-process numbers: no-replay forgetting +5.43, in-process-replay forgetting +2.73 (both from a starting loss of 0.43)."
    );

    // Saved for memory_tier_diff.rs - a second checkpoint, same format as
    // the phase-1 one, letting a later tool compute per-layer drift
    // between the two without needing to rerun any training.
    let mut flat_after = token_emb.to_flat();
    flat_after.extend(pos_emb.to_flat());
    for block in &blocks {
        flat_after.extend(block.to_flat());
    }
    flat_after.extend(final_ln.to_flat());
    flat_after.extend(output_proj.to_flat());
    let text: String = flat_after.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" ");
    fs::write(AFTER_REPLAY_CHECKPOINT_PATH, text).expect("failed to write post-phase-2 checkpoint");
    println!("saved post-phase-2 (with replay) weights to {AFTER_REPLAY_CHECKPOINT_PATH}");
}
