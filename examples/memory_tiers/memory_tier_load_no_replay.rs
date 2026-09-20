// No-replay control for memory_tier_load.rs, existing for one reason:
// memory_tier_diff.rs needs a same-lineage "what if we hadn't replayed"
// checkpoint to compare per-layer drift against. Identical to
// memory_tier_load.rs except replay_prob is 0 and the output path differs
// - duplicated rather than parameterized, same reasoning as every other
// duplicated file in this project (catastrophic_forgetting.rs's own
// train() already showed the alternative - an Option<replay> parameter -
// works fine within one process; across two genuinely separate binaries,
// a full duplicate keeps each program's behavior readable standalone).
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

const CORPUS_B: &str = "Twinkle, twinkle, little star,\n\
How I wonder what you are!\n\
Up above the world so high,\n\
Like a diamond in the sky.";

const CHECKPOINT_PATH: &str = "memory_tier_checkpoint.txt";
const AFTER_NOREPLAY_CHECKPOINT_PATH: &str = "memory_tier_checkpoint_after_noreplay.txt";

const D_MODEL: usize = 32;
const N_HEADS: usize = 4;
const D_FF: usize = 64;
const SEQ_LEN: usize = 16;
const N_BLOCKS: usize = 2;
const VOCAB_SIZE: usize = 256;

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

    let loss_a_on_load = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, SEQ_LEN);
    println!("loss on corpus A immediately after loading (no retraining yet): {loss_a_on_load:.6}");

    let opt = Sgd { lr: 0.3 };
    let steps = 2000;
    let mut rng = Rng::new(2);
    println!("\nphase 2: training on corpus B, NO replay ({steps} steps)");
    for step in 0..steps {
        let (input, target) = sample_window(&mut rng, &corpus_b, SEQ_LEN);
        let mut tape = Tape::new();
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
        if step % 500 == 0 {
            println!("  step {step:>4}: loss = {:.4}", tape.value(loss).data[0]);
        }
    }

    let loss_a_after_b = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, SEQ_LEN);
    let loss_b_after_b = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_b, SEQ_LEN);
    println!("\nloss on A after cross-process no-replay phase 2: {loss_a_after_b:.4}");
    println!("loss on B after cross-process no-replay phase 2: {loss_b_after_b:.4}");
    println!("forgetting on A (cross-process, no replay): {:+.4}", loss_a_after_b - loss_a_on_load);

    let mut flat_after = token_emb.to_flat();
    flat_after.extend(pos_emb.to_flat());
    for block in &blocks {
        flat_after.extend(block.to_flat());
    }
    flat_after.extend(final_ln.to_flat());
    flat_after.extend(output_proj.to_flat());
    let text: String = flat_after.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" ");
    fs::write(AFTER_NOREPLAY_CHECKPOINT_PATH, text).expect("failed to write post-phase-2 checkpoint");
    println!("saved post-phase-2 (no replay) weights to {AFTER_NOREPLAY_CHECKPOINT_PATH}");
}
