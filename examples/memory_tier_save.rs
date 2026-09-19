// First integration of the three memory-tier scaffolding stages
// ([20e5038] forgetting, [c2f0ad2] replay, [d151b29] checkpointing) - each
// was previously its own isolated demo (catastrophic_forgetting.rs never
// touches checkpointing, checkpoint_save.rs/checkpoint_load.rs never touch
// replay). This pair of programs (memory_tier_save.rs / memory_tier_load.rs)
// trains phase 1, persists BOTH the weights and the replay snapshot to
// disk, then a genuinely separate process (memory_tier_load.rs) resumes
// with replay-mitigated phase 2 - the actual scenario the scaffolding was
// building toward, not three independent proofs.
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

const CORPUS_A: &str = "Shall I compare thee to a summer's day?\n\
Thou art more lovely and more temperate.\n\
Rough winds do shake the darling buds of May,\n\
And summer's lease hath all too short a date.";

const CHECKPOINT_PATH: &str = "memory_tier_checkpoint.txt";
const REPLAY_PATH: &str = "memory_tier_replay.txt";

// Must match memory_tier_load.rs exactly - same reasoning as
// checkpoint_save.rs/checkpoint_load.rs, neither file's shapes are
// self-described.
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
    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, VOCAB_SIZE, D_MODEL);
    let mut pos_emb = Embedding::new(&mut rng, SEQ_LEN, D_MODEL);
    let mut blocks: Vec<TransformerBlock> =
        (0..N_BLOCKS).map(|_| TransformerBlock::new(&mut rng, D_MODEL, N_HEADS, D_FF)).collect();
    let mut final_ln = LayerNorm::new(D_MODEL);
    let mut output_proj = Linear::new(&mut rng, D_MODEL, VOCAB_SIZE);
    let opt = Sgd { lr: 0.3 };

    let steps = 2000;
    println!("phase 1: training on corpus A only ({steps} steps)");
    for step in 0..steps {
        let (input, target) = sample_window(&mut rng, &corpus_a, SEQ_LEN);
        let mut tape = Tape::new();
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
        if step % 500 == 0 {
            println!("  step {step:>4}: loss = {:.4}", tape.value(loss).data[0]);
        }
    }
    let loss_a_after_a = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, SEQ_LEN);
    println!("loss on corpus A after phase 1: {loss_a_after_a:.6}");

    // Same bounded-snapshot reasoning as catastrophic_forgetting.rs: a
    // small FIXED sample taken once, not full ongoing access to corpus A -
    // the point of persisting it is that a later process genuinely
    // wouldn't have corpus A available to resample from.
    let mut snapshot_rng = Rng::new(99);
    let replay_buffer: Vec<(Vec<usize>, Vec<usize>)> =
        (0..REPLAY_WINDOWS).map(|_| sample_window(&mut snapshot_rng, &corpus_a, SEQ_LEN)).collect();

    let mut flat = token_emb.to_flat();
    flat.extend(pos_emb.to_flat());
    for block in &blocks {
        flat.extend(block.to_flat());
    }
    flat.extend(final_ln.to_flat());
    flat.extend(output_proj.to_flat());
    let weights_text: String = flat.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(" ");
    fs::write(CHECKPOINT_PATH, weights_text).expect("failed to write checkpoint");
    println!("saved {} floats to {CHECKPOINT_PATH}", flat.len());

    // Replay buffer as a flat list of token ids (input window then target
    // window, per snapshot entry) - same "no headers, hardcoded shapes on
    // both sides" format as the weight checkpoint above, just usize
    // instead of f32.
    let mut replay_flat: Vec<usize> = Vec::with_capacity(REPLAY_WINDOWS * 2 * SEQ_LEN);
    for (input, target) in &replay_buffer {
        replay_flat.extend(input);
        replay_flat.extend(target);
    }
    let replay_text: String = replay_flat.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ");
    fs::write(REPLAY_PATH, replay_text).expect("failed to write replay buffer");
    println!("saved {REPLAY_WINDOWS} replay windows ({} tokens) to {REPLAY_PATH}", replay_flat.len());

    println!("\nrun `cargo run --release --example memory_tier_load` next - a separate process, no memory of phase 1 beyond these two files.");
}
