// Direct follow-up to memory_tier_diff.rs's counter-intuitive result:
// replay caused MORE overall weight drift from baseline than no-replay,
// not less. Open question was whether that was a one-off, or whether
// drift actually scales with how much replay happens. Sweeps replay_prob
// across several values (0.0 replicates the no-replay control, 1.0 trains
// on nothing but the tiny 8-window snapshot) from the SAME loaded
// baseline, measuring both weight drift and forgetting for each -
// answers "does drift track replay_prob monotonically" directly, in one
// process, no per-value separate checkpoint files needed.
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

const CORPUS_B: &str = "Twinkle, twinkle, little star,\n\
How I wonder what you are!\n\
Up above the world so high,\n\
Like a diamond in the sky.";

const CHECKPOINT_PATH: &str = "memory_tier_checkpoint.txt";
const REPLAY_PATH: &str = "memory_tier_replay.txt";

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

fn flat_of(token_emb: &Embedding, pos_emb: &Embedding, blocks: &[TransformerBlock], final_ln: &LayerNorm, output_proj: &Linear) -> Vec<f32> {
    let mut flat = token_emb.to_flat();
    flat.extend(pos_emb.to_flat());
    for block in blocks {
        flat.extend(block.to_flat());
    }
    flat.extend(final_ln.to_flat());
    flat.extend(output_proj.to_flat());
    flat
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

fn main() {
    let corpus_a = encode_bytes(CORPUS_A);
    let corpus_b = encode_bytes(CORPUS_B);

    let weights_text = fs::read_to_string(CHECKPOINT_PATH)
        .unwrap_or_else(|_| panic!("couldn't read {CHECKPOINT_PATH} - run memory_tier_save first"));
    let flat: Vec<f32> = weights_text.split_whitespace().map(|s| s.parse().expect("bad float in checkpoint")).collect();
    let mut offset = 0usize;
    let base_token_emb = Embedding::from_flat(&flat, &mut offset, VOCAB_SIZE, D_MODEL);
    let base_pos_emb = Embedding::from_flat(&flat, &mut offset, SEQ_LEN, D_MODEL);
    let base_blocks: Vec<TransformerBlock> =
        (0..N_BLOCKS).map(|_| TransformerBlock::from_flat(&flat, &mut offset, D_MODEL, N_HEADS, D_FF)).collect();
    let base_final_ln = LayerNorm::from_flat(&flat, &mut offset, D_MODEL);
    let base_output_proj = Linear::from_flat(&flat, &mut offset, D_MODEL, VOCAB_SIZE);
    assert_eq!(offset, flat.len(), "checkpoint had leftover/missing floats - architecture mismatch");
    let baseline_flat = flat_of(&base_token_emb, &base_pos_emb, &base_blocks, &base_final_ln, &base_output_proj);
    let loss_a_baseline = eval_loss(&base_token_emb, &base_pos_emb, &base_blocks, &base_final_ln, &base_output_proj, &corpus_a, SEQ_LEN);
    println!("loaded baseline: loss on A = {loss_a_baseline:.6}");

    let replay_text = fs::read_to_string(REPLAY_PATH)
        .unwrap_or_else(|_| panic!("couldn't read {REPLAY_PATH} - run memory_tier_save first"));
    let replay_flat: Vec<usize> = replay_text.split_whitespace().map(|s| s.parse().expect("bad token id in replay file")).collect();
    let replay_buffer: Vec<(Vec<usize>, Vec<usize>)> = (0..REPLAY_WINDOWS)
        .map(|i| {
            let base = i * 2 * SEQ_LEN;
            (replay_flat[base..base + SEQ_LEN].to_vec(), replay_flat[base + SEQ_LEN..base + 2 * SEQ_LEN].to_vec())
        })
        .collect();

    let opt = Sgd { lr: 0.3 };
    let steps = 2000;

    println!("\nreplay_prob | loss on A | forgetting on A | loss on B | L2 drift from baseline");
    for &replay_prob in &[0.0f32, 0.05, 0.15, 0.30, 0.50, 1.0] {
        let mut token_emb = base_token_emb.clone();
        let mut pos_emb = base_pos_emb.clone();
        let mut blocks = base_blocks.clone();
        let mut final_ln = base_final_ln.clone();
        let mut output_proj = base_output_proj.clone();
        // Same seed at every replay_prob - only the sampling policy
        // differs, isolating replay_prob as the one varying input, same
        // control memory_tier_load.rs/memory_tier_load_no_replay.rs
        // already used against each other.
        let mut rng = Rng::new(2);

        for _ in 0..steps {
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
        }

        let loss_a = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, SEQ_LEN);
        let loss_b = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_b, SEQ_LEN);
        let drift = l2_distance(&baseline_flat, &flat_of(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj));
        println!(
            "  {replay_prob:.2}       | {loss_a:.4}    | {:+.4}          | {loss_b:.4}    | {drift:.4}",
            loss_a - loss_a_baseline
        );
    }
}
