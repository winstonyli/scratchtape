use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{ForwardOut, apply_grad, decode_bytes, encode_bytes, sample_window};

/// batch_size independent windows, concatenated in row order (rows
/// [0,seq_len) = sample 0, [seq_len,2*seq_len) = sample 1, ...) - matches
/// what TransformerBlock::forward_batched and the position ids below both
/// assume. Independent sampling, not de-duplicated against each other -
/// standard minibatch SGD, batch-mates overlapping is expected/fine.
fn sample_batch(rng: &mut Rng, corpus: &[usize], seq_len: usize, batch_size: usize) -> (Vec<usize>, Vec<usize>) {
    let mut inputs = Vec::with_capacity(batch_size * seq_len);
    let mut targets = Vec::with_capacity(batch_size * seq_len);
    for _ in 0..batch_size {
        let (inp, tgt) = sample_window(rng, corpus, seq_len);
        inputs.extend(inp);
        targets.extend(tgt);
    }
    (inputs, targets)
}

const SONNET_18: &str = "Shall I compare thee to a summer's day?\n\
Thou art more lovely and more temperate.\n\
Rough winds do shake the darling buds of May,\n\
And summer's lease hath all too short a date.";

const TWINKLE: &str = "Twinkle, twinkle, little star,\n\
How I wonder what you are!\n\
Up above the world so high,\n\
Like a diamond in the sky.";

fn forward(
    tape: &mut Tape,
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    input_ids: &[usize],
    seq_len: usize,
    batch_size: usize,
) -> (Var, ForwardOut) {
    // Positions reset to 0..seq_len for EACH batch chunk - this is a
    // position WITHIN a sample's own sequence, not a row index into the
    // stacked buffer. Unbatched examples get this for free since
    // input_ids.len() == seq_len there; batching makes it a real thing to
    // get right, unlike every other row-independent layer in this forward
    // pass.
    let positions: Vec<usize> = (0..batch_size).flat_map(|_| 0..seq_len).collect();
    let tok_out = token_emb.forward(tape, input_ids);
    let pos_out = pos_emb.forward(tape, &positions);
    let mut x = tape.add(tok_out.y, pos_out.y);

    let mut block_outs = Vec::with_capacity(blocks.len());
    for block in blocks {
        let out = block.forward_batched(tape, x, batch_size);
        x = out.y;
        block_outs.push(out);
    }

    let ln_out = final_ln.forward(tape, x);
    let proj_out = output_proj.forward(tape, ln_out.y);
    let logits = proj_out.y;
    (logits, ForwardOut { tok_out, pos_out, block_outs, ln_out, proj_out })
}

fn generate(
    rng: &mut Rng,
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    seed: &[usize],
    gen_len: usize,
    seq_len: usize,
    temperature: f32,
) -> Vec<usize> {
    // Generation is always batch_size=1 - unbatched forward_batched(...,1)
    // degenerates exactly back to plain per-sample attention.
    let mut sequence = seed.to_vec();
    for _ in 0..gen_len {
        let start = sequence.len().saturating_sub(seq_len);
        let window = sequence[start..].to_vec();
        let window_len = window.len();

        let mut tape = Tape::new();
        let (logits, _) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &window, window_len, 1);
        let logits_val = tape.value(logits);
        let vocab = logits_val.shape[1];
        let last_row = logits_val.shape[0] - 1;
        let last_logits = &logits_val.data[last_row * vocab..last_row * vocab + vocab];

        let scaled: Vec<f32> = last_logits.iter().map(|&x| x / temperature).collect();
        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();

        let r = rng.next_f32();
        let mut cumulative = 0.0;
        let mut chosen = probs.len() - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                chosen = i;
                break;
            }
        }
        sequence.push(chosen);
    }
    sequence
}

fn main() {
    // Same corpus and model dims as tiny_lm_scaled2.rs - isolates the
    // minibatching change on its own, rather than compounding it with
    // another scale-up at the same time.
    let combined = format!("{SONNET_18}\n{TWINKLE}\n");
    let corpus_text = combined.repeat(8);
    let encoded = encode_bytes(&corpus_text);
    println!("corpus: {} bytes ({}x the original tiny_lm.rs corpus)", encoded.len(), encoded.len() / 172);

    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (256, 16, 512, 64, 6);
    let vocab_size = 256;
    // batch_size=16: stacked rows = 1024, so the FFN matmul becomes
    // 1024x256x512 ~ 134M FLOPs - lands almost exactly on the 512-cube
    // benchmark point already measured at ~1.35x GPU win (512^3 = 134M
    // FLOPs too), the actual point of this exercise: prove batching gets
    // real matmul calls past the CPU/GPU crossover found earlier.
    let batch_size = 16;
    println!("model: d_model={d_model} n_heads={n_heads} d_ff={d_ff} seq_len={seq_len} n_blocks={n_blocks} batch_size={batch_size}");

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> = (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.03 };

    // Fewer steps than tiny_lm_scaled2.rs (4000): each step now covers
    // batch_size=16x the samples, so fewer steps see a comparable amount of
    // data - measure before assuming more are needed, same as every other
    // scale-up in this project.
    let steps = 250;
    let start_time = Instant::now();
    for step in 0..=steps {
        let (input, target) = sample_batch(&mut rng, &encoded, seq_len, batch_size);
        // Node count is the SAME as the unbatched model (batching doesn't
        // add tape nodes, just makes each node's NdArray bigger) - same
        // capacity as tiny_lm_scaled2.rs.
        let mut tape = Tape::with_capacity(6000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input, seq_len, batch_size);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);

        if step % 25 == 0 {
            println!("step {step:>5}: loss = {:.4} ({:.1}s elapsed)", tape.value(loss).data[0], start_time.elapsed().as_secs_f32());
        }
    }
    println!("total training time: {:.1}s for {steps} steps ({} samples/step)", start_time.elapsed().as_secs_f32(), batch_size);

    println!("\ngenerated (seed \"Shall I\", temperature=0.8):");
    let seed = encode_bytes("Shall I");
    let generated = generate(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &seed, 150, seq_len, 0.8);
    println!("{}", decode_bytes(&generated));
}
