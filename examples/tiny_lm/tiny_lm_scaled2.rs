use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{apply_grad, decode_bytes, encode_bytes, forward, sample_window};

const SONNET_18: &str = "Shall I compare thee to a summer's day?\n\
Thou art more lovely and more temperate.\n\
Rough winds do shake the darling buds of May,\n\
And summer's lease hath all too short a date.";

const TWINKLE: &str = "Twinkle, twinkle, little star,\n\
How I wonder what you are!\n\
Up above the world so high,\n\
Like a diamond in the sky.";

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
    let mut sequence = seed.to_vec();
    for _ in 0..gen_len {
        let start = sequence.len().saturating_sub(seq_len);
        let window = sequence[start..].to_vec();

        let mut tape = Tape::new();
        let (logits, _) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &window);
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
    // Same corpus as tiny_lm_scaled.rs - this file isolates the MODEL-size
    // axis (vs a separate future corpus-size experiment), so the data stays
    // fixed while architecture grows.
    let combined = format!("{SONNET_18}\n{TWINKLE}\n");
    let corpus_text = combined.repeat(8);
    let encoded = encode_bytes(&corpus_text);
    println!("corpus: {} bytes ({}x the original tiny_lm.rs corpus)", encoded.len(), encoded.len() / 172);

    // 2x jump on d_model/n_heads/d_ff, 1.5x on n_blocks from tiny_lm_scaled.rs
    // - another defensible-not-extreme step, meant to re-stress the matmul
    // path just optimized (cache-blocking + AVX2/FMA + broadcast fast path)
    // with real FLOPs growth (~6x estimated) before deciding whether GPU is
    // actually justified. seq_len held fixed - isolating model size from
    // context length, which belongs to a separate corpus-scale experiment.
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (256, 16, 512, 64, 6);
    let vocab_size = 256;
    println!("model: d_model={d_model} n_heads={n_heads} d_ff={d_ff} seq_len={seq_len} n_blocks={n_blocks}");

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> =
        (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    let steps = 4000;
    let start_time = Instant::now();
    for step in 0..=steps {
        let (input, target) = sample_window(&mut rng, &encoded, seq_len);
        // ~3x tiny_lm_scaled.rs's capacity (2x from n_heads, 1.5x from
        // n_blocks - node count scales with op count, not tensor size, so
        // d_model/d_ff growth alone doesn't add nodes).
        let mut tape = Tape::with_capacity(6000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);

        if step % 400 == 0 {
            println!("step {step:>5}: loss = {:.4} ({:.1}s elapsed)", tape.value(loss).data[0], start_time.elapsed().as_secs_f32());
        }
    }
    println!("total training time: {:.1}s for {steps} steps", start_time.elapsed().as_secs_f32());

    println!("\ngenerated (seed \"Shall I\", temperature=0.8):");
    let seed = encode_bytes("Shall I");
    let generated = generate(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &seed, 150, seq_len, 0.8);
    println!("{}", decode_bytes(&generated));
}
