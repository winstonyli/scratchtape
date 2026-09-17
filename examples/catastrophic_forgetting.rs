use engine::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng, TransformerBlock, TransformerBlockOut};
use engine::optim::Sgd;
use engine::tape::{Tape, Var};

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

// Deliberately different register/vocabulary from CORPUS_A (not just a
// different sonnet) - makes forgetting easier to see/measure, since the two
// tasks' byte distributions differ more starkly.
const CORPUS_B: &str = "Twinkle, twinkle, little star,\n\
How I wonder what you are!\n\
Up above the world so high,\n\
Like a diamond in the sky.";

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

/// Average cross-entropy over a full deterministic (non-overlapping) sweep
/// of the corpus - not a tiny held-out slice. This isn't measuring
/// generalization to withheld data (not the point here); it's measuring
/// how well the model currently predicts a corpus it was previously trained
/// on, as a proxy for retained knowledge - exactly what catastrophic-
/// forgetting research actually measures. A full sweep is also less noisy
/// than one or two held-out windows would be, at effectively zero extra cost.
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
    for step in 0..steps {
        let (input, target) = sample_window(rng, corpus, seq_len);
        let mut tape = Tape::new();
        let (logits, out) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, token_emb, pos_emb, blocks, final_ln, output_proj, opt);

        if step % 500 == 0 {
            println!("    step {step:>4}: loss = {:.4}", tape.value(loss).data[0]);
        }
    }
}

fn main() {
    let corpus_a = encode_bytes(CORPUS_A);
    let corpus_b = encode_bytes(CORPUS_B);
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (32, 4, 64, 16, 2);
    let vocab_size = 256;
    let steps_per_phase = 2000;

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> =
        (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    println!("phase 1: training on corpus A only ({steps_per_phase} steps)");
    train(&mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &corpus_a, seq_len, steps_per_phase, &opt);
    let loss_a_after_a = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, seq_len);
    println!("  loss on A after training on A: {loss_a_after_a:.4}");

    let loss_b_before_phase2 = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_b, seq_len);
    println!("  [diagnostic] loss on B BEFORE any phase-2 training: {loss_b_before_phase2:.4}");

    println!("\nphase 2: training on corpus B only, no replay ({steps_per_phase} steps)");
    train(&mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &corpus_b, seq_len, steps_per_phase, &opt);
    let loss_a_after_b = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, seq_len);
    let loss_b_after_b = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_b, seq_len);
    println!("  loss on A after training on B: {loss_a_after_b:.4}");
    println!("  loss on B after training on B: {loss_b_after_b:.4}");

    println!("\nforgetting on A: {:+.4} (loss increase from phase 1 to after phase 2)", loss_a_after_b - loss_a_after_a);
}
