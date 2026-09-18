use engine::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng, TransformerBlock, TransformerBlockOut};
use engine::optim::Sgd;
use engine::tape::{Tape, Var};
use std::time::Instant;

fn encode_bytes(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
}

fn decode_bytes(ids: &[usize]) -> String {
    let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn sample_window(rng: &mut Rng, corpus: &[usize], seq_len: usize) -> (Vec<usize>, Vec<usize>) {
    let max_start = corpus.len() - seq_len - 1;
    let start = (rng.next_f32() * max_start as f32) as usize;
    (corpus[start..start + seq_len].to_vec(), corpus[start + 1..start + seq_len + 1].to_vec())
}

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

/// Mean cross-entropy over `n_windows` random windows, no gradient update -
/// pure evaluation. Called on the held-out region to measure generalization
/// (loss on text the model never trained on), and separately on the train
/// region for a directly comparable in-sample number.
fn eval_loss(rng: &mut Rng, token_emb: &Embedding, pos_emb: &Embedding, blocks: &[TransformerBlock], final_ln: &LayerNorm, output_proj: &Linear, corpus: &[usize], seq_len: usize, n_windows: usize) -> f32 {
    let mut total = 0.0;
    for _ in 0..n_windows {
        let (input, target) = sample_window(rng, corpus, seq_len);
        let mut tape = Tape::new();
        let (logits, _) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        total += tape.value(loss).data[0];
    }
    total / n_windows as f32
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
    // Real, diverse public-domain prose (not a repeated stress-test corpus
    // like tiny_lm_scaled.rs) - Project Gutenberg ebook #21, "Three Hundred
    // Aesop's Fables" (George Fyler Townsend translation), Gutenberg
    // boilerplate header/footer and the trailing alphabetical index
    // stripped, fetched verbatim rather than reproduced from memory (same
    // accuracy-risk reasoning as every other corpus choice in this project).
    let full_text = include_str!("../data/aesops_fables.txt");
    let full_encoded = encode_bytes(full_text);

    // 90/10 train/held-out split - held-out text is never sampled during
    // training, only used to measure whether the model generalizes to
    // unseen text or just memorizes, the actual point of using a bigger,
    // non-repeated corpus instead of just scaling the model again.
    let split = (full_encoded.len() as f32 * 0.9) as usize;
    let (train, held_out) = full_encoded.split_at(split);
    println!("corpus: {} bytes total, {} train / {} held-out", full_encoded.len(), train.len(), held_out.len());

    // Same model scale as tiny_lm_scaled.rs - isolates the corpus-size axis
    // on its own, rather than compounding it with a model-size change too.
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
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

    // Same step count as tiny_lm_scaled.rs, for a directly comparable
    // trajectory - measure before assuming more are needed, same discipline
    // as every other scale-up in this project. One step only ever sees one
    // 64-byte window regardless of corpus size, so this corpus (100x
    // bigger) won't get full coverage in 4000 steps - an honest first
    // datapoint, not a claim of exhaustive training.
    let steps = 4000;
    let start_time = Instant::now();
    for step in 0..=steps {
        let (input, target) = sample_window(&mut rng, train, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);

        if step % 400 == 0 {
            let train_eval = eval_loss(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, train, seq_len, 20);
            let held_out_eval = eval_loss(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, held_out, seq_len, 20);
            println!(
                "step {step:>5}: train_loss = {:.4}, eval train = {train_eval:.4}, eval held-out = {held_out_eval:.4} ({:.1}s elapsed)",
                tape.value(loss).data[0], start_time.elapsed().as_secs_f32()
            );
        }
    }
    println!("total training time: {:.1}s for {steps} steps", start_time.elapsed().as_secs_f32());

    println!("\ngenerated (seed \"The Fox\", temperature=0.8):");
    let seed = encode_bytes("The Fox");
    let generated = generate(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &seed, 200, seq_len, 0.8);
    println!("{}", decode_bytes(&generated));
}
