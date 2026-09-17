use engine::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng, TransformerBlock, TransformerBlockOut};
use engine::optim::Sgd;
use engine::tape::{Tape, Var};

/// Byte-level: fixed 256 vocab, token id = byte value. Duplicated from
/// byte_tokenizer.rs rather than shared - each is a couple of lines,
/// engineering a shared module for this would be more code than it saves.
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

const CORPUS: &str = "Shall I compare thee to a summer's day?\n\
Thou art more lovely and more temperate.\n\
Rough winds do shake the darling buds of May,\n\
And summer's lease hath all too short a date.";

/// Model wiring stays example-local, unlike Embedding/LayerNorm/
/// TransformerBlock themselves - this is the one and only consumer of this
/// specific assembly (token embed + positional embed + N blocks + final
/// norm + output projection), unlike those components which each had 2+
/// known consumers before being promoted to the library.
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

/// Temperature-sampled decoding over a sliding window - no KV-cache (each
/// step recomputes the whole window from scratch), so context is hard-capped
/// at seq_len by the positional embedding table's fixed size. Plain scalar
/// math here, not Tape::softmax - no gradient needed at generation time, so
/// the max-subtraction stability trick (parked for the Tape version) is
/// cheap to apply here for free.
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
    let encoded = encode_bytes(CORPUS);
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (32, 4, 64, 16, 2);
    let vocab_size = 256;

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> =
        (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);

    // Plain SGD, not Adam: TransformerBlock::apply_grad(&Sgd) already
    // exists and cascades correctly through every parameter regardless of
    // model size, since SGD is stateless. Adam here would need a matching
    // per-parameter (m,v,t) state mirrored across ~74 parameter tensors -
    // real bookkeeping not yet earned by evidence. If SGD proves too slow
    // in practice, that becomes the concrete reason to build it.
    let opt = Sgd { lr: 0.3 };

    let steps = 4000;
    for step in 0..=steps {
        let (input, target) = sample_window(&mut rng, &encoded, seq_len);
        let mut tape = Tape::new();
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);

        if step % 400 == 0 {
            println!("step {step:>5}: loss = {:.4}", tape.value(loss).data[0]);
        }
    }

    // 172-byte corpus, many training steps - expect memorization, not
    // generalization. Point of this demo is proving the pipeline trains
    // end-to-end, not demonstrating real language-modeling capability.
    println!("\ngenerated (seed \"Shall I\", temperature=0.8):");
    let seed = encode_bytes("Shall I");
    let generated = generate(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &seed, 80, seq_len, 0.8);
    println!("{}", decode_bytes(&generated));
}
