use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;

#[path = "../common/mod.rs"]
mod common;
use common::{apply_grad, encode_bytes, eval_loss, forward, sample_window};

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

/// Average cross-entropy over a full deterministic (non-overlapping) sweep
/// of the corpus - not a tiny held-out slice. This isn't measuring
/// generalization to withheld data (not the point here); it's measuring
/// how well the model currently predicts a corpus it was previously trained
/// on, as a proxy for retained knowledge - exactly what catastrophic-
/// forgetting research actually measures. A full sweep is also less noisy
/// than one or two held-out windows would be, at effectively zero extra cost.
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
    replay: Option<&[(Vec<usize>, Vec<usize>)]>,
    replay_prob: f32,
) {
    for step in 0..steps {
        let (input, target) = match replay {
            Some(buf) if !buf.is_empty() && rng.next_f32() < replay_prob => {
                let idx = (rng.next_f32() * buf.len() as f32) as usize;
                buf[idx].clone()
            }
            _ => sample_window(rng, corpus, seq_len),
        };
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
    train(&mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &corpus_a, seq_len, steps_per_phase, &opt, None, 0.0);
    let loss_a_after_a = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, seq_len);
    println!("  loss on A after training on A: {loss_a_after_a:.4}");

    // Bounded replay snapshot: a small FIXED sample of corpus-A windows,
    // taken once and never updated - not full access to corpus_a itself.
    // Real replay buffers exist because the original data usually isn't
    // fully retrievable in a genuine streaming setting; testing with a
    // deliberately tiny snapshot (not the whole corpus) is the more
    // faithful and more interesting version of this question - does even
    // this little retained data prevent most of the forgetting?
    let mut snapshot_rng = Rng::new(99);
    let replay_buffer: Vec<(Vec<usize>, Vec<usize>)> =
        (0..8).map(|_| sample_window(&mut snapshot_rng, &corpus_a, seq_len)).collect();

    // Fork the phase-1-trained model into an independent copy - both
    // phase-2 conditions (no replay vs with replay) must start from
    // identical weights, so the only thing allowed to differ is the
    // sampling policy.
    let mut token_emb2 = token_emb.clone();
    let mut pos_emb2 = pos_emb.clone();
    let mut blocks2 = blocks.clone();
    let mut final_ln2 = final_ln.clone();
    let mut output_proj2 = output_proj.clone();

    println!("\nphase 2: training on corpus B only, no replay ({steps_per_phase} steps)");
    train(&mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &corpus_b, seq_len, steps_per_phase, &opt, None, 0.0);
    let loss_a_after_b = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_a, seq_len);
    let loss_b_after_b = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &corpus_b, seq_len);
    println!("  loss on A after training on B: {loss_a_after_b:.4}");
    println!("  loss on B after training on B: {loss_b_after_b:.4}");

    // Fresh RNG seed for this run, not continuing the shared stream from
    // above - avoids coupling this run's stochastic training dynamics to
    // the no-replay run's via a shared pseudorandom sequence.
    println!("\nphase 2b: training on corpus B WITH replay (15% of steps drawn from an 8-window snapshot of corpus A, {steps_per_phase} steps)");
    let mut rng2 = Rng::new(2);
    train(
        &mut rng2, &mut token_emb2, &mut pos_emb2, &mut blocks2, &mut final_ln2, &mut output_proj2,
        &corpus_b, seq_len, steps_per_phase, &opt, Some(&replay_buffer), 0.15,
    );
    let loss_a_after_b_replay = eval_loss(&token_emb2, &pos_emb2, &blocks2, &final_ln2, &output_proj2, &corpus_a, seq_len);
    let loss_b_after_b_replay = eval_loss(&token_emb2, &pos_emb2, &blocks2, &final_ln2, &output_proj2, &corpus_b, seq_len);
    println!("  loss on A after training on B WITH replay: {loss_a_after_b_replay:.4}");
    println!("  loss on B after training on B WITH replay: {loss_b_after_b_replay:.4}");

    println!("\nforgetting on A, no replay:   {:+.4}", loss_a_after_b - loss_a_after_a);
    println!("forgetting on A, with replay: {:+.4}", loss_a_after_b_replay - loss_a_after_a);
}
