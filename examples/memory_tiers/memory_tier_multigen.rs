// Extends the 2-corpus memory-tier demos to a 4-corpus SEQUENCE (A, B, C,
// D) - the actual continual-learning question none of the earlier
// demos asked: with only two phases, the replay buffer never needs
// curating (one fixed snapshot, taken once, used once). With a longer
// sequence it does - a fixed 8-window budget can't hold every corpus
// ever seen, so something has to be evicted every phase. Tests a simple,
// describable policy (evict the oldest half, add fresh windows from the
// corpus just finished) and asks the natural question: does "generations
// since a corpus was last replayed" predict how forgotten it ends up?
use scratchtape::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng, TransformerBlock, TransformerBlockOut};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};

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

// Traditional/public-domain nursery rhymes, same choice of genre as
// CORPUS_B - the criterion that actually matters (established by
// catastrophic_forgetting.rs's own reasoning) is byte-distribution
// divergence between corpora, not genre difference.
const CORPUS_C: &str = "Baa, baa, black sheep,\n\
Have you any wool?\n\
Yes sir, yes sir,\n\
Three bags full.";

const CORPUS_D: &str = "Jack and Jill went up the hill\n\
To fetch a pail of water.\n\
Jack fell down and broke his crown,\n\
And Jill came tumbling after.";

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
    replay: &[(&'static str, Vec<usize>, Vec<usize>)],
    replay_prob: f32,
) {
    for _ in 0..steps {
        let (input, target) = if !replay.is_empty() && rng.next_f32() < replay_prob {
            let idx = (rng.next_f32() * replay.len() as f32) as usize;
            let (_, input, target) = &replay[idx];
            (input.clone(), target.clone())
        } else {
            sample_window(rng, corpus, seq_len)
        };
        let mut tape = Tape::new();
        let (logits, out) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, token_emb, pos_emb, blocks, final_ln, output_proj, opt);
    }
}

/// Fixed BUDGET-sized buffer, half-life curation: evict the oldest
/// BUDGET/2 entries (the front of the Vec - older pushes sit at lower
/// indices), then fill the freed slots with fresh windows sampled from
/// the corpus that phase just finished. Means any one corpus's windows
/// survive roughly one extra generation past the phase that added them,
/// then are gone - simple and fully described, not tuned for a
/// particular outcome.
fn curate(
    mut buffer: Vec<(&'static str, Vec<usize>, Vec<usize>)>,
    just_finished_label: &'static str,
    just_finished_corpus: &[usize],
    seq_len: usize,
    budget: usize,
    rng: &mut Rng,
) -> Vec<(&'static str, Vec<usize>, Vec<usize>)> {
    let evict = (budget / 2).min(buffer.len());
    buffer.drain(0..evict);
    while buffer.len() < budget {
        let (input, target) = sample_window(rng, just_finished_corpus, seq_len);
        buffer.push((just_finished_label, input, target));
    }
    buffer
}

/// Tally of how many buffer slots currently come from each corpus label -
/// the actual, factual composition, printed instead of a hand-narrated
/// guess about which corpora "should" still be represented.
fn composition(buffer: &[(&'static str, Vec<usize>, Vec<usize>)]) -> String {
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for (label, _, _) in buffer {
        if let Some(entry) = counts.iter_mut().find(|(l, _)| l == label) {
            entry.1 += 1;
        } else {
            counts.push((label, 1));
        }
    }
    counts.iter().map(|(l, c)| format!("{c}x{l}")).collect::<Vec<_>>().join(", ")
}

fn main() {
    let corpora: [(&str, &str); 4] = [("A", CORPUS_A), ("B", CORPUS_B), ("C", CORPUS_C), ("D", CORPUS_D)];
    let encoded: Vec<Vec<usize>> = corpora.iter().map(|&(_, text)| encode_bytes(text)).collect();

    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (32, 4, 64, 16, 2);
    let vocab_size = 256;
    let steps_per_phase = 2000;
    let replay_prob = 0.15;
    let budget = 8;

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> =
        (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    let mut buffer: Vec<(&'static str, Vec<usize>, Vec<usize>)> = Vec::new();
    let mut snapshot_rng = Rng::new(99);

    for (phase, &(name, _)) in corpora.iter().enumerate() {
        println!("phase {phase} ({name}): training {steps_per_phase} steps, replay buffer = [{}]", composition(&buffer));
        train(
            &mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj,
            &encoded[phase], seq_len, steps_per_phase, &opt, &buffer, replay_prob,
        );
        buffer = curate(buffer, name, &encoded[phase], seq_len, budget, &mut snapshot_rng);

        print!("  loss now on: ");
        for (j, &(name_j, _)) in corpora.iter().enumerate().take(phase + 1) {
            let loss = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &encoded[j], seq_len);
            print!("{name_j}={loss:.3} ");
        }
        println!();
    }

    println!("\nreplay buffer after the final phase's curation = [{}] (not used for further training - no phase 4 exists)", composition(&buffer));
}
