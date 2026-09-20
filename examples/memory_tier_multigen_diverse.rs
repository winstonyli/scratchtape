// Rerun of memory_tier_multigen_scaled.rs's 4-phase continual-learning
// experiment, but with the confound it surfaced removed: that run quartered
// ONE book (Aesop's Fables) and found essentially no forgetting, traced to
// all 4 "phases" sharing the same vocabulary/register/byte statistics - not
// a real different-task sequence. This run swaps the 4 quarters for 4
// genuinely different real public-domain texts (fable, detective fiction,
// scientific argument, free verse), each trimmed to roughly the same byte
// budget as one of the old quarters (~58-60KB), so scale and step count
// are unchanged and vocabulary divergence is the only thing that changed.
use scratchtape::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng, TransformerBlock, TransformerBlockOut};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};
use std::time::Instant;

fn encode_bytes(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
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
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, token_emb, pos_emb, blocks, final_ln, output_proj, opt);
    }
}

/// Same half-life curation policy as memory_tier_multigen_scaled.rs - see
/// its doc comment. Unchanged on purpose: this experiment varies corpus
/// divergence, not the policy being tested.
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
    // 4 genuinely different real texts, not quarters of one book - fable,
    // detective fiction, scientific argument, free verse. Phase A is
    // truncated to the same ~59.5KB the other three were independently
    // trimmed to (Project Gutenberg source, boilerplate stripped, cut at a
    // paragraph boundary), so no phase gets a size advantage.
    let aesop_full = encode_bytes(include_str!("../data/aesops_fables.txt"));
    let encoded: Vec<Vec<usize>> = vec![
        aesop_full[..59470].to_vec(),
        encode_bytes(include_str!("../data/sherlock_holmes.txt")),
        encode_bytes(include_str!("../data/origin_of_species.txt")),
        encode_bytes(include_str!("../data/leaves_of_grass.txt")),
    ];
    let labels = ["A(fables)", "B(holmes)", "C(origin)", "D(whitman)"];
    for (label, corpus) in labels.iter().zip(encoded.iter()) {
        println!("corpus {label}: {} bytes", corpus.len());
    }

    // Same architecture and step budget as memory_tier_multigen_scaled.rs -
    // scale and total compute held fixed, corpus divergence is the only
    // variable being changed by this file.
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    let steps_per_phase = 4000;
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
    let start_time = Instant::now();

    for phase in 0..4 {
        let name = labels[phase];
        println!(
            "\nphase {phase} ({name}): training {steps_per_phase} steps, replay buffer = [{}] ({:.1}s elapsed)",
            composition(&buffer),
            start_time.elapsed().as_secs_f32()
        );
        train(
            &mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj,
            &encoded[phase], seq_len, steps_per_phase, &opt, &buffer, replay_prob,
        );
        buffer = curate(buffer, name, &encoded[phase], seq_len, budget, &mut snapshot_rng);

        print!("  loss now on: ");
        for j in 0..=phase {
            let loss = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &encoded[j], seq_len);
            print!("{}={loss:.3} ", labels[j]);
        }
        println!();
    }
    println!("\ntotal training time: {:.1}s", start_time.elapsed().as_secs_f32());

    println!("\nreplay buffer after the final phase's curation = [{}] (not used for further training - no phase 4 exists)", composition(&buffer));
}
