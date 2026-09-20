// Rerun of memory_tier_multigen_scaled.rs's 4-phase continual-learning
// experiment, but with the confound it surfaced removed: that run quartered
// ONE book (Aesop's Fables) and found essentially no forgetting, traced to
// all 4 "phases" sharing the same vocabulary/register/byte statistics - not
// a real different-task sequence. This run swaps the 4 quarters for 4
// genuinely different real public-domain texts (fable, detective fiction,
// scientific argument, free verse), each trimmed to roughly the same byte
// budget as one of the old quarters (~58-60KB), so scale and step count
// are unchanged and vocabulary divergence is the only thing that changed.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{apply_grad, composition, curate, encode_bytes, eval_loss, forward, sample_window};

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
fn main() {
    // 4 genuinely different real texts, not quarters of one book - fable,
    // detective fiction, scientific argument, free verse. Phase A is
    // truncated to the same ~59.5KB the other three were independently
    // trimmed to (Project Gutenberg source, boilerplate stripped, cut at a
    // paragraph boundary), so no phase gets a size advantage.
    let aesop_full = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let encoded: Vec<Vec<usize>> = vec![
        aesop_full[..59470].to_vec(),
        encode_bytes(include_str!("../../data/sherlock_holmes.txt")),
        encode_bytes(include_str!("../../data/origin_of_species.txt")),
        encode_bytes(include_str!("../../data/leaves_of_grass.txt")),
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
