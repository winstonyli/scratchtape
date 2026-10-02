// Follow-up to memory_tier_multigen_diverse_no_replay.rs's finding that
// budget=8 replay made no measurable difference against genuinely
// different real corpora - traced to 8 fixed windows covering a
// negligible fraction of a ~59KB corpus (~930 possible seq_len=64 windows
// per corpus), unlike the toy demos where budget=8 covered a real share
// of a 4-line corpus. This reruns the same diverse-corpus setup with
// budget=128 (16x), replay_prob unchanged at 0.15, to test whether a
// bigger budget is what replay actually needed at this scale, or whether
// the mechanism itself doesn't transfer regardless of size.
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

/// Same half-life curation policy as memory_tier_multigen_diverse.rs - see
/// its doc comment. Unchanged on purpose: this experiment varies budget
/// size, not the policy being tested.
fn main() {
    // Identical corpus construction to memory_tier_multigen_diverse.rs -
    // see its doc comment for sourcing.
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

    // Same architecture and step budget as memory_tier_multigen_diverse.rs.
    // Only budget differs (128 instead of 8) - the one variable this file
    // tests.
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    let steps_per_phase = 4000;
    let replay_prob = 0.15;
    let budget = 128;

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> = (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    let mut buffer: Vec<(&'static str, Vec<usize>, Vec<usize>)> = Vec::new();
    let mut snapshot_rng = Rng::new(99);
    let start_time = Instant::now();

    for phase in 0..4 {
        let name = labels[phase];
        println!("\nphase {phase} ({name}): training {steps_per_phase} steps, replay buffer = [{}] ({:.1}s elapsed)", composition(&buffer), start_time.elapsed().as_secs_f32());
        train(&mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &encoded[phase], seq_len, steps_per_phase, &opt, &buffer, replay_prob);
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
