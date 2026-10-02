// Scaled-up rerun of memory_tier_multigen.rs's 4-phase continual-learning
// experiment - same design, same questions, but at real scale instead of
// toy scale. Every memory-tier demo so far (catastrophic_forgetting.rs
// through memory_tier_multigen.rs) used 4-line nursery-rhyme corpora at
// d_model=32; this uses tiny_lm_corpus.rs's proven real-corpus scale
// (d_model=128, Aesop's Fables) instead, split into 4 sequential phases
// by contiguous byte range rather than 4 separate short texts. Tests
// whether the toy-scale findings (threshold-not-dial replay effect, even
// actively-replayed corpora still forgetting hard, new-task learning
// getting harder as generations accumulate) hold up, or were artifacts of
// running everything on texts small enough to fit in a handful of
// gradient steps - the same "is this real or a toy-scale artifact"
// instinct already applied to the KR&R line's compute-matched rerun,
// aimed at the memory-tier line instead.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{apply_grad, composition, curate, encode_bytes, eval_loss, forward, sample_window};

/// Full deterministic sweep, same as memory_tier_multigen.rs's - at
/// ~59KB per phase and seq_len=64 this is ~930 windows per call, forward-
/// only, cheap relative to training.
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

/// Same half-life curation policy as memory_tier_multigen.rs - see its
/// doc comment. Unchanged on purpose: this experiment varies scale, not
/// the policy being tested.
fn main() {
    // Same corpus tiny_lm_corpus.rs uses, split into 4 contiguous quarters
    // by byte range - not by fable boundary, which would need parsing
    // structure this file has no reason to add. Each quarter still spans
    // many distinct fables (Aesop's Fables has hundreds), so this isn't a
    // single repeated theme the way it might sound - it's simply "the
    // first/second/third/fourth stretch of a long stream of real prose."
    let full_text = include_str!("../../data/aesops_fables.txt");
    let full_encoded = encode_bytes(full_text);
    let quarter = full_encoded.len() / 4;
    let encoded: Vec<Vec<usize>> = (0..4)
        .map(|i| {
            let end = if i == 3 { full_encoded.len() } else { (i + 1) * quarter };
            full_encoded[i * quarter..end].to_vec()
        })
        .collect();
    let labels = ["A", "B", "C", "D"];
    println!("corpus: {} bytes total, split into 4 phases of {} bytes each (last phase gets the remainder)", full_encoded.len(), quarter);

    // Same architecture tiny_lm_corpus.rs's whole KR&R investigation was
    // built on - the "real scale" this experiment is testing against,
    // not the toy d_model=32 every earlier memory-tier demo used.
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    // 4000 steps/phase, matching tiny_lm.rs's own first proven convergence
    // budget - 4 phases x 4000 = 16000 total steps, the same total step
    // count tiny_lm_corpus.rs's single-corpus runs used, so this spends
    // the same total training compute just distributed across 4
    // sequential phases instead of one continuous run.
    let steps_per_phase = 4000;
    let replay_prob = 0.15;
    let budget = 8;

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
        println!("phase {phase} ({name}): training {steps_per_phase} steps, replay buffer = [{}] ({:.1}s elapsed)", composition(&buffer), start_time.elapsed().as_secs_f32());
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
