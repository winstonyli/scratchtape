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

/// Full deterministic sweep, same as memory_tier_multigen.rs's - at
/// ~59KB per phase and seq_len=64 this is ~930 windows per call, forward-
/// only, cheap relative to training.
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

/// Same half-life curation policy as memory_tier_multigen.rs - see its
/// doc comment. Unchanged on purpose: this experiment varies scale, not
/// the policy being tested.
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
    println!(
        "corpus: {} bytes total, split into 4 phases of {} bytes each (last phase gets the remainder)",
        full_encoded.len(),
        quarter
    );

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
            "phase {phase} ({name}): training {steps_per_phase} steps, replay buffer = [{}] ({:.1}s elapsed)",
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
