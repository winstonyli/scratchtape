// Not every consuming binary uses every function here, and `pub` alone
// doesn't suppress dead_code in a binary crate the way it would in a
// library (no external crate can be "using" the rest) - one blanket
// allow for the whole shared module, rather than one per function.
#![allow(dead_code)]

// Shared helpers pulled out of examples/ after they crossed this
// project's own "2+ consumers" promotion bar many times over -
// encode_bytes alone was byte-identical across 22 files. Kept here, not
// in src/, since none of this is an engine primitive - it's example
// glue (byte encoding, window sampling, precision/recall scoring, the
// standard tiny-transformer forward/apply_grad pair). Cargo doesn't
// auto-discover or share code between examples on its own, so each
// consuming file pulls this in explicitly:
//
//   #[path = "../common/mod.rs"]
//   mod common;
//   use common::{encode_bytes, sample_window, ...};
//
// Every item here is `pub` (needed so consumers can import it at all;
// the dead_code allow above handles the "unused by this particular
// binary" side of it).
//
// Not everything duplicated across examples ended up here - `train()`'s
// body genuinely differs per experiment (that's the actual variable
// being tested in each file, not copy-paste residue), and
// tiny_lm_batched.rs's `forward`/`generate` differ for a real reason
// (batched blocks need a different position calc and call
// `forward_batched`, not `forward`) - both were left local rather than
// forced into a shared shape that would misrepresent them as identical.
use scratchtape::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng, TransformerBlock, TransformerBlockOut};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};
use scratchtape::tensor::NdArray;

pub fn encode_bytes(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
}

pub fn decode_bytes(ids: &[usize]) -> String {
    let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

pub fn sample_window(rng: &mut Rng, corpus: &[usize], seq_len: usize) -> (Vec<usize>, Vec<usize>) {
    let max_start = corpus.len() - seq_len - 1;
    let start = (rng.next_f32() * max_start as f32) as usize;
    (corpus[start..start + seq_len].to_vec(), corpus[start + 1..start + seq_len + 1].to_vec())
}

pub fn shuffle(items: &mut [usize], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize;
        items.swap(i, j);
    }
}

/// Precision/recall/F1 for a binary classifier's predictions against
/// true labels - most categories these get used on are well under 50%
/// positive, so accuracy alone can hide a rule that never fires
/// correctly (high accuracy, zero recall).
pub fn precision_recall_f1(labels: &[usize], predictions: &[usize]) -> (f32, f32, f32) {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_ = 0;
    for i in 0..labels.len() {
        match (predictions[i], labels[i]) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }
    }
    let precision = if tp + fp == 0 { 0.0 } else { tp as f32 / (tp + fp) as f32 };
    let recall = if tp + fn_ == 0 { 0.0 } else { tp as f32 / (tp + fn_) as f32 };
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    (precision, recall, f1)
}

pub fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Differentiable sigmoid built from existing tape ops (exp/div/add) -
/// no dedicated Tape::sigmoid primitive exists, this composes one.
pub fn sigmoid(tape: &mut Tape, x: Var) -> Var {
    let neg_x = tape.scale(x, -1.0);
    let exp_neg_x = tape.exp(neg_x);
    let one = tape.leaf(NdArray::new(vec![1.0; tape.value(exp_neg_x).data.len()], tape.value(exp_neg_x).shape.clone()));
    let denom = tape.add(one, exp_neg_x);
    tape.div(one, denom)
}

pub struct ForwardOut {
    pub tok_out: EmbeddingOut,
    pub pos_out: EmbeddingOut,
    pub block_outs: Vec<TransformerBlockOut>,
    pub ln_out: LayerNormOut,
    pub proj_out: LinearOut,
}

/// The standard byte-level transformer forward pass: token + positional
/// embedding, a stack of causal transformer blocks, final layer norm,
/// output projection. Shared by every example that trains this exact
/// architecture (not `tiny_lm_batched.rs`, which needs `forward_batched`
/// and a different position calc for its batched blocks instead).
pub fn forward(
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

pub fn apply_grad(
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

/// The byte-level transformer's parts, for examples that train and score it
/// as one unit (the batched counterpart of `forward`/`eval_loss` above).
pub struct Model {
    pub token_emb: Embedding,
    pub pos_emb: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub final_ln: LayerNorm,
    pub output_proj: Linear,
}

impl Model {
    /// `input_ids` holds `batch` equal windows stacked row-wise; positions
    /// restart at 0 for each window (see tiny_lm_batched.rs).
    pub fn forward(&self, tape: &mut Tape, input_ids: &[usize], batch: usize, softmax1: bool) -> (Var, ForwardOut) {
        let positions: Vec<usize> = (0..batch).flat_map(|_| 0..input_ids.len() / batch).collect();
        let tok_out = self.token_emb.forward(tape, input_ids);
        let pos_out = self.pos_emb.forward(tape, &positions);
        let mut x = tape.add(tok_out.y, pos_out.y);
        let mut block_outs = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let out = block.forward_full(tape, x, batch, softmax1);
            x = out.y;
            block_outs.push(out);
        }
        let ln_out = self.final_ln.forward(tape, x);
        let proj_out = self.output_proj.forward(tape, ln_out.y);
        (proj_out.y, ForwardOut { tok_out, pos_out, block_outs, ln_out, proj_out })
    }

    /// Mean CE over every non-overlapping `seq_len` window of `corpus` - the
    /// same windowing ngram_baseline.rs and attention_uniformity_check.rs use.
    pub fn full_ce(&self, corpus: &[usize], seq_len: usize, softmax1: bool) -> f32 {
        let starts: Vec<usize> = (0..corpus.len() - seq_len).step_by(seq_len).collect();
        let mut total = 0.0;
        for &s in &starts {
            let mut tape = Tape::new();
            let (logits, _) = self.forward(&mut tape, &corpus[s..s + seq_len], 1, softmax1);
            let loss = tape.cross_entropy(logits, &corpus[s + 1..s + seq_len + 1]);
            total += tape.value(loss).data[0];
        }
        total / starts.len() as f32
    }
}

/// Deterministic sliding-window sweep over the whole corpus (not random
/// sampling) - used by every memory-tier example for a reproducible
/// loss readout. `tiny_lm_corpus.rs` has its own separate randomly-
/// sampled `eval_loss` (different signature, takes an `Rng` and a
/// window count) for a genuinely different purpose - kept local there,
/// not merged into this one.
pub fn eval_loss(token_emb: &Embedding, pos_emb: &Embedding, blocks: &[TransformerBlock], final_ln: &LayerNorm, output_proj: &Linear, corpus: &[usize], seq_len: usize) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    let mut start = 0;
    while start + seq_len < corpus.len() {
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

/// Half-life replay-buffer curation: evict the oldest half, fill freed
/// slots with fresh windows from the corpus that just finished. Shared
/// by the whole memory_tier_multigen* family - see any of those files'
/// doc comments for the policy's full reasoning.
pub fn curate(
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

pub fn composition(buffer: &[(&'static str, Vec<usize>, Vec<usize>)]) -> String {
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

/// Concatenates every parameter into one flat vector, fixed order
/// (token_emb, pos_emb, blocks..., final_ln, output_proj) - the same
/// order `reconstruct` below expects. Second real consumer
/// (memory_tier_multigen_diverse_consolidate.rs and
/// memory_tier_multigen_diverse_consolidate_fisher.rs) is what promoted
/// this out of being copy-pasted between them.
pub fn flatten_all(token_emb: &Embedding, pos_emb: &Embedding, blocks: &[TransformerBlock], final_ln: &LayerNorm, output_proj: &Linear) -> Vec<f32> {
    let mut flat = token_emb.to_flat();
    flat.extend(pos_emb.to_flat());
    for block in blocks {
        flat.extend(block.to_flat());
    }
    flat.extend(final_ln.to_flat());
    flat.extend(output_proj.to_flat());
    flat
}

/// Inverse of flatten_all - rebuilds each component from its slice of
/// the flat vector, in the same fixed order.
pub fn reconstruct(
    flat: &[f32],
    vocab_size: usize,
    d_model: usize,
    seq_len: usize,
    n_blocks: usize,
    n_heads: usize,
    d_ff: usize,
) -> (Embedding, Embedding, Vec<TransformerBlock>, LayerNorm, Linear) {
    let mut offset = 0usize;
    let token_emb = Embedding::from_flat(flat, &mut offset, vocab_size, d_model);
    let pos_emb = Embedding::from_flat(flat, &mut offset, seq_len, d_model);
    let blocks = (0..n_blocks).map(|_| TransformerBlock::from_flat(flat, &mut offset, d_model, n_heads, d_ff)).collect();
    let final_ln = LayerNorm::from_flat(flat, &mut offset, d_model);
    let output_proj = Linear::from_flat(flat, &mut offset, d_model, vocab_size);
    assert_eq!(offset, flat.len(), "flat vector had leftover/missing floats - architecture mismatch");
    (token_emb, pos_emb, blocks, final_ln, output_proj)
}

/// One of the bundled corpora by file name without `.txt` (`aesops_fables`,
/// `sherlock_holmes`, `leaves_of_grass`, `origin_of_species`).
#[allow(dead_code)]
pub fn corpus(name: &str) -> &'static str {
    match name {
        "aesops_fables" => include_str!("../../data/aesops_fables.txt"),
        "sherlock_holmes" => include_str!("../../data/sherlock_holmes.txt"),
        "leaves_of_grass" => include_str!("../../data/leaves_of_grass.txt"),
        "origin_of_species" => include_str!("../../data/origin_of_species.txt"),
        _ => panic!("unknown corpus {name}"),
    }
}

/// The texts of a bundled corpus, one per book.
fn books(name: &str) -> Vec<String> {
    match name {
        "all_four" => ["aesops_fables", "leaves_of_grass", "origin_of_species", "sherlock_holmes"].iter().map(|n| corpus(n).to_string()).collect(),
        // Fetched by scripts/fetch_gutenberg.sh into the gitignored data/gutenberg/.
        "novels6" => ["frankenstein", "pride_and_prejudice", "tale_of_two_cities", "dracula", "great_expectations", "moby_dick"]
            .iter()
            .map(|n| std::fs::read_to_string(format!("data/gutenberg/{n}.txt")).unwrap_or_else(|e| panic!("data/gutenberg/{n}.txt: {e} (run scripts/fetch_gutenberg.sh)")))
            .collect(),
        _ => vec![corpus(name).to_string()],
    }
}

/// (train, held-out) bytes of a bundled corpus: the last 10% of the text
/// is held out. `all_four` and `novels6` take the last 10% of *each* book,
/// so the held-out set mixes all the books.
#[allow(dead_code)]
pub fn split_corpus(name: &str) -> (Vec<usize>, Vec<usize>) {
    let (mut train, mut held_out) = (vec![], vec![]);
    for book in books(name) {
        let full = encode_bytes(&book);
        let split = (full.len() as f32 * 0.9) as usize;
        train.extend_from_slice(&full[..split]);
        held_out.extend_from_slice(&full[split..]);
    }
    (train, held_out)
}

/// Where each book's held-out text starts inside `split_corpus`'s held-out vector.
#[allow(dead_code)]
pub fn held_out_docs(name: &str) -> Vec<usize> {
    let (mut starts, mut at) = (vec![], 0);
    for book in books(name) {
        let full = encode_bytes(&book);
        starts.push(at);
        at += full.len() - (full.len() as f32 * 0.9) as usize;
    }
    starts
}
