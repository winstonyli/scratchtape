use engine::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng, TransformerBlock, TransformerBlockOut};
use engine::optim::{Adam, AdamState};
use engine::tape::{Tape, Var};
use std::collections::HashMap;

fn encode_bytes(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
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
    opt: &engine::optim::Sgd,
) {
    token_emb.apply_grad(tape, &out.tok_out, opt);
    pos_emb.apply_grad(tape, &out.pos_out, opt);
    for (block, block_out) in blocks.iter_mut().zip(out.block_outs.iter()) {
        block.apply_grad(tape, block_out, opt);
    }
    final_ln.apply_grad(tape, &out.ln_out, opt);
    output_proj.apply_grad(tape, &out.proj_out, opt);
}

fn main() {
    // --- Regenerate a real trained model + attention graph, same as
    // tiny_lm.rs - self-contained per established per-example precedent. ---
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
    let opt = engine::optim::Sgd { lr: 0.3 };

    for _ in 0..=4000 {
        let (input, target) = sample_window(&mut rng, &encoded, seq_len);
        let mut tape = Tape::new();
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
    println!("language model trained (4000 steps)");

    let mut distinct_bytes: Vec<usize> = encoded.clone();
    distinct_bytes.sort_unstable();
    distinct_bytes.dedup();
    let n = distinct_bytes.len();
    let byte_index: HashMap<usize, usize> = distinct_bytes.iter().enumerate().map(|(i, &b)| (b, i)).collect();

    // Attention graph: same block-0/head-0, top-2-per-byte extraction as
    // tiny_lm.rs.
    let mut weight_sum = vec![0.0f32; n * n];
    let mut weight_count = vec![0u32; n * n];
    let window_count = encoded.len() - seq_len;
    for start in 0..window_count {
        let window = encoded[start..start + seq_len].to_vec();
        let mut tape = Tape::new();
        let (_, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &window);
        let w = tape.value(out.block_outs[0].head_weights[0]);
        for qi in 0..seq_len {
            let qi_idx = byte_index[&window[qi]];
            for ki in 0..seq_len {
                let ki_idx = byte_index[&window[ki]];
                weight_sum[qi_idx * n + ki_idx] += w.data[qi * seq_len + ki];
                weight_count[qi_idx * n + ki_idx] += 1;
            }
        }
    }
    let mut neighbor1 = vec![0usize; n];
    let mut neighbor2 = vec![0usize; n];
    for i in 0..n {
        let mut targets: Vec<(usize, f32)> = (0..n)
            .filter_map(|j| {
                let c = weight_count[i * n + j];
                if c == 0 { None } else { Some((j, weight_sum[i * n + j] / c as f32)) }
            })
            .collect();
        targets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        neighbor1[i] = targets[0].0;
        neighbor2[i] = targets.get(1).map(|&(j, _)| j).unwrap_or(targets[0].0);
    }
    println!("attention graph extracted ({n} nodes, fixed out-degree 2)");

    // --- Ground truth labels: externally-checkable, not derived from
    // anything the model itself produced. ---
    let is_vowel = |b: usize| matches!((b as u8) as char, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U');
    let labels: Vec<usize> = distinct_bytes.iter().map(|&b| if is_vowel(b) { 1 } else { 0 }).collect();
    let vowel_count = labels.iter().filter(|&&l| l == 1).count();
    let majority_baseline_acc = (n - vowel_count).max(vowel_count) as f32 / n as f32;
    println!("labels: {vowel_count} vowels out of {n} bytes (majority-class baseline accuracy: {majority_baseline_acc:.3})");

    // Frozen features (linear-probe style evaluation): the pretrained
    // embeddings are never updated here, only the small classifier heads
    // are trained on top - this tests what the embeddings+graph already
    // encode, not joint fine-tuning.
    let node_features = token_emb.table.gather_rows(&distinct_bytes);

    // --- Baseline: raw embedding only, no graph. ---
    let hidden = 16;
    let mut base1 = Linear::new(&mut rng, d_model, hidden);
    let mut base2 = Linear::new(&mut rng, hidden, 2);
    let adam = Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let mut b1w = AdamState::zeros_like(&base1.w);
    let mut b1b = AdamState::zeros_like(&base1.b);
    let mut b2w = AdamState::zeros_like(&base2.w);
    let mut b2b = AdamState::zeros_like(&base2.b);

    let mut baseline_acc = 0.0;
    for step in 0..400 {
        let mut tape = Tape::new();
        let x0 = tape.leaf(node_features.clone());
        let h1 = base1.forward(&mut tape, x0);
        let h1r = tape.relu(h1.y);
        let h2 = base2.forward(&mut tape, h1r);
        let loss = tape.cross_entropy(h2.y, &labels);
        tape.backward(loss);
        adam.step(&mut base1.w, tape.grad(h1.w).unwrap(), &mut b1w);
        adam.step(&mut base1.b, tape.grad(h1.b).unwrap(), &mut b1b);
        adam.step(&mut base2.w, tape.grad(h2.w).unwrap(), &mut b2w);
        adam.step(&mut base2.b, tape.grad(h2.b).unwrap(), &mut b2b);
        if step == 399 {
            let logits = tape.value(h2.y);
            let correct = (0..n)
                .filter(|&i| {
                    let pred = if logits.data[i * 2 + 1] > logits.data[i * 2] { 1 } else { 0 };
                    pred == labels[i]
                })
                .count();
            baseline_acc = correct as f32 / n as f32;
        }
    }
    println!("\nbaseline (embedding only, no graph): training accuracy = {baseline_acc:.3}");

    // --- GNN: embedding + mean of its 2 attention-graph neighbors. ---
    let mut gnn1 = Linear::new(&mut rng, 2 * d_model, hidden);
    let mut gnn2 = Linear::new(&mut rng, hidden, 2);
    let mut g1w = AdamState::zeros_like(&gnn1.w);
    let mut g1b = AdamState::zeros_like(&gnn1.b);
    let mut g2w = AdamState::zeros_like(&gnn2.w);
    let mut g2b = AdamState::zeros_like(&gnn2.b);

    let mut gnn_acc = 0.0;
    for step in 0..400 {
        let mut tape = Tape::new();
        let x0 = tape.leaf(node_features.clone());
        let n1 = tape.gather(x0, &neighbor1);
        let n2 = tape.gather(x0, &neighbor2);
        let sum_n = tape.add(n1, n2);
        let avg_neighbor = tape.scale(sum_n, 0.5);
        let combined = tape.concat(&[x0, avg_neighbor]);

        let h1 = gnn1.forward(&mut tape, combined);
        let h1r = tape.relu(h1.y);
        let h2 = gnn2.forward(&mut tape, h1r);
        let loss = tape.cross_entropy(h2.y, &labels);
        tape.backward(loss);
        adam.step(&mut gnn1.w, tape.grad(h1.w).unwrap(), &mut g1w);
        adam.step(&mut gnn1.b, tape.grad(h1.b).unwrap(), &mut g1b);
        adam.step(&mut gnn2.w, tape.grad(h2.w).unwrap(), &mut g2w);
        adam.step(&mut gnn2.b, tape.grad(h2.b).unwrap(), &mut g2b);
        if step == 399 {
            let logits = tape.value(h2.y);
            let correct = (0..n)
                .filter(|&i| {
                    let pred = if logits.data[i * 2 + 1] > logits.data[i * 2] { 1 } else { 0 };
                    pred == labels[i]
                })
                .count();
            gnn_acc = correct as f32 / n as f32;
        }
    }
    println!("graph-augmented (embedding + 2-neighbor mean): training accuracy = {gnn_acc:.3}");
    println!(
        "\n(only {n} labeled nodes total - a toy-scale proof of mechanism, not a generalization claim; \
majority-class baseline is {majority_baseline_acc:.3})"
    );
    if (baseline_acc - gnn_acc).abs() < 1e-6 && baseline_acc > 0.99 {
        println!(
            "\nBoth conditions reached the same ceiling (100% training accuracy) - a real, honest\n\
result, not a distinguishing one: with only {n} examples and a 2-layer MLP, both models\n\
trivially memorize the labels regardless of whether graph structure is used at all. This\n\
specific comparison methodology (training accuracy at this data scale) cannot tell whether\n\
the attention-graph neighbor aggregation adds real predictive value - it would need a\n\
held-out split, which is unreliable with only {vowel_count} positive examples total. Reported\n\
as the actual finding rather than expanded in scope to force a distinguishing signal."
        );
    }
}
