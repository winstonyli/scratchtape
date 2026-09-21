use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::{Adam, AdamState, Sgd};
use scratchtape::tape::{Tape, Var};
use scratchtape::tensor::NdArray;
use std::collections::HashMap;
use std::thread;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{ForwardOut, apply_grad, decode_bytes, encode_bytes, precision_recall_f1, sample_window, shuffle};

/// Stratified k-fold linear-probe comparison (frozen embedding-only vs
/// graph-augmented with attention-graph neighbor mean), extracted from the
/// original is-vowel probe ([178af66]) so the same held-out rigor applies
/// to other externally-checkable categories without re-deriving the fold
/// logic each time. Returns (mean baseline accuracy, mean graph-augmented
/// accuracy, majority-class baseline accuracy) for the caller to compare
/// across categories.
fn run_kfold_probe(
    category: &str,
    labels: &[usize],
    node_features: &NdArray,
    neighbor1: &[usize],
    neighbor2: &[usize],
    d_model: usize,
    k_folds: usize,
    fold_seed: u64,
    verbose: bool,
) -> (f32, f32, f32) {
    let n_nodes = labels.len();
    let positive_count = labels.iter().filter(|&&l| l == 1).count();
    let majority_baseline = (n_nodes - positive_count).max(positive_count) as f32 / n_nodes as f32;
    if verbose {
        println!("\n{category} probe: {positive_count} positive / {n_nodes} nodes (majority-class baseline = {majority_baseline:.3})");
    }

    // Stratified: positive and negative indices shuffled and round-robin'd
    // into folds separately, so every fold gets a proportional share of
    // whichever class is rarer instead of risking an all-one-class test
    // fold under plain random splitting.
    let mut fold_rng = Rng::new(fold_seed);
    let mut pos_idxs: Vec<usize> = (0..n_nodes).filter(|&i| labels[i] == 1).collect();
    let mut neg_idxs: Vec<usize> = (0..n_nodes).filter(|&i| labels[i] == 0).collect();
    shuffle(&mut pos_idxs, &mut fold_rng);
    shuffle(&mut neg_idxs, &mut fold_rng);
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k_folds];
    for (i, &idx) in pos_idxs.iter().chain(neg_idxs.iter()).enumerate() {
        folds[i % k_folds].push(idx);
    }

    let hidden = 16;
    let adam = Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let train_steps = 400;

    let held_out_accuracy = |test_idx: &[usize], train_idx: &[usize], use_graph: bool, seed: u64| -> f32 {
        let mut clf_rng = Rng::new(seed);
        let in_dim = if use_graph { 2 * d_model } else { d_model };
        let mut l1 = Linear::new(&mut clf_rng, in_dim, hidden);
        let mut l2 = Linear::new(&mut clf_rng, hidden, 2);
        let mut s1w = AdamState::zeros_like(&l1.w);
        let mut s1b = AdamState::zeros_like(&l1.b);
        let mut s2w = AdamState::zeros_like(&l2.w);
        let mut s2b = AdamState::zeros_like(&l2.b);
        let train_labels: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();

        let mut acc = 0.0;
        for step in 0..train_steps {
            let mut tape = Tape::new();
            let x0 = tape.leaf(node_features.clone());
            let features = if use_graph {
                let n1 = tape.gather(x0, neighbor1);
                let n2 = tape.gather(x0, neighbor2);
                let sum_n = tape.add(n1, n2);
                let avg_neighbor = tape.scale(sum_n, 0.5);
                tape.concat(&[x0, avg_neighbor])
            } else {
                x0
            };
            let h1 = l1.forward(&mut tape, features);
            let h1r = tape.relu(h1.y);
            let h2 = l2.forward(&mut tape, h1r);
            let train_logits = tape.gather(h2.y, train_idx);
            let loss = tape.cross_entropy(train_logits, &train_labels);
            tape.backward(loss);
            l1.apply_grad_with(&tape, &h1, &adam, &mut s1w, &mut s1b);
            l2.apply_grad_with(&tape, &h2, &adam, &mut s2w, &mut s2b);
            if step == train_steps - 1 {
                let logits = tape.value(h2.y);
                let correct = test_idx
                    .iter()
                    .filter(|&&i| {
                        let pred = if logits.data[i * 2 + 1] > logits.data[i * 2] { 1 } else { 0 };
                        pred == labels[i]
                    })
                    .count();
                acc = correct as f32 / test_idx.len() as f32;
            }
        }
        acc
    };

    let mut baseline_accs = Vec::new();
    let mut gnn_accs = Vec::new();
    for fold in 0..k_folds {
        let test_idx = &folds[fold];
        let train_idx: Vec<usize> = (0..n_nodes).filter(|i| !test_idx.contains(i)).collect();
        let baseline_acc = held_out_accuracy(test_idx, &train_idx, false, 1000 + fold as u64);
        let gnn_acc = held_out_accuracy(test_idx, &train_idx, true, 2000 + fold as u64);
        if verbose {
            println!("  fold {fold}: test_size={}, baseline={baseline_acc:.3}, gnn={gnn_acc:.3}", test_idx.len());
        }
        baseline_accs.push(baseline_acc);
        gnn_accs.push(gnn_acc);
    }

    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (mean_baseline, mean_gnn) = (mean(&baseline_accs), mean(&gnn_accs));
    if verbose {
        println!("{k_folds}-fold held-out mean accuracy: baseline (embedding only) = {mean_baseline:.3}, graph-augmented = {mean_gnn:.3}");
    }
    (mean_baseline, mean_gnn, majority_baseline)
}

/// The first genuinely symbolic KR&R artifact in this project. Every prior
/// extraction (k-means clusters, the attention-derived graph, the
/// embedding-nearest-neighbor graph) is a numeric summary a human still
/// has to interpret; a decision tree over embedding dimensions produces
/// literal if-then rules instead - the dimensions themselves stay opaque
/// (a learned embedding space was never going to have named axes), but the
/// rule structure itself (which threshold, which branch) is fully
/// explicit, not a similarity score or a cluster id.
enum Tree {
    Leaf(usize),
    Split { dim: usize, threshold: f32, left: Box<Tree>, right: Box<Tree> },
}

fn gini(labels: &[usize], idxs: &[usize]) -> f32 {
    if idxs.is_empty() {
        return 0.0;
    }
    let n = idxs.len() as f32;
    let p = idxs.iter().filter(|&&i| labels[i] == 1).count() as f32 / n;
    1.0 - p * p - (1.0 - p) * (1.0 - p)
}

fn majority_class(labels: &[usize], idxs: &[usize]) -> usize {
    let pos = idxs.iter().filter(|&&i| labels[i] == 1).count();
    if pos * 2 >= idxs.len() { 1 } else { 0 }
}

/// Greedy, axis-aligned, depth-limited: at each node, try every (dimension,
/// midpoint-between-consecutive-sorted-values) split and keep whichever
/// minimizes weighted Gini impurity. `min_samples` stops splitting a node
/// with too few examples to trust a further split (same reasoning behind
/// this project's care with small-n classifiers elsewhere - the GNN
/// probe's stratified folds, its ceiling-effect finding on 34 nodes).
fn fit_tree(features: &[Vec<f32>], labels: &[usize], idxs: &[usize], depth: usize, min_samples: usize) -> Tree {
    let all_same = idxs.iter().all(|&i| labels[i] == labels[idxs[0]]);
    if depth == 0 || idxs.len() < min_samples || all_same {
        return Tree::Leaf(majority_class(labels, idxs));
    }

    let d_model = features[0].len();
    let mut best: Option<(usize, f32, f32)> = None; // (dim, threshold, weighted gini)
    for dim in 0..d_model {
        let mut values: Vec<f32> = idxs.iter().map(|&i| features[i][dim]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values.dedup();
        for w in values.windows(2) {
            let threshold = (w[0] + w[1]) / 2.0;
            let left: Vec<usize> = idxs.iter().copied().filter(|&i| features[i][dim] <= threshold).collect();
            let right: Vec<usize> = idxs.iter().copied().filter(|&i| features[i][dim] > threshold).collect();
            if left.is_empty() || right.is_empty() {
                continue;
            }
            let w_gini =
                (left.len() as f32 * gini(labels, &left) + right.len() as f32 * gini(labels, &right)) / idxs.len() as f32;
            if best.is_none_or(|(_, _, best_gini)| w_gini < best_gini) {
                best = Some((dim, threshold, w_gini));
            }
        }
    }

    match best {
        None => Tree::Leaf(majority_class(labels, idxs)),
        Some((dim, threshold, _)) => {
            let left: Vec<usize> = idxs.iter().copied().filter(|&i| features[i][dim] <= threshold).collect();
            let right: Vec<usize> = idxs.iter().copied().filter(|&i| features[i][dim] > threshold).collect();
            Tree::Split {
                dim,
                threshold,
                left: Box::new(fit_tree(features, labels, &left, depth - 1, min_samples)),
                right: Box::new(fit_tree(features, labels, &right, depth - 1, min_samples)),
            }
        }
    }
}

fn predict_tree(tree: &Tree, features: &[f32]) -> usize {
    match tree {
        Tree::Leaf(class) => *class,
        Tree::Split { dim, threshold, left, right } => {
            if features[*dim] <= *threshold { predict_tree(left, features) } else { predict_tree(right, features) }
        }
    }
}

fn print_tree(tree: &Tree, indent: usize, positive_name: &str) {
    let pad = "  ".repeat(indent);
    match tree {
        Tree::Leaf(class) => {
            println!("{pad}predict: {}", if *class == 1 { positive_name } else { "not" });
        }
        Tree::Split { dim, threshold, left, right } => {
            println!("{pad}if dim[{dim}] <= {threshold:.4}:");
            print_tree(left, indent + 1, positive_name);
            println!("{pad}else:");
            print_tree(right, indent + 1, positive_name);
        }
    }
}

/// Typed knowledge-graph edge list: for a given relation predicate,
/// `adj[i]` lists every `j` related to byte-index `i`. Reused across the
/// three relation types (cluster, attention, embedding-NN) and their
/// union - the same predicates already used for the cross-extraction
/// consistency check and rule-chaining, now supporting real multi-hop
/// traversal instead of only single-hop pairwise checks.
fn build_adjacency<F: Fn(usize, usize) -> bool>(n: usize, related: F) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i != j && related(i, j) {
                adj[i].push(j);
            }
        }
    }
    adj
}

/// Breadth-first reachability up to `max_hops`, including `start` itself.
fn bfs_reachable(start: usize, max_hops: usize, adj: &[Vec<usize>]) -> std::collections::HashSet<usize> {
    let mut visited = std::collections::HashSet::new();
    visited.insert(start);
    let mut frontier = vec![start];
    for _ in 0..max_hops {
        let mut next_frontier = Vec::new();
        for &node in &frontier {
            for &neighbor in &adj[node] {
                if visited.insert(neighbor) {
                    next_frontier.push(neighbor);
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }
    visited
}

/// Plain Lloyd's k-means, post-hoc analysis only - duplicated from
/// tiny_lm.rs rather than shared, same reasoning as every other duplicated
/// helper in this project. Reruns that file's parked symbolic/KR&R first
/// step on this corpus: the 172-byte corpus there produced clusters that
/// split on raw byte frequency, not any structural property, with not-enough-
/// signal named as the suspected cause. This corpus is ~100x bigger, so it's
/// the direct test of that suspicion, not a new technique.
fn kmeans(points: &[Vec<f32>], k: usize, iterations: usize, rng: &mut Rng) -> Vec<usize> {
    let n = points.len();
    let dim = points[0].len();

    let mut centroid_idxs: Vec<usize> = Vec::new();
    while centroid_idxs.len() < k {
        let idx = (rng.next_f32() * n as f32) as usize;
        if !centroid_idxs.contains(&idx) {
            centroid_idxs.push(idx);
        }
    }
    let mut centroids: Vec<Vec<f32>> = centroid_idxs.iter().map(|&i| points[i].clone()).collect();
    let mut assignments = vec![0usize; n];

    for _ in 0..iterations {
        for (i, p) in points.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f32::INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist: f32 = p.iter().zip(centroid.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            assignments[i] = best;
        }
        for c in 0..k {
            let members: Vec<&Vec<f32>> =
                points.iter().zip(assignments.iter()).filter(|&(_, &a)| a == c).map(|(p, _)| p).collect();
            if !members.is_empty() {
                let mut mean = vec![0.0f32; dim];
                for m in &members {
                    for d in 0..dim {
                        mean[d] += m[d];
                    }
                }
                for v in mean.iter_mut() {
                    *v /= members.len() as f32;
                }
                centroids[c] = mean;
            }
        }
    }
    assignments
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
    forward_full(tape, token_emb, pos_emb, blocks, final_ln, output_proj, input_ids, false, false)
}

/// Same as forward(), plus softmax1/QK-norm switches threaded straight
/// through to TransformerBlock::forward_full - added to directly re-test
/// the frequency-sink theory below (softmax1 was tried against it once,
/// diverged regardless of lr, reverted; QK-norm was later proven elsewhere
/// ([e9d5721]) to eliminate that exact divergence in a standalone block,
/// but never tried on this real corpus-scale model). forward() itself stays
/// a thin plain-softmax/no-QK-norm wrapper so every existing call site
/// (eval_loss, generate, train_with_diagnostics, and both flag-less callers
/// below) keeps working and keeps producing bit-identical, already-recorded
/// results - this is opt-in, not a default change.
fn forward_full(
    tape: &mut Tape,
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    input_ids: &[usize],
    use_softmax1: bool,
    use_qknorm: bool,
) -> (Var, ForwardOut) {
    let positions: Vec<usize> = (0..input_ids.len()).collect();
    let tok_out = token_emb.forward(tape, input_ids);
    let pos_out = pos_emb.forward(tape, &positions);
    let mut x = tape.add(tok_out.y, pos_out.y);

    let mut block_outs = Vec::with_capacity(blocks.len());
    for block in blocks {
        let out = block.forward_full(tape, x, 1, use_softmax1, use_qknorm);
        x = out.y;
        block_outs.push(out);
    }

    let ln_out = final_ln.forward(tape, x);
    let proj_out = output_proj.forward(tape, ln_out.y);
    let logits = proj_out.y;
    (logits, ForwardOut { tok_out, pos_out, block_outs, ln_out, proj_out })
}

/// Mean cross-entropy over `n_windows` random windows, no gradient update -
/// pure evaluation. Called on the held-out region to measure generalization
/// (loss on text the model never trained on), and separately on the train
/// region for a directly comparable in-sample number.
fn eval_loss(rng: &mut Rng, token_emb: &Embedding, pos_emb: &Embedding, blocks: &[TransformerBlock], final_ln: &LayerNorm, output_proj: &Linear, corpus: &[usize], seq_len: usize, n_windows: usize) -> f32 {
    eval_loss_full(rng, token_emb, pos_emb, blocks, final_ln, output_proj, corpus, seq_len, n_windows, false, false)
}

/// Same as eval_loss, plus softmax1/QK-norm switches threaded through to
/// forward_full - a model trained under those flags must be EVALUATED
/// under them too, same reasoning as attention_graph_full. Existing callers
/// stay on eval_loss (plain softmax, no QK-norm), unaffected.
fn eval_loss_full(
    rng: &mut Rng,
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    corpus: &[usize],
    seq_len: usize,
    n_windows: usize,
    use_softmax1: bool,
    use_qknorm: bool,
) -> f32 {
    let mut total = 0.0;
    for _ in 0..n_windows {
        let (input, target) = sample_window(rng, corpus, seq_len);
        let mut tape = Tape::new();
        let (logits, _) =
            forward_full(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &input, use_softmax1, use_qknorm);
        let loss = tape.cross_entropy(logits, &target);
        total += tape.value(loss).data[0];
    }
    total / n_windows as f32
}

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

/// Same wiring as main()'s training loop, minus every diagnostic (no
/// grad_accum, no periodic eval, no generation). Returns the full model,
/// not just token_emb - the cross-seed attention-graph check needs a real
/// forward() pass (pos_emb, blocks, final_ln, output_proj), the same
/// requirement that turned train_with_diagnostics's return type into
/// TrainedModel earlier.
fn train_token_embedding(
    seed: u64,
    train: &[usize],
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    seq_len: usize,
    n_blocks: usize,
    vocab_size: usize,
    steps: usize,
) -> TrainedModel {
    train_token_embedding_full(seed, train, d_model, n_heads, d_ff, seq_len, n_blocks, vocab_size, steps, false, false)
}

/// Same as train_token_embedding, plus softmax1/QK-norm switches threaded
/// through to forward_full - lets the frequency-sink re-test below train a
/// real model under the condition being tested, not just analyze one
/// trained under the default. Existing callers stay on train_token_embedding
/// (plain softmax, no QK-norm), unaffected.
fn train_token_embedding_full(
    seed: u64,
    train: &[usize],
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    seq_len: usize,
    n_blocks: usize,
    vocab_size: usize,
    steps: usize,
    use_softmax1: bool,
    use_qknorm: bool,
) -> TrainedModel {
    let mut rng = Rng::new(seed);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> =
        (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    for step in 0..=steps {
        let (input, target) = sample_window(&mut rng, train, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) =
            forward_full(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input, use_softmax1, use_qknorm);
        let loss = tape.cross_entropy(logits, &target);
        let loss_val = tape.value(loss).data[0];
        // Only reachable with use_softmax1/use_qknorm - plain softmax (every
        // existing caller) never diverges here, so this is a strict addition
        // for an under-tested combination at a step count ([e9d5721]'s own
        // proof only ran 2000), not a behavior change for anything recorded
        // so far. Breaks early with a clear signal rather than burning the
        // rest of the run on NaN-corrupted weights.
        if loss_val.is_nan() {
            println!("  seed {seed} (softmax1={use_softmax1}, qknorm={use_qknorm}) diverged to NaN at step {step}");
            break;
        }
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
    TrainedModel { token_emb, pos_emb, blocks, final_ln, output_proj }
}

/// Connected components of an adjacency matrix via BFS - used twice below
/// (majority-agreement graph and strict-unanimity graph), the two-consumer
/// bar this project applies before factoring out a helper.
fn connected_components(adj: &[Vec<bool>], n: usize) -> Vec<usize> {
    let mut component = vec![usize::MAX; n];
    let mut next_component = 0;
    for start in 0..n {
        if component[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        component[start] = next_component;
        while let Some(i) = stack.pop() {
            for j in 0..n {
                if adj[i][j] && component[j] == usize::MAX {
                    component[j] = next_component;
                    stack.push(j);
                }
            }
        }
        next_component += 1;
    }
    component
}

/// Bundles the full model rather than returning 5 separate values - needed
/// so the primary run's attention-graph extraction below (which needs
/// pos_emb, blocks, final_ln, output_proj, not just token_emb) can run a
/// real forward() pass after training, unlike the k-means step which only
/// ever needed token_emb's rows.
struct TrainedModel {
    token_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    final_ln: LayerNorm,
    output_proj: Linear,
}

/// Third independent KR&R extraction, after k-means clustering and the
/// attention-derived relational graph: raw embedding-space nearest
/// neighbors by Euclidean distance. No training dynamics involved at all -
/// not attention weights (behavioral), not a k-means partition (a flat
/// grouping) - pure representational geometry, computed once on the final
/// trained table with no sampling/coverage gaps (every filtered byte has a
/// well-defined embedding vector, unlike the attention graph's 5 sampling-
/// gap bytes). Returns, per filtered_bytes index, an ascending-by-distance
/// list of (neighbor index, distance) pairs, self excluded, matching the
/// attention graph's top-k format for direct comparison.
fn embedding_neighbor_graph(model: &TrainedModel, filtered_bytes: &[usize], d_model: usize, k: usize) -> Vec<Vec<(usize, f32)>> {
    let n = filtered_bytes.len();
    let rows: Vec<Vec<f32>> = filtered_bytes
        .iter()
        .map(|&b| model.token_emb.table.data[b * d_model..b * d_model + d_model].to_vec())
        .collect();
    (0..n)
        .map(|i| {
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f32 = rows[i].iter().zip(rows[j].iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(k);
            dists
        })
        .collect()
}

/// Averages block-0/head-0 attention weight from query byte to key byte
/// over `attn_windows` random samples from `train`, restricted to
/// `filtered_bytes` (same noise-exclusion reasoning as the clustering
/// step). Returns, per filtered_bytes index, a descending-by-weight list
/// of (target index, average weight) pairs - empty if that byte never
/// appeared in a usable sampled window. `window_seed` is shared across
/// calls with different models so every seed sees the exact same sampled
/// windows - isolates the model-training seed as the only varying input,
/// the same control kmeans_rng=999 gives the clustering cross-seed check.
fn attention_graph(
    model: &TrainedModel,
    train: &[usize],
    filtered_bytes: &[usize],
    seq_len: usize,
    attn_windows: usize,
    window_seed: u64,
) -> (Vec<Vec<(usize, f32)>>, usize) {
    attention_graph_full(model, train, filtered_bytes, seq_len, attn_windows, window_seed, false, false)
}

/// Same as attention_graph, plus softmax1/QK-norm switches threaded through
/// to forward_full - a model trained under those flags must also be
/// ANALYZED under them, or the extracted attention weights would reflect
/// plain-softmax behavior applied to softmax1/QK-norm-trained parameters,
/// not what the model actually does. Existing callers stay on
/// attention_graph (plain softmax, no QK-norm), unaffected.
fn attention_graph_full(
    model: &TrainedModel,
    train: &[usize],
    filtered_bytes: &[usize],
    seq_len: usize,
    attn_windows: usize,
    window_seed: u64,
    use_softmax1: bool,
    use_qknorm: bool,
) -> (Vec<Vec<(usize, f32)>>, usize) {
    let byte_index: HashMap<usize, usize> = filtered_bytes.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let n_bytes = filtered_bytes.len();
    let mut weight_sum = vec![0.0f32; n_bytes * n_bytes];
    let mut weight_count = vec![0u32; n_bytes * n_bytes];

    let mut attn_rng = Rng::new(window_seed);
    let mut windows_used = 0;
    for _ in 0..attn_windows {
        let (window, _) = sample_window(&mut attn_rng, train, seq_len);
        if window.iter().any(|b| !byte_index.contains_key(b)) {
            continue;
        }
        windows_used += 1;
        let mut tape = Tape::new();
        let (_, out) = forward_full(
            &mut tape,
            &model.token_emb,
            &model.pos_emb,
            &model.blocks,
            &model.final_ln,
            &model.output_proj,
            &window,
            use_softmax1,
            use_qknorm,
        );
        let weights = tape.value(out.block_outs[0].head_weights[0]);
        for qi in 0..seq_len {
            let qi_idx = byte_index[&window[qi]];
            for ki in 0..seq_len {
                let ki_idx = byte_index[&window[ki]];
                weight_sum[qi_idx * n_bytes + ki_idx] += weights.data[qi * seq_len + ki];
                weight_count[qi_idx * n_bytes + ki_idx] += 1;
            }
        }
    }

    let graph = (0..n_bytes)
        .map(|i| {
            let mut targets: Vec<(usize, f32)> = (0..n_bytes)
                .filter_map(|j| {
                    let c = weight_count[i * n_bytes + j];
                    if c == 0 {
                        None
                    } else {
                        Some((j, weight_sum[i * n_bytes + j] / c as f32))
                    }
                })
                .collect();
            targets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            targets
        })
        .collect();
    (graph, windows_used)
}

/// Same computation as attention_graph, generalized across every (block,
/// head) pair in one pass over the sampled windows - a single forward()
/// call already produces every block's every head's attention weights
/// internally, so comparing all of them costs the same set of forward
/// passes as comparing just one did, not n_blocks*n_heads times more.
/// Returns [block][head][byte index] -> sorted (target index, avg weight).
/// Used only on the primary model (unlike attention_graph, not needed per-
/// seed) to test the named block-0/head-0-only limitation directly:
/// multihead_attention_recall.rs already found one head blind and another
/// sighted on the identical query - this checks whether that same kind of
/// per-head specialization shows up here too.
fn attention_graph_all_heads(
    model: &TrainedModel,
    train: &[usize],
    filtered_bytes: &[usize],
    seq_len: usize,
    attn_windows: usize,
    window_seed: u64,
    n_blocks: usize,
    n_heads: usize,
) -> Vec<Vec<Vec<Vec<(usize, f32)>>>> {
    let byte_index: HashMap<usize, usize> = filtered_bytes.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let n_bytes = filtered_bytes.len();
    let mut weight_sum = vec![vec![vec![0.0f32; n_bytes * n_bytes]; n_heads]; n_blocks];
    let mut weight_count = vec![vec![vec![0u32; n_bytes * n_bytes]; n_heads]; n_blocks];

    let mut attn_rng = Rng::new(window_seed);
    for _ in 0..attn_windows {
        let (window, _) = sample_window(&mut attn_rng, train, seq_len);
        if window.iter().any(|b| !byte_index.contains_key(b)) {
            continue;
        }
        let mut tape = Tape::new();
        let (_, out) =
            forward(&mut tape, &model.token_emb, &model.pos_emb, &model.blocks, &model.final_ln, &model.output_proj, &window);
        for block in 0..n_blocks {
            for head in 0..n_heads {
                let weights = tape.value(out.block_outs[block].head_weights[head]);
                for qi in 0..seq_len {
                    let qi_idx = byte_index[&window[qi]];
                    for ki in 0..seq_len {
                        let ki_idx = byte_index[&window[ki]];
                        weight_sum[block][head][qi_idx * n_bytes + ki_idx] += weights.data[qi * seq_len + ki];
                        weight_count[block][head][qi_idx * n_bytes + ki_idx] += 1;
                    }
                }
            }
        }
    }

    (0..n_blocks)
        .map(|block| {
            (0..n_heads)
                .map(|head| {
                    (0..n_bytes)
                        .map(|i| {
                            let mut targets: Vec<(usize, f32)> = (0..n_bytes)
                                .filter_map(|j| {
                                    let c = weight_count[block][head][i * n_bytes + j];
                                    if c == 0 {
                                        None
                                    } else {
                                        Some((j, weight_sum[block][head][i * n_bytes + j] / c as f32))
                                    }
                                })
                                .collect();
                            targets.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                            targets
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Same training loop as train_token_embedding, plus every diagnostic the
/// primary/reference run needs (grad_accum, periodic eval, generation) -
/// duplicated rather than parameterized with a bunch of Option<_> diagnostic
/// flags, same reasoning as every other duplicated helper in this file.
/// Log lines are collected rather than printed directly so output stays
/// deterministic and un-interleaved when this runs concurrently with the
/// other seeds under thread::scope - each thread has its own stdout calls
/// otherwise racing for output order with no benefit.
fn train_with_diagnostics(
    seed: u64,
    train: &[usize],
    held_out: &[usize],
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    seq_len: usize,
    n_blocks: usize,
    vocab_size: usize,
    steps: usize,
) -> (TrainedModel, Vec<f32>, Vec<String>) {
    let mut rng = Rng::new(seed);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> =
        (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    let mut grad_accum = vec![0.0f32; vocab_size];
    let mut log = Vec::new();
    let start_time = Instant::now();
    for step in 0..=steps {
        let (input, target) = sample_window(&mut rng, train, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);

        let tok_grad = tape.grad(out.tok_out.table).unwrap();
        for b in 0..vocab_size {
            let row = &tok_grad.data[b * d_model..b * d_model + d_model];
            grad_accum[b] += row.iter().map(|g| g.abs()).sum::<f32>();
        }

        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);

        if step % 1600 == 0 {
            let train_eval = eval_loss(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, train, seq_len, 20);
            let held_out_eval = eval_loss(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, held_out, seq_len, 20);
            log.push(format!(
                "step {step:>5}: train_loss = {:.4}, eval train = {train_eval:.4}, eval held-out = {held_out_eval:.4} ({:.1}s elapsed)",
                tape.value(loss).data[0], start_time.elapsed().as_secs_f32()
            ));
        }
    }
    log.push(format!("total training time: {:.1}s for {steps} steps", start_time.elapsed().as_secs_f32()));

    let seed_text = encode_bytes("The Fox");
    let generated = generate(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &seed_text, 200, seq_len, 0.8);
    log.push(format!("\ngenerated (seed \"The Fox\", temperature=0.8):\n{}", decode_bytes(&generated)));

    (TrainedModel { token_emb, pos_emb, blocks, final_ln, output_proj }, grad_accum, log)
}

fn main() {
    // Real, diverse public-domain prose (not a repeated stress-test corpus
    // like tiny_lm_scaled.rs) - Project Gutenberg ebook #21, "Three Hundred
    // Aesop's Fables" (George Fyler Townsend translation), Gutenberg
    // boilerplate header/footer and the trailing alphabetical index
    // stripped, fetched verbatim rather than reproduced from memory (same
    // accuracy-risk reasoning as every other corpus choice in this project).
    let full_text = include_str!("../../data/aesops_fables.txt");
    let full_encoded = encode_bytes(full_text);

    // 90/10 train/held-out split - held-out text is never sampled during
    // training, only used to measure whether the model generalizes to
    // unseen text or just memorizes, the actual point of using a bigger,
    // non-repeated corpus instead of just scaling the model again.
    let split = (full_encoded.len() as f32 * 0.9) as usize;
    let (train, held_out) = full_encoded.split_at(split);
    println!("corpus: {} bytes total, {} train / {} held-out", full_encoded.len(), train.len(), held_out.len());

    // Same model scale as tiny_lm_scaled.rs - isolates the corpus-size axis
    // on its own, rather than compounding it with a model-size change too.
    // Model-size scale-up tried twice and reverted both times:
    // - 2x model, SAME 16000 steps ([3ed92e9]): worse held-out loss (2.10
    //   vs 1.93) and more fragmentation (35 strict-consensus components vs
    //   32) - a compute-optimal-scaling confound, not evidence capacity
    //   doesn't help.
    // - 2x model, 4x steps (64000, compute-matched-ish) - confirms the
    //   confound exactly: held-out loss 1.9417 (essentially identical to
    //   1.9335) and 32 strict-consensus components (exactly matching).
    //   But capacity doesn't clearly HELP either once fairly trained - it
    //   just catches back up to parity, at ~20x the wall-clock cost
    //   (20410s vs ~820-1020s) for no net gain. No free lunch from scaling
    //   at this corpus size.
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    println!("model: d_model={d_model} n_heads={n_heads} d_ff={d_ff} seq_len={seq_len} n_blocks={n_blocks}");

    // 4x tiny_lm_corpus's original step count - direct test of the
    // undertraining theory from the cross-seed stability check: train loss
    // was still falling at step 4000, so most byte embeddings likely hadn't
    // converged to a seed-independent geometry yet. If stability scores
    // rise broadly here, that confirms it; if they plateau, the flat
    // low-stability result at 4000 steps was a real method limit, not just
    // "not done training."
    let steps = 16000;

    // The 3 seed runs (1 primary + diagnostics, 2 lean) are fully
    // independent - no shared mutable state, each owns its own model and
    // rng. thread::scope runs them concurrently instead of sequentially:
    // same total compute, wall-clock bounded by the slowest one instead of
    // the sum of all three, on a machine with cores to spare (16 here).
    // Not GPU, not minibatching - both already measured net-negative at
    // this model's scale ([479e3b9], [29dad53]) - this is a different axis
    // (parallelizing independent runs, not speeding up one run).
    println!("\ntraining 3 seeds concurrently for cross-seed cluster stability...");
    let wall_clock_start = Instant::now();
    let (primary, seed2_model, seed3_model) = thread::scope(|scope| {
        let primary_handle = scope.spawn(|| {
            train_with_diagnostics(1, train, held_out, d_model, n_heads, d_ff, seq_len, n_blocks, vocab_size, steps)
        });
        let seed2_handle =
            scope.spawn(|| train_token_embedding(2, train, d_model, n_heads, d_ff, seq_len, n_blocks, vocab_size, steps));
        let seed3_handle =
            scope.spawn(|| train_token_embedding(3, train, d_model, n_heads, d_ff, seq_len, n_blocks, vocab_size, steps));
        (primary_handle.join().unwrap(), seed2_handle.join().unwrap(), seed3_handle.join().unwrap())
    });
    println!("all 3 seeds trained in {:.1}s wall-clock", wall_clock_start.elapsed().as_secs_f32());

    let (model, grad_accum, primary_log) = primary;
    for line in &primary_log {
        println!("{line}");
    }

    // Restricted to bytes actually seen in `train`: the same reasoning as
    // tiny_lm.rs - any byte that only appears in held_out never received a
    // gradient, so its embedding row is untrained noise, not signal.
    let mut distinct_bytes: Vec<usize> = train.to_vec();
    distinct_bytes.sort_unstable();
    distinct_bytes.dedup();

    // Diagnostic dump, sorted ascending by accumulated |gradient| - printed
    // before any cutoff decision so the actual distribution (not an assumed
    // shape) drives where the noise/signal line gets drawn.
    let mut by_accum = distinct_bytes.clone();
    by_accum.sort_by(|&a, &b| grad_accum[a].partial_cmp(&grad_accum[b]).unwrap());
    println!("\nper-byte accumulated |gradient| over training, ascending:");
    for &b in &by_accum {
        println!("  {:?}: {:.4}", (b as u8) as char, grad_accum[b]);
    }

    // Bottom-20th-percentile cutoff: the lowest-accumulated-gradient fifth
    // of trained bytes is treated as noise and excluded before clustering,
    // rather than clustered alongside real signal. 20% is a starting point,
    // not derived from this distribution - the printed dump above is what
    // lets that choice be checked and adjusted, not just asserted.
    let cutoff_count = distinct_bytes.len() / 5;
    let noise_floor = grad_accum[by_accum[cutoff_count]];
    let filtered_bytes: Vec<usize> = distinct_bytes.iter().copied().filter(|&b| grad_accum[b] >= noise_floor).collect();
    let excluded: Vec<String> =
        distinct_bytes.iter().filter(|&&b| grad_accum[b] < noise_floor).map(|&b| format!("{:?}", (b as u8) as char)).collect();
    println!(
        "\nexcluding {} of {} bytes as noise (accumulated |gradient| < {noise_floor:.4}): {}",
        excluded.len(), distinct_bytes.len(), excluded.join(" ")
    );

    let embedding_rows: Vec<Vec<f32>> =
        filtered_bytes.iter().map(|&b| model.token_emb.table.data[b * d_model..b * d_model + d_model].to_vec()).collect();

    // Fixed k-means-init seed, independent of the training rng - isolates
    // model-training randomness as the only thing varying across the
    // cross-seed runs below. Reusing the training rng here would conflate
    // two different sources of randomness (model init/sampling vs centroid
    // init) into one number, making the seeds incomparable.
    let k = 10;
    let mut kmeans_rng = Rng::new(999);
    let assignments = kmeans(&embedding_rows, k, 50, &mut kmeans_rng);

    println!(
        "\nk-means clusters (k={k}) over the {} bytes remaining after noise filtering:",
        filtered_bytes.len()
    );
    for c in 0..k {
        let members: Vec<String> = filtered_bytes
            .iter()
            .zip(assignments.iter())
            .filter(|&(_, &a)| a == c)
            .map(|(&b, _)| format!("{:?}", (b as u8) as char))
            .collect();
        println!("  cluster {c}: {}", members.join(" "));
    }

    // Cross-seed stability: same corpus, same filtered byte set, same
    // k-means init seed - only the model's own init+sampling seed varies.
    // Cluster *labels* aren't comparable across independently-trained runs
    // (cluster 3 in seed 2's run has no relation to cluster 3 in seed 1's),
    // so agreement is measured pairwise instead: do bytes i and j land in
    // the same cluster together, regardless of which numbered cluster that
    // is. That sidesteps the label-permutation problem entirely.
    let mut all_assignments: Vec<Vec<usize>> = vec![assignments.clone()];
    for seed_model in [&seed2_model, &seed3_model] {
        let rows: Vec<Vec<f32>> = filtered_bytes
            .iter()
            .map(|&b| seed_model.token_emb.table.data[b * d_model..b * d_model + d_model].to_vec())
            .collect();
        let mut seed_kmeans_rng = Rng::new(999);
        let a = kmeans(&rows, k, 50, &mut seed_kmeans_rng);
        all_assignments.push(a);
    }

    let n = filtered_bytes.len();
    let n_runs = all_assignments.len();
    // unanimous[i][j]: true only if every run agrees on whether i and j
    // share a cluster (all-same or all-different) - the only two possible
    // outcomes are "unanimous" or "2-1 split" since n_runs = 3.
    let same_cluster = |r: usize, i: usize, j: usize| all_assignments[r][i] == all_assignments[r][j];
    let mut unanimous = vec![vec![false; n]; n];
    let mut majority_same = vec![vec![false; n]; n];
    let mut unanimous_same = vec![vec![false; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let agree_count = (0..n_runs).filter(|&r| same_cluster(r, i, j)).count();
            unanimous[i][j] = agree_count == 0 || agree_count == n_runs;
            majority_same[i][j] = agree_count * 2 > n_runs;
            unanimous_same[i][j] = agree_count == n_runs;
        }
    }

    // Per-byte stability score: fraction of relationships to every other
    // byte that stayed unanimous across all 3 seeds. A byte whose cluster
    // membership is real structure should agree on most of its
    // relationships regardless of training seed; one placed arbitrarily by
    // whatever the embedding happened to look like at that seed should not.
    let mut stability: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let agreeing = (0..n).filter(|&j| j != i && unanimous[i][j]).count();
            (i, agreeing as f32 / (n - 1) as f32)
        })
        .collect();
    stability.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("\nper-byte cross-seed stability (fraction of pairwise relationships unanimous across all 3 seeds), ascending:");
    for &(idx, score) in &stability {
        println!("  {:?}: {:.3}", (filtered_bytes[idx] as u8) as char, score);
    }

    // Consensus clusters: connected components of the majority-agreement
    // graph (edge i-j if >=2 of 3 seeds put them in the same cluster).
    // This is the actual answer to "what structure survives across seeds" -
    // the per-byte scores above show confidence, this shows the resulting
    // groups.
    let component = connected_components(&majority_same, n);
    let next_component = component.iter().copied().max().map_or(0, |m| m + 1);
    println!("\nconsensus clusters (connected components of >=2/3-seed agreement):");
    for c in 0..next_component {
        let members: Vec<String> = (0..n)
            .filter(|&i| component[i] == c)
            .map(|i| format!("{:?}", (filtered_bytes[i] as u8) as char))
            .collect();
        if !members.is_empty() {
            println!("  component {c}: {}", members.join(" "));
        }
    }

    // Same connected-components analysis, but only wiring an edge when all
    // 3 seeds agree (not just a 2/3 majority) - the direct test of whether
    // the giant >=2/3-agreement component above is real shared structure or
    // a transitivity artifact: a single 2/3-agreeing edge is enough to
    // bridge two otherwise-unrelated sub-groups into one component, but a
    // 3/3 requirement can't be bridged by one seed's idiosyncrasy the same
    // way. No retraining needed - reuses this run's all_assignments as-is.
    let strict_component = connected_components(&unanimous_same, n);
    let strict_next = strict_component.iter().copied().max().map_or(0, |m| m + 1);
    println!("\nstrict consensus clusters (connected components of 3/3-seed agreement):");
    for c in 0..strict_next {
        let members: Vec<String> = (0..n)
            .filter(|&i| strict_component[i] == c)
            .map(|i| format!("{:?}", (filtered_bytes[i] as u8) as char))
            .collect();
        if !members.is_empty() {
            println!("  component {c}: {}", members.join(" "));
        }
    }

    // Second, independent KR&R extraction from tiny_lm.rs, rerun here on
    // the scaled corpus: averaged attention weights (block 0, head 0 only -
    // same named limitation as tiny_lm.rs, still not re-checked) discretized
    // into a top-2-targets-per-byte relational graph. Restricted to
    // filtered_bytes, the same noise-excluded set the clustering above
    // uses. Exhaustive coverage (every window position) was tractable on
    // tiny_lm.rs's 172-byte corpus (~156 windows); this corpus's train
    // region has ~214,000 possible window positions, so exhaustive coverage
    // would mean ~214,000 forward passes - random sampling instead, same
    // proportionate-scope reasoning as eval_loss's 20-window sample.
    let attn_windows = 20000;
    let (primary_graph, windows_used) = attention_graph(&model, train, &filtered_bytes, seq_len, attn_windows, 777);

    println!(
        "\nattention-derived relational graph (block 0, head 0), top-2 targets per byte, {windows_used} sampled windows:"
    );
    for (i, &b) in filtered_bytes.iter().enumerate() {
        let top: Vec<String> =
            primary_graph[i].iter().take(2).map(|&(j, w)| format!("{:?}({:.2})", (filtered_bytes[j] as u8) as char, w)).collect();
        println!("  {:?} -> {}", (b as u8) as char, top.join(", "));
    }

    // Cross-seed stability on the attention graph - same rigor already
    // applied to clustering, never applied to this extraction. Unlike
    // k-means cluster labels, a target here IS a byte identity (not an
    // arbitrary cluster number), so there's no label-permutation problem to
    // sidestep - top-1 targets are directly comparable across seeds.
    // Same window_seed (777) for every seed model: isolates the model's
    // own training seed as the only varying input, the same control
    // kmeans_rng=999 gives the clustering check.
    let (seed2_graph, _) = attention_graph(&seed2_model, train, &filtered_bytes, seq_len, attn_windows, 777);
    let (seed3_graph, _) = attention_graph(&seed3_model, train, &filtered_bytes, seq_len, attn_windows, 777);

    println!("\nattention-graph cross-seed top-1 target stability (byte -> seed1 | seed2 | seed3 top-1 target):");
    let mut unanimous_count = 0;
    let mut majority_count = 0;
    let mut none_count = 0;
    let mut gap_count = 0;
    for (i, &b) in filtered_bytes.iter().enumerate() {
        let top1 = |g: &[Vec<(usize, f32)>]| g[i].first().map(|&(j, _)| filtered_bytes[j]);
        let (t1, t2, t3) = (top1(&primary_graph), top1(&seed2_graph), top1(&seed3_graph));
        let label = match (t1, t2, t3) {
            (Some(a), Some(bb), Some(c)) if a == bb && bb == c => {
                unanimous_count += 1;
                "unanimous"
            }
            (Some(a), Some(bb), Some(c)) if a == bb || a == c || bb == c => {
                majority_count += 1;
                "majority"
            }
            (Some(_), Some(_), Some(_)) => {
                none_count += 1;
                "none"
            }
            _ => {
                gap_count += 1;
                "gap"
            }
        };
        let fmt = |t: Option<usize>| t.map_or("-".to_string(), |x| format!("{:?}", (x as u8) as char));
        println!("  {:?} -> {} | {} | {} ({label})", (b as u8) as char, fmt(t1), fmt(t2), fmt(t3));
    }
    println!(
        "summary: {unanimous_count} unanimous, {majority_count} majority (2/3), {none_count} all-different, {gap_count} sampling-gap, out of {} bytes",
        filtered_bytes.len()
    );

    // Third independent extraction: raw embedding-space nearest neighbors,
    // no attention behavior or clustering partition involved at all - see
    // embedding_neighbor_graph's doc comment. Same cross-seed rigor as the
    // attention graph, but no label-permutation problem (neighbor identity
    // is directly comparable) and no sampling-gap case (every byte has a
    // well-defined embedding vector, unlike attention needing window
    // coverage) - simpler three-way split than the attention graph's four.
    let nn_k = 2;
    let primary_nn = embedding_neighbor_graph(&model, &filtered_bytes, d_model, nn_k);
    println!("\nembedding-space nearest neighbors (k={nn_k}, Euclidean distance):");
    for (i, &b) in filtered_bytes.iter().enumerate() {
        let top: Vec<String> =
            primary_nn[i].iter().map(|&(j, d)| format!("{:?}({d:.2})", (filtered_bytes[j] as u8) as char)).collect();
        println!("  {:?} -> {}", (b as u8) as char, top.join(", "));
    }

    let seed2_nn = embedding_neighbor_graph(&seed2_model, &filtered_bytes, d_model, nn_k);
    let seed3_nn = embedding_neighbor_graph(&seed3_model, &filtered_bytes, d_model, nn_k);
    println!("\nembedding-neighbor cross-seed top-1 nearest-neighbor stability (byte -> seed1 | seed2 | seed3):");
    let (mut nn_unanimous, mut nn_majority, mut nn_none) = (0, 0, 0);
    for (i, &b) in filtered_bytes.iter().enumerate() {
        let top1 = |g: &[Vec<(usize, f32)>]| filtered_bytes[g[i][0].0];
        let (t1, t2, t3) = (top1(&primary_nn), top1(&seed2_nn), top1(&seed3_nn));
        let label = if t1 == t2 && t2 == t3 {
            nn_unanimous += 1;
            "unanimous"
        } else if t1 == t2 || t1 == t3 || t2 == t3 {
            nn_majority += 1;
            "majority"
        } else {
            nn_none += 1;
            "none"
        };
        let fmt = |x: usize| format!("{:?}", (x as u8) as char);
        println!("  {:?} -> {} | {} | {} ({label})", (b as u8) as char, fmt(t1), fmt(t2), fmt(t3));
    }
    println!(
        "summary: {nn_unanimous} unanimous, {nn_majority} majority (2/3), {nn_none} all-different, out of {} bytes",
        filtered_bytes.len()
    );

    // Group-membership stability: exact top-1 identity wasn't cross-seed
    // stable for digits (no digit appeared in the unanimous list above),
    // despite digits forming the tightest cluster in any one seed. Tests
    // the weaker, more plausible claim directly: does a category member's
    // nearest neighbor stay IN THE SAME CATEGORY across seeds, even when
    // the exact identity doesn't (e.g. a digit's nearest neighbor stays
    // "some digit" every time, even if which specific digit varies)?
    // Reuses the 3 already-computed nn graphs, no new computation beyond
    // the category check itself. Predicates duplicated from the linear-
    // probe section below rather than reordered/shared, same reasoning as
    // every other duplicated one-liner in this file.
    let categories_gm: [(&str, &dyn Fn(usize) -> bool); 4] = [
        ("is-vowel", &|b: usize| matches!((b as u8) as char, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U')),
        ("is-uppercase", &|b: usize| ((b as u8) as char).is_ascii_uppercase()),
        ("is-digit", &|b: usize| ((b as u8) as char).is_ascii_digit()),
        ("is-punctuation", &|b: usize| ((b as u8) as char).is_ascii_punctuation()),
    ];
    println!("\ngroup-membership cross-seed stability (does top-1 neighbor stay in-category across seeds, even if exact identity doesn't):");
    for (name, predicate) in categories_gm {
        let members: Vec<usize> = (0..filtered_bytes.len()).filter(|&i| predicate(filtered_bytes[i])).collect();
        if members.is_empty() {
            continue;
        }
        let (mut unanimous_in, mut unanimous_out, mut split) = (0, 0, 0);
        for &i in &members {
            let in_group = |g: &[Vec<(usize, f32)>]| predicate(filtered_bytes[g[i][0].0]);
            let agree = [in_group(&primary_nn), in_group(&seed2_nn), in_group(&seed3_nn)].iter().filter(|&&x| x).count();
            match agree {
                3 => unanimous_in += 1,
                0 => unanimous_out += 1,
                _ => split += 1,
            }
        }
        println!(
            "  {name}: {unanimous_in} always stay in-category, {unanimous_out} always leave, {split} split, out of {} members",
            members.len()
        );
    }

    // First formal-reasoning step in this whole line: every extraction so
    // far (k-means clustering, the attention graph, the embedding-NN
    // graph) has only ever been checked against ITSELF (cross-seed
    // stability). Never checked against each OTHER - three independent
    // views of the same trained model that might agree, contradict, or
    // simply not overlap. Pure post-hoc analysis over data already
    // computed for the primary model (assignments, primary_graph,
    // primary_nn) - no retraining needed for this step specifically, it
    // only needs the pipeline to reach this point once per run.
    let cluster_same = |i: usize, j: usize| assignments[i] == assignments[j];
    let attn_related = |i: usize, j: usize| {
        primary_graph[i].iter().take(2).any(|&(k, _)| k == j) || primary_graph[j].iter().take(2).any(|&(k, _)| k == i)
    };
    let nn_related = |i: usize, j: usize| {
        primary_nn[i].iter().any(|&(k, _)| k == j) || primary_nn[j].iter().any(|&(k, _)| k == i)
    };

    let consistency_n = filtered_bytes.len();
    let mut total_pairs = 0;
    let mut agree_all3 = 0;
    let (mut cluster_vs_attn_agree, mut cluster_vs_nn_agree, mut attn_vs_nn_agree) = (0, 0, 0);
    let mut attn_nn_agree_cluster_disagrees: Vec<(usize, usize)> = Vec::new();
    let mut cluster_agrees_attn_nn_disagree: Vec<(usize, usize)> = Vec::new();
    for i in 0..consistency_n {
        for j in (i + 1)..consistency_n {
            total_pairs += 1;
            let (c, a, e) = (cluster_same(i, j), attn_related(i, j), nn_related(i, j));
            if c == a && a == e {
                agree_all3 += 1;
            }
            if c == a {
                cluster_vs_attn_agree += 1;
            }
            if c == e {
                cluster_vs_nn_agree += 1;
            }
            if a == e {
                attn_vs_nn_agree += 1;
            }
            if a && e && !c {
                attn_nn_agree_cluster_disagrees.push((i, j));
            }
            if c && !a && !e {
                cluster_agrees_attn_nn_disagree.push((i, j));
            }
        }
    }
    println!(
        "\ncross-extraction formal consistency check (k-means / attention-graph top-2 / embedding-NN top-2), {total_pairs} byte pairs:"
    );
    println!("  cluster<->attention pairwise agreement: {:.3}", cluster_vs_attn_agree as f32 / total_pairs as f32);
    println!("  cluster<->embedding-NN pairwise agreement: {:.3}", cluster_vs_nn_agree as f32 / total_pairs as f32);
    println!("  attention<->embedding-NN pairwise agreement: {:.3}", attn_vs_nn_agree as f32 / total_pairs as f32);
    println!("  all 3 methods agree: {:.3} ({agree_all3} of {total_pairs} pairs)", agree_all3 as f32 / total_pairs as f32);

    println!(
        "\n  attention + embedding-NN both relate a pair, but clustering split them apart ({} pairs, first 15):",
        attn_nn_agree_cluster_disagrees.len()
    );
    for &(i, j) in attn_nn_agree_cluster_disagrees.iter().take(15) {
        println!("    {:?} <-> {:?}", (filtered_bytes[i] as u8) as char, (filtered_bytes[j] as u8) as char);
    }

    println!(
        "\n  clustering grouped a pair together, but neither attention nor embedding-NN corroborate ({} pairs, first 15):",
        cluster_agrees_attn_nn_disagree.len()
    );
    for &(i, j) in cluster_agrees_attn_nn_disagree.iter().take(15) {
        println!("    {:?} <-> {:?}", (filtered_bytes[i] as u8) as char, (filtered_bytes[j] as u8) as char);
    }

    // Second formal-reasoning step: rule-chaining inference instead of
    // just comparing extractions against each other. Combines the three
    // relation types above into one graph (related(i,j) = same-cluster OR
    // attention-related OR embedding-NN-related), then infers each byte's
    // category by majority vote over its RELATED bytes' TRUE labels - a
    // leave-one-out relational nearest-neighbor rule, not a trained
    // classifier. No gradient descent, no embeddings read directly - only
    // the extracted graph structure plus other bytes' known labels,
    // chained through one inference rule. Tests whether that's enough to
    // recover category membership, against the same categories the
    // neural probe and decision tree were already measured on.
    let is_vowel_rc = |b: usize| matches!((b as u8) as char, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U');
    let is_uppercase_rc = |b: usize| ((b as u8) as char).is_ascii_uppercase();
    let is_digit_rc = |b: usize| ((b as u8) as char).is_ascii_digit();
    let is_punctuation_rc = |b: usize| ((b as u8) as char).is_ascii_punctuation();
    // Compound intersections, added for the hierarchy/lattice check below:
    // does "uppercase AND vowel" (A,E,I,O,U) behave as a real,
    // distinguishable sub-category nested within both is-uppercase and
    // is-vowel, or does the representation only support flat categories?
    let is_uppercase_vowel_rc = |b: usize| matches!((b as u8) as char, 'A' | 'E' | 'I' | 'O' | 'U');
    let is_lowercase_vowel_rc = |b: usize| matches!((b as u8) as char, 'a' | 'e' | 'i' | 'o' | 'u');
    let rc_categories: [(&str, &dyn Fn(usize) -> bool); 6] = [
        ("is-vowel", &is_vowel_rc),
        ("is-uppercase", &is_uppercase_rc),
        ("is-digit", &is_digit_rc),
        ("is-punctuation", &is_punctuation_rc),
        ("is-uppercase-vowel", &is_uppercase_vowel_rc),
        ("is-lowercase-vowel", &is_lowercase_vowel_rc),
    ];
    let related = |i: usize, j: usize| cluster_same(i, j) || attn_related(i, j) || nn_related(i, j);

    println!("\nrule-chaining inference (majority vote of a byte's related-graph neighbors' TRUE labels, leave-one-out):");
    for (name, predicate) in rc_categories {
        let labels: Vec<usize> = (0..consistency_n).map(|i| if predicate(filtered_bytes[i]) { 1 } else { 0 }).collect();
        let mut predictions = vec![0usize; consistency_n];
        let mut isolated = 0;
        for i in 0..consistency_n {
            let (mut pos, mut neg) = (0, 0);
            for j in 0..consistency_n {
                if i == j || !related(i, j) {
                    continue;
                }
                if labels[j] == 1 {
                    pos += 1;
                } else {
                    neg += 1;
                }
            }
            predictions[i] = if pos == 0 && neg == 0 {
                isolated += 1;
                0
            } else if pos > neg {
                1
            } else {
                0
            };
        }
        let correct = (0..consistency_n).filter(|&i| predictions[i] == labels[i]).count();
        let (p, r, f1) = precision_recall_f1(&labels, &predictions);
        println!(
            "  {name}: accuracy={:.3} ({correct}/{consistency_n}), precision={p:.3}, recall={r:.3}, f1={f1:.3}, {isolated} isolated bytes",
            correct as f32 / consistency_n as f32
        );
    }

    // Fourth extraction, building on the second (attention graph): GNN
    // message-passing over the attention graph ([178af66]), is-vowel
    // prediction as an externally-checkable
    // probe (not model-derived). Originally inconclusive by a stated data
    // limit, not by finding: only 34 labeled nodes / 7 positive examples
    // meant both baseline and graph-augmented classifiers hit a 100%
    // training-accuracy ceiling regardless of graph structure, and a
    // held-out split was named as "unreliable with only 7 positive examples
    // total." filtered_bytes has 92 nodes here - enough to actually run a
    // held-out comparison instead of training-accuracy-only. Reuses
    // primary_graph's already-computed top-2 targets rather than
    // recomputing the attention graph a second time.
    // Falls back to self (i) when a byte has no attention data at all (the
    // 5 sampling-gap bytes from the cross-seed check above, empty
    // target list) - a safe identity/no-op edge rather than an out-of-
    // bounds index, consistent with neighbor2's existing fallback to
    // neighbor1 when there's no 2nd target.
    let n_nodes = filtered_bytes.len();
    let neighbor1: Vec<usize> = (0..n_nodes).map(|i| primary_graph[i].first().map(|&(j, _)| j).unwrap_or(i)).collect();
    let neighbor2: Vec<usize> =
        (0..n_nodes).map(|i| primary_graph[i].get(1).map(|&(j, _)| j).unwrap_or(neighbor1[i])).collect();

    let node_features = model.token_emb.table.gather_rows(&filtered_bytes);
    let k_folds = 5;

    // Extended beyond the original is-vowel probe to 3 more externally-
    // checkable categories, same k-fold rigor, reusing run_kfold_probe -
    // does the embedding geometry (and graph augmentation) generalize past
    // vowel-ness specifically, or was that one category special?
    let is_vowel = |b: usize| matches!((b as u8) as char, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U');
    let is_uppercase = |b: usize| ((b as u8) as char).is_ascii_uppercase();
    let is_digit = |b: usize| ((b as u8) as char).is_ascii_digit();
    let is_punctuation = |b: usize| ((b as u8) as char).is_ascii_punctuation();
    // Hierarchy/lattice check: does "uppercase AND vowel" (A,E,I,O,U) form
    // a genuine, distinguishable sub-category nested within both
    // is-uppercase and is-vowel, or does this representation only support
    // flat categories? Same k-fold rigor (neural probe + decision tree)
    // as every flat category above, applied to the intersection.
    let is_uppercase_vowel = |b: usize| matches!((b as u8) as char, 'A' | 'E' | 'I' | 'O' | 'U');
    let is_lowercase_vowel = |b: usize| matches!((b as u8) as char, 'a' | 'e' | 'i' | 'o' | 'u');

    let categories: [(&str, &dyn Fn(usize) -> bool, u64); 6] = [
        ("is-vowel", &is_vowel, 42),
        ("is-uppercase", &is_uppercase, 43),
        ("is-digit", &is_digit, 44),
        ("is-punctuation", &is_punctuation, 45),
        ("is-uppercase-vowel", &is_uppercase_vowel, 46),
        ("is-lowercase-vowel", &is_lowercase_vowel, 47),
    ];

    let mut summary = Vec::new();
    for (name, predicate, fold_seed) in categories {
        let labels: Vec<usize> = filtered_bytes.iter().map(|&b| if predicate(b) { 1 } else { 0 }).collect();
        let (base, gnn, majority) =
            run_kfold_probe(name, &labels, &node_features, &neighbor1, &neighbor2, d_model, k_folds, fold_seed, true);
        summary.push((name, majority, base, gnn));
    }
    println!("\nlinear-probe summary (category: majority-baseline | embedding-only | graph-augmented):");
    for (name, majority, base, gnn) in &summary {
        println!("  {name}: {majority:.3} | {base:.3} | {gnn:.3}");
    }

    // Rule extraction: same held-out k-fold rigor, same embedding-only
    // feature space as the linear probe's baseline condition, but a
    // decision tree instead of a small neural classifier - trades whatever
    // accuracy the neural net's nonlinearity buys for literal, inspectable
    // rules. depth=3 and min_samples=6 are conservative on purpose: with
    // ~73 training rows per fold, an unconstrained tree would just
    // memorize (the exact ceiling-effect risk the GNN probe's own history
    // already flagged at this data scale).
    let feature_rows: Vec<Vec<f32>> = (0..n_nodes).map(|i| node_features.data[i * d_model..(i + 1) * d_model].to_vec()).collect();
    let tree_depth = 3;
    let tree_min_samples = 6;

    println!("\ndecision-tree summary (category: majority-baseline | embedding-only-neural | decision-tree, held-out):");
    for (name, predicate, fold_seed) in categories {
        let labels: Vec<usize> = filtered_bytes.iter().map(|&b| if predicate(b) { 1 } else { 0 }).collect();
        let mut fold_rng = Rng::new(fold_seed);
        let mut pos_idxs: Vec<usize> = (0..n_nodes).filter(|&i| labels[i] == 1).collect();
        let mut neg_idxs: Vec<usize> = (0..n_nodes).filter(|&i| labels[i] == 0).collect();
        shuffle(&mut pos_idxs, &mut fold_rng);
        shuffle(&mut neg_idxs, &mut fold_rng);
        let mut folds: Vec<Vec<usize>> = vec![Vec::new(); k_folds];
        for (i, &idx) in pos_idxs.iter().chain(neg_idxs.iter()).enumerate() {
            folds[i % k_folds].push(idx);
        }

        let mut tree_accs = Vec::new();
        for fold in 0..k_folds {
            let test_idx = &folds[fold];
            let train_idx: Vec<usize> = (0..n_nodes).filter(|i| !test_idx.contains(i)).collect();
            let tree = fit_tree(&feature_rows, &labels, &train_idx, tree_depth, tree_min_samples);
            let correct = test_idx.iter().filter(|&&i| predict_tree(&tree, &feature_rows[i]) == labels[i]).count();
            tree_accs.push(correct as f32 / test_idx.len() as f32);
        }
        let mean_tree_acc = tree_accs.iter().sum::<f32>() / tree_accs.len() as f32;
        let (_, majority, neural_base, _) = summary.iter().find(|&&(n, ..)| n == name).unwrap();
        println!("  {name}: {majority:.3} | {neural_base:.3} | {mean_tree_acc:.3}");
    }

    // The actual deliverable: an explicit rule, fit on every node (not a
    // held-out split - this is the "what did it learn" artifact, the
    // k-fold numbers above are the separate "does it generalize" claim).
    let vowel_labels: Vec<usize> = filtered_bytes.iter().map(|&b| if is_vowel(b) { 1 } else { 0 }).collect();
    let all_idx: Vec<usize> = (0..n_nodes).collect();
    let full_tree = fit_tree(&feature_rows, &vowel_labels, &all_idx, tree_depth, tree_min_samples);
    println!("\nis-vowel decision tree, fit on all {n_nodes} nodes (dims are opaque learned axes, not named features):");
    print_tree(&full_tree, 1, "vowel");

    // Punctuation just emerged as the strongest cross-seed signal in the
    // whole embedding-geometry line (55% group-membership stability, two
    // clean unanimous sentence-role clusters {,;:} vs {.?!}) - printing its
    // rule directly tests the strongest finding against the newest method.
    let punct_labels: Vec<usize> = filtered_bytes.iter().map(|&b| if is_punctuation(b) { 1 } else { 0 }).collect();
    let punct_tree = fit_tree(&feature_rows, &punct_labels, &all_idx, tree_depth, tree_min_samples);
    println!("\nis-punctuation decision tree, fit on all {n_nodes} nodes:");
    print_tree(&punct_tree, 1, "punctuation");

    // Direct test of the named block-0/head-0-only limitation this
    // extraction has carried since tiny_lm.rs: does every head show the
    // same self-attention/arbitrary-sink pattern, or does per-head
    // specialization (multihead_attention_recall.rs's "one head blind, one
    // sighted" finding) show up here too? One extra pass over the same
    // sampled windows gets every block's every head at once - the forward
    // pass already computes them all internally.
    let all_heads = attention_graph_all_heads(&model, train, &filtered_bytes, seq_len, attn_windows, 777, n_blocks, n_heads);
    println!("\nper-head self-attention rate and top sink target (block 0-{}, head 0-{}):", n_blocks - 1, n_heads - 1);
    for block in 0..n_blocks {
        for head in 0..n_heads {
            let graph = &all_heads[block][head];
            let mut self_count = 0;
            let mut covered = 0;
            let mut sink_counts: HashMap<usize, usize> = HashMap::new();
            for (i, targets) in graph.iter().enumerate() {
                if let Some(&(top, _)) = targets.first() {
                    covered += 1;
                    if top == i {
                        self_count += 1;
                    } else {
                        *sink_counts.entry(top).or_insert(0) += 1;
                    }
                }
            }
            let self_rate = self_count as f32 / covered as f32;
            let top_sink = sink_counts.iter().max_by_key(|&(_, &c)| c);
            let sink_desc = top_sink
                .map(|(&j, &c)| format!("{:?}({c})", (filtered_bytes[j] as u8) as char))
                .unwrap_or_else(|| "-".to_string());
            println!("  block {block} head {head}: self-attention rate = {self_rate:.2}, top sink target = {sink_desc}");
        }
    }

    // Direct follow-up to the linear-probe summary: block 0/head 0's graph
    // hurt is-uppercase (-5.6pp vs embedding-only) while helping every
    // other category. Given how sharply heads specialize (just shown
    // above), some other head's graph might do the opposite. No
    // retraining needed - all_heads already has every head's targets;
    // this just re-derives neighbor1/neighbor2 per head and reruns the
    // is-uppercase probe's graph-augmented condition against each.
    let uppercase_labels: Vec<usize> =
        filtered_bytes.iter().map(|&b| if ((b as u8) as char).is_ascii_uppercase() { 1 } else { 0 }).collect();
    // Reuses the embedding-only baseline already computed in the summary
    // loop above (same labels, same fold_seed=43) rather than recomputing
    // it - the baseline doesn't depend on which head's graph is used at
    // all, so there's nothing new to learn from rerunning it.
    let uppercase_baseline = summary.iter().find(|&&(name, ..)| name == "is-uppercase").unwrap().2;
    println!(
        "\nis-uppercase graph-augmented accuracy per (block, head) - embedding-only baseline = {uppercase_baseline:.3}:"
    );
    for block in 0..n_blocks {
        for head in 0..n_heads {
            let graph = &all_heads[block][head];
            let head_neighbor1: Vec<usize> = (0..n_nodes).map(|i| graph[i].first().map(|&(j, _)| j).unwrap_or(i)).collect();
            let head_neighbor2: Vec<usize> =
                (0..n_nodes).map(|i| graph[i].get(1).map(|&(j, _)| j).unwrap_or(head_neighbor1[i])).collect();
            let (_, gnn_acc, _) = run_kfold_probe(
                "is-uppercase",
                &uppercase_labels,
                &node_features,
                &head_neighbor1,
                &head_neighbor2,
                d_model,
                k_folds,
                43,
                false,
            );
            let delta = gnn_acc - uppercase_baseline;
            println!("  block {block} head {head}: graph-augmented = {gnn_acc:.3} ({delta:+.3})");
        }
    }

    // Direct follow-up to the hierarchy/lattice check: block 0/head 0's
    // graph erased the neural probe's entire +3.2pp gain for is-uppercase-
    // vowel (0.978 -> 0.946, back to majority baseline) - the same head
    // already shown to be the single worst pick for plain is-uppercase.
    // Does the intersection category get restored by the SAME heads that
    // helped plain is-uppercase (block 3 head 5, block 3 head 0), or does
    // the doubly-constrained category need a different head entirely?
    let uppercase_vowel_labels: Vec<usize> =
        filtered_bytes.iter().map(|&b| matches!((b as u8) as char, 'A' | 'E' | 'I' | 'O' | 'U')).map(|m| m as usize).collect();
    let uppercase_vowel_baseline = summary.iter().find(|&&(name, ..)| name == "is-uppercase-vowel").unwrap().2;
    println!(
        "\nis-uppercase-vowel graph-augmented accuracy per (block, head) - embedding-only baseline = {uppercase_vowel_baseline:.3}:"
    );
    for block in 0..n_blocks {
        for head in 0..n_heads {
            let graph = &all_heads[block][head];
            let head_neighbor1: Vec<usize> = (0..n_nodes).map(|i| graph[i].first().map(|&(j, _)| j).unwrap_or(i)).collect();
            let head_neighbor2: Vec<usize> =
                (0..n_nodes).map(|i| graph[i].get(1).map(|&(j, _)| j).unwrap_or(head_neighbor1[i])).collect();
            let (_, gnn_acc, _) = run_kfold_probe(
                "is-uppercase-vowel",
                &uppercase_vowel_labels,
                &node_features,
                &head_neighbor1,
                &head_neighbor2,
                d_model,
                k_folds,
                46,
                false,
            );
            let delta = gnn_acc - uppercase_vowel_baseline;
            println!("  block {block} head {head}: graph-augmented = {gnn_acc:.3} ({delta:+.3})");
        }
    }

    // Exhaustive formal verification for the decision tree: rule-chaining
    // above already evaluates exactly (leave-one-out over all 92 nodes,
    // not a fold estimate), but the tree's numbers so far were only
    // 5-fold held-out accuracy (~18 test rows per fold). Leave-one-out
    // (92 refits) is cheap at this data scale and gives an exact, non-
    // estimated number instead. Precision/recall/F1 alongside accuracy -
    // these categories are mostly under 15% positive, so accuracy alone
    // can't distinguish "learned something" from "always predicts not."
    println!("\nexhaustive leave-one-out verification (decision tree, exact - 92 refits, not a fold estimate):");
    for (name, predicate, _fold_seed) in categories {
        let labels: Vec<usize> = filtered_bytes.iter().map(|&b| if predicate(b) { 1 } else { 0 }).collect();
        let mut predictions = vec![0usize; n_nodes];
        for i in 0..n_nodes {
            let train_idx: Vec<usize> = (0..n_nodes).filter(|&j| j != i).collect();
            let tree = fit_tree(&feature_rows, &labels, &train_idx, tree_depth, tree_min_samples);
            predictions[i] = predict_tree(&tree, &feature_rows[i]);
        }
        let correct = (0..n_nodes).filter(|&i| predictions[i] == labels[i]).count();
        let (p, r, f1) = precision_recall_f1(&labels, &predictions);
        println!(
            "  {name}: accuracy={:.3} ({correct}/{n_nodes}), precision={p:.3}, recall={r:.3}, f1={f1:.3}",
            correct as f32 / n_nodes as f32
        );
    }

    // First KR&R query capability: real multi-hop traversal instead of
    // only single-hop pairwise checks (everything above only ever asked
    // "are i and j related", never "what's reachable from i in N steps").
    // Formalizes the three relation types (already used for the
    // consistency check and rule-chaining) into typed adjacency lists.
    // Demonstrated on bytes already shown to have distinctive relational
    // behavior this session: '\n' and 'N' (both hit near-1.0 cross-seed
    // attention stability), 'a' (the recurring block-0 frequency-sink
    // target), and 'e' (highest raw training-gradient signal of any byte).
    let cluster_adj = build_adjacency(n_nodes, cluster_same);
    let attn_adj = build_adjacency(n_nodes, attn_related);
    let nn_adj = build_adjacency(n_nodes, nn_related);
    let combined_adj = build_adjacency(n_nodes, related);

    println!("\nknowledge-graph multi-hop query (bytes reachable within N hops, self excluded):");
    for &target in &[b'a', b'\n', b'N', b'e'] {
        let Some(idx) = filtered_bytes.iter().position(|&b| b as u8 == target) else {
            continue;
        };
        println!("  from {:?}:", target as char);
        for (rel_name, adj) in [
            ("cluster", &cluster_adj),
            ("attention", &attn_adj),
            ("embedding-NN", &nn_adj),
            ("combined (any)", &combined_adj),
        ] {
            let one_hop = bfs_reachable(idx, 1, adj).len() - 1;
            let two_hop = bfs_reachable(idx, 2, adj).len() - 1;
            println!("    {rel_name}: 1-hop reaches {one_hop}, 2-hop reaches {two_hop} (of {} total)", n_nodes - 1);
        }
    }

    // Revisits the frequency-sink theory ([f30c608]) against softmax1+QK-norm,
    // now that QK-norm is proven ([e9d5721]) to eliminate the exact runaway
    // the original softmax1 attempt ([d54c8f2]) hit on this same
    // TransformerBlock architecture. The first pass at this (single seed 4)
    // found a diversity increase but left two things unchecked - exactly the
    // gap this project's own methodology already flagged once before:
    // ccd411e found single-seed attention-graph findings (like the original
    // 'a'-sink itself) can be seed-arbitrary, not real structure, so a
    // single-seed comparison here proves nothing on its own; and nothing
    // measured whether the fix costs or helps actual language-modeling
    // quality, only that it doesn't diverge and reshapes attention. Three
    // fresh seeds (4,5,6 - unused by the seeds-1-3 clustering/attention-graph
    // cross-seed checks above, so this replication is fully independent of
    // them), each trained as a control/treatment pair from identical
    // seed+data+steps, varying only use_softmax1+use_qknorm together (the
    // fix requires both - softmax1 alone is the known-divergent condition).
    // All 6 runs concurrent via thread::scope, same reasoning as the 3-seed
    // run above.
    println!("\nrevisiting the frequency-sink theory across 3 fresh seeds: softmax1+QK-norm vs plain softmax, identical seed/data/steps per pair...");
    let qknorm_wall_clock_start = Instant::now();
    let qknorm_seeds = [4u64, 5, 6];
    let (control_models, treatment_models): (Vec<TrainedModel>, Vec<TrainedModel>) = thread::scope(|scope| {
        let handles: Vec<_> = qknorm_seeds
            .iter()
            .map(|&seed| {
                let control = scope.spawn(move || {
                    train_token_embedding_full(seed, train, d_model, n_heads, d_ff, seq_len, n_blocks, vocab_size, steps, false, false)
                });
                let treatment = scope.spawn(move || {
                    train_token_embedding_full(seed, train, d_model, n_heads, d_ff, seq_len, n_blocks, vocab_size, steps, true, true)
                });
                (control, treatment)
            })
            .collect();
        handles.into_iter().map(|(c, t)| (c.join().unwrap(), t.join().unwrap())).unzip()
    });
    println!("all {} control+treatment pairs trained in {:.1}s wall-clock", qknorm_seeds.len(), qknorm_wall_clock_start.elapsed().as_secs_f32());

    let sink_idx = filtered_bytes.iter().position(|&b| b == b'a' as usize);
    let count_top1_target = |graph: &[Vec<(usize, f32)>], target_idx: usize| {
        graph.iter().filter(|targets| targets.first().map(|&(j, _)| j) == Some(target_idx)).count()
    };
    let distinct_top1_targets = |graph: &[Vec<(usize, f32)>]| {
        let mut targets: Vec<usize> = graph.iter().filter_map(|t| t.first().map(|&(j, _)| j)).collect();
        targets.sort_unstable();
        targets.dedup();
        targets.len()
    };

    println!(
        "\nper-seed comparison (train loss | held-out loss | distinct top-1 targets of {} | bytes defaulting to 'a'):",
        filtered_bytes.len()
    );
    let mut diversity_deltas = Vec::with_capacity(qknorm_seeds.len());
    let mut held_out_deltas = Vec::with_capacity(qknorm_seeds.len());
    for (i, &seed) in qknorm_seeds.iter().enumerate() {
        let (control, treatment) = (&control_models[i], &treatment_models[i]);
        // Fresh eval rng per model/corpus combination, same convention
        // train_with_diagnostics's own periodic eval already uses -
        // n_windows=20 matches that call's own sample size.
        let control_train_loss =
            eval_loss_full(&mut Rng::new(seed), &control.token_emb, &control.pos_emb, &control.blocks, &control.final_ln, &control.output_proj, train, seq_len, 20, false, false);
        let control_held_out_loss =
            eval_loss_full(&mut Rng::new(seed), &control.token_emb, &control.pos_emb, &control.blocks, &control.final_ln, &control.output_proj, held_out, seq_len, 20, false, false);
        let treatment_train_loss =
            eval_loss_full(&mut Rng::new(seed), &treatment.token_emb, &treatment.pos_emb, &treatment.blocks, &treatment.final_ln, &treatment.output_proj, train, seq_len, 20, true, true);
        let treatment_held_out_loss =
            eval_loss_full(&mut Rng::new(seed), &treatment.token_emb, &treatment.pos_emb, &treatment.blocks, &treatment.final_ln, &treatment.output_proj, held_out, seq_len, 20, true, true);

        // Analyzed under the SAME flags each was trained with - plain
        // forward() on the treatment model would silently measure
        // plain-softmax behavior applied to softmax1/QK-norm-trained
        // parameters, not what the model actually does.
        let (control_graph, _) = attention_graph_full(control, train, &filtered_bytes, seq_len, attn_windows, 777, false, false);
        let (treatment_graph, _) = attention_graph_full(treatment, train, &filtered_bytes, seq_len, attn_windows, 777, true, true);
        let (control_diversity, treatment_diversity) = (distinct_top1_targets(&control_graph), distinct_top1_targets(&treatment_graph));
        let (control_sink, treatment_sink) =
            sink_idx.map_or((0, 0), |idx| (count_top1_target(&control_graph, idx), count_top1_target(&treatment_graph, idx)));

        println!(
            "  seed {seed} control:   {control_train_loss:.4} | {control_held_out_loss:.4} | {control_diversity} | {control_sink}"
        );
        println!(
            "  seed {seed} treatment: {treatment_train_loss:.4} | {treatment_held_out_loss:.4} | {treatment_diversity} | {treatment_sink}"
        );
        diversity_deltas.push(treatment_diversity as i32 - control_diversity as i32);
        held_out_deltas.push(treatment_held_out_loss - control_held_out_loss);
    }

    // The actual replication check: does the diversity increase found on
    // seed 4 alone hold up, or was it that one seed's own noise (the
    // ccd411e concern this rerun exists to address)? And separately, does
    // the fix cost anything in held-out loss, the question the first pass
    // never asked at all.
    let seeds_with_higher_diversity = diversity_deltas.iter().filter(|&&d| d > 0).count();
    let mean_diversity_delta = diversity_deltas.iter().sum::<i32>() as f32 / diversity_deltas.len() as f32;
    let mean_held_out_delta = held_out_deltas.iter().sum::<f32>() / held_out_deltas.len() as f32;
    println!(
        "\ncross-seed summary: {seeds_with_higher_diversity} of {} seeds show higher top-1-target diversity under softmax1+QK-norm (mean delta {mean_diversity_delta:+.1}); mean held-out loss delta (treatment - control) {mean_held_out_delta:+.4}",
        qknorm_seeds.len()
    );
}
