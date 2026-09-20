// Minimal differentiable forward-chaining rule (NSFR/Chainformer-style -
// scoped way down, not a full Neural Theorem Prover), built to re-test a
// specific finding from tiny_lm_corpus.rs's rule-chaining inference: a
// hand-written majority-vote rule over a combined relation graph beat a
// decision tree on every tested category, but precision/recall revealed
// several of those wins were hollow (zero recall - the rule never once
// caught a true positive). That rule combined 3 relation types (same-
// cluster, attention-related, embedding-NN-related) with an untrained,
// equal-weight OR before voting. This asks the direct follow-up: does
// LEARNING how much to trust each relation type - instead of OR-ing them
// unweighted - fix the hollow-recall failure?
//
// Self-contained: builds its own relation graph from raw corpus byte
// co-occurrence statistics rather than a trained transformer's attention/
// embeddings/clustering, so the comparison doesn't require dragging in
// tiny_lm_corpus.rs's ~1500 lines of extraction machinery. Two relation
// types stand in for the original's three "different notions of
// related": SHORT-range co-occurrence (within 3 byte positions - closer
// to orthographic/positional adjacency) and LONG-range co-occurrence
// (within 20 positions - closer to broader contextual association).
// Both are thresholded at each matrix's own 60th percentile of nonzero
// pairwise counts (keeps the ~40% most frequent co-occurring pairs as
// "related", adaptive to the corpus rather than a hand-picked constant).
//
// OLD method (exact replication of tiny_lm_corpus.rs's rule on this
// graph): related(i,j) = short(i,j) || long(i,j); predict a node's
// category by hard majority vote over related neighbors' TRUE labels,
// leave-one-out.
//
// NEW method: score(i) = sigmoid(w_short * evidence_short(i) +
// w_long * evidence_long(i) + bias), where evidence_*(i) = sum over
// j != i of (relation(i,j) ? true_label(j) : 0) - the same neighbor-
// evidence features the old rule uses, just combined with LEARNED
// weights instead of an untrained OR. w_short/w_long/bias are 3 scalars
// fit by gradient descent (logistic regression, via this project's own
// tape - no new engine primitives needed, sigmoid is built from
// existing exp/div/add). Evidence still comes from ALL other nodes' true
// labels (same transductive setup the original uses); only the 3
// combination weights are learned, via 5-fold cross-validation so
// "learned" means generalizes to held-out nodes, not memorizes them.
use scratchtape::nn::Rng;
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use scratchtape::tensor::NdArray;

fn encode_bytes(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
}

fn shuffle(items: &mut [usize], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        let j = (rng.next_f32() * (i + 1) as f32) as usize;
        items.swap(i, j);
    }
}

/// Precision/recall/F1 for a binary classifier's predictions against true
/// labels - same reasoning as tiny_lm_corpus.rs's identical helper: most
/// of these categories are well under 50% positive, so accuracy alone
/// can hide a rule that never fires correctly (high accuracy, zero
/// recall).
fn precision_recall_f1(labels: &[usize], predictions: &[usize]) -> (f32, f32, f32) {
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

/// Symmetric co-occurrence counts between distinct byte VALUES (not
/// positions) within `window` positions of each other in the corpus.
fn cooccurrence_counts(bytes: &[usize], node_of: &[i32; 256], n_nodes: usize, window: usize) -> Vec<Vec<u32>> {
    let mut counts = vec![vec![0u32; n_nodes]; n_nodes];
    for center in 0..bytes.len() {
        let a = node_of[bytes[center]];
        if a < 0 {
            continue;
        }
        for offset in 1..=window {
            let Some(other) = bytes.get(center + offset) else { break };
            let b = node_of[*other];
            if b < 0 || b == a {
                continue;
            }
            counts[a as usize][b as usize] += 1;
            counts[b as usize][a as usize] += 1;
        }
    }
    counts
}

/// Adadptive threshold at the given percentile of nonzero counts - keeps
/// the corpus's own statistics in charge of what "related" means instead
/// of a hand-picked magic constant.
fn percentile_threshold(counts: &[Vec<u32>], percentile: f32) -> u32 {
    let mut nonzero: Vec<u32> = counts.iter().flatten().copied().filter(|&c| c > 0).collect();
    if nonzero.is_empty() {
        return 1;
    }
    nonzero.sort_unstable();
    let idx = ((nonzero.len() as f32 * percentile) as usize).min(nonzero.len() - 1);
    nonzero[idx]
}

fn adjacency_from_counts(counts: &[Vec<u32>], threshold: u32) -> Vec<Vec<bool>> {
    counts.iter().map(|row| row.iter().map(|&c| c >= threshold).collect()).collect()
}

/// tiny_lm_corpus.rs's exact rule, replicated on this graph: majority
/// vote of a node's related neighbors' TRUE labels, leave-one-out.
fn hard_majority_vote(labels: &[usize], related: impl Fn(usize, usize) -> bool, n: usize) -> Vec<usize> {
    (0..n)
        .map(|i| {
            let (mut pos, mut neg) = (0, 0);
            for j in 0..n {
                if i == j || !related(i, j) {
                    continue;
                }
                if labels[j] == 1 {
                    pos += 1;
                } else {
                    neg += 1;
                }
            }
            if pos > neg { 1 } else { 0 }
        })
        .collect()
}

fn sigmoid(tape: &mut Tape, x: scratchtape::tape::Var) -> scratchtape::tape::Var {
    let neg_x = tape.scale(x, -1.0);
    let exp_neg_x = tape.exp(neg_x);
    let one = tape.leaf(NdArray::new(vec![1.0; tape.value(exp_neg_x).data.len()], tape.value(exp_neg_x).shape.clone()));
    let denom = tape.add(one, exp_neg_x);
    tape.div(one, denom)
}

/// Fits w_short/w_long/bias by gradient descent on the train fold, then
/// returns predictions for the test fold. Evidence features (x_short,
/// x_long) are fixed, precomputed from ALL n nodes' true labels - only
/// the 3 combination weights are what training actually learns.
fn fit_and_predict(
    x_short: &[f32],
    x_long: &[f32],
    labels: &[usize],
    train_idx: &[usize],
    test_idx: &[usize],
) -> Vec<usize> {
    let train_x_short: Vec<f32> = train_idx.iter().map(|&i| x_short[i]).collect();
    let train_x_long: Vec<f32> = train_idx.iter().map(|&i| x_long[i]).collect();
    let train_y: Vec<f32> = train_idx.iter().map(|&i| labels[i] as f32).collect();
    let n_train = train_idx.len();

    let mut w_short = NdArray::new(vec![0.0], vec![1, 1]);
    let mut w_long = NdArray::new(vec![0.0], vec![1, 1]);
    let mut bias = NdArray::new(vec![0.0], vec![1, 1]);
    let opt = Sgd { lr: 0.3 };

    let steps = 300;
    for _ in 0..steps {
        let mut tape = Tape::new();
        let xs = tape.leaf(NdArray::new(train_x_short.clone(), vec![n_train, 1]));
        let xl = tape.leaf(NdArray::new(train_x_long.clone(), vec![n_train, 1]));
        let y = tape.leaf(NdArray::new(train_y.clone(), vec![n_train, 1]));
        let ws = tape.leaf(w_short.clone());
        let wl = tape.leaf(w_long.clone());
        let b = tape.leaf(bias.clone());

        let term_short = tape.mul(xs, ws);
        let term_long = tape.mul(xl, wl);
        let sum_terms_lin = tape.add(term_short, term_long);
        let logit = tape.add(sum_terms_lin, b);
        let p = sigmoid(&mut tape, logit);

        // Binary cross-entropy: -(y*log(p) + (1-y)*log(1-p)), averaged.
        let one = tape.leaf(NdArray::new(vec![1.0; n_train], vec![n_train, 1]));
        let one_minus_y = tape.sub(one, y);
        let one_minus_p = tape.sub(one, p);
        let log_p = tape.log(p);
        let log_one_minus_p = tape.log(one_minus_p);
        let term1 = tape.mul(y, log_p);
        let term2 = tape.mul(one_minus_y, log_one_minus_p);
        let sum_terms = tape.add(term1, term2);
        let neg_sum = tape.scale(sum_terms, -1.0 / n_train as f32);
        let loss = tape.sum(neg_sum);

        tape.backward(loss);
        opt.step(&mut w_short, tape.grad(ws).unwrap());
        opt.step(&mut w_long, tape.grad(wl).unwrap());
        opt.step(&mut bias, tape.grad(b).unwrap());
    }

    test_idx
        .iter()
        .map(|&i| {
            let logit = w_short.data[0] * x_short[i] + w_long.data[0] * x_long[i] + bias.data[0];
            if 1.0 / (1.0 + (-logit).exp()) > 0.5 { 1 } else { 0 }
        })
        .collect()
}

fn main() {
    let corpus = encode_bytes(include_str!("../data/aesops_fables.txt"));

    let mut node_of = [-1i32; 256];
    let mut values: Vec<usize> = Vec::new();
    for &b in &corpus {
        if node_of[b] < 0 {
            node_of[b] = values.len() as i32;
            values.push(b);
        }
    }
    let n = values.len();
    println!("corpus: {} bytes, {n} distinct byte values (nodes)", corpus.len());

    let short_counts = cooccurrence_counts(&corpus, &node_of, n, 3);
    let long_counts = cooccurrence_counts(&corpus, &node_of, n, 20);
    let short_threshold = percentile_threshold(&short_counts, 0.6);
    let long_threshold = percentile_threshold(&long_counts, 0.6);
    let short_adj = adjacency_from_counts(&short_counts, short_threshold);
    let long_adj = adjacency_from_counts(&long_counts, long_threshold);
    let short_edges: usize = short_adj.iter().map(|r| r.iter().filter(|&&x| x).count()).sum();
    let long_edges: usize = long_adj.iter().map(|r| r.iter().filter(|&&x| x).count()).sum();
    println!(
        "short-range (window=3) threshold={short_threshold}, {short_edges} directed edges; \
         long-range (window=20) threshold={long_threshold}, {long_edges} directed edges"
    );

    let is_vowel = |b: usize| matches!((b as u8) as char, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U');
    let is_uppercase = |b: usize| ((b as u8) as char).is_ascii_uppercase();
    let is_digit = |b: usize| ((b as u8) as char).is_ascii_digit();
    let is_punctuation = |b: usize| ((b as u8) as char).is_ascii_punctuation();
    let is_uppercase_vowel = |b: usize| matches!((b as u8) as char, 'A' | 'E' | 'I' | 'O' | 'U');
    let is_lowercase_vowel = |b: usize| matches!((b as u8) as char, 'a' | 'e' | 'i' | 'o' | 'u');
    let categories: [(&str, &dyn Fn(usize) -> bool); 6] = [
        ("is-vowel", &is_vowel),
        ("is-uppercase", &is_uppercase),
        ("is-digit", &is_digit),
        ("is-punctuation", &is_punctuation),
        ("is-uppercase-vowel", &is_uppercase_vowel),
        ("is-lowercase-vowel", &is_lowercase_vowel),
    ];

    let related = |i: usize, j: usize| short_adj[i][j] || long_adj[i][j];
    let mut rng = Rng::new(1);
    let k_folds = 5;

    println!("\n{:<20} {:>28} | {:>28}", "category", "OLD: hard OR + majority vote", "NEW: learned weighted combination");
    for (name, predicate) in categories {
        let labels: Vec<usize> = values.iter().map(|&b| if predicate(b) { 1 } else { 0 }).collect();
        let positives = labels.iter().filter(|&&l| l == 1).count();

        let old_predictions = hard_majority_vote(&labels, related, n);
        let (old_p, old_r, old_f1) = precision_recall_f1(&labels, &old_predictions);
        let old_correct = (0..n).filter(|&i| old_predictions[i] == labels[i]).count();

        let x_short: Vec<f32> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i && short_adj[i][j]).map(|j| labels[j] as f32).sum())
            .collect();
        let x_long: Vec<f32> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i && long_adj[i][j]).map(|j| labels[j] as f32).sum())
            .collect();

        let mut order: Vec<usize> = (0..n).collect();
        shuffle(&mut order, &mut rng);
        let mut new_predictions = vec![0usize; n];
        for fold in 0..k_folds {
            let test_idx: Vec<usize> = order.iter().enumerate().filter(|(k, _)| k % k_folds == fold).map(|(_, &i)| i).collect();
            let train_idx: Vec<usize> = order.iter().enumerate().filter(|(k, _)| k % k_folds != fold).map(|(_, &i)| i).collect();
            let fold_predictions = fit_and_predict(&x_short, &x_long, &labels, &train_idx, &test_idx);
            for (&i, &p) in test_idx.iter().zip(fold_predictions.iter()) {
                new_predictions[i] = p;
            }
        }
        let (new_p, new_r, new_f1) = precision_recall_f1(&labels, &new_predictions);
        let new_correct = (0..n).filter(|&i| new_predictions[i] == labels[i]).count();

        println!(
            "{name:<20} ({positives}/{n} positive) acc={:.3} p={old_p:.3} r={old_r:.3} f1={old_f1:.3} | acc={:.3} p={new_p:.3} r={new_r:.3} f1={new_f1:.3}",
            old_correct as f32 / n as f32,
            new_correct as f32 / n as f32,
        );
    }
}
