// A minimal differentiable backward-chaining prover (Rocktaschel & Riedel-
// style NTP, scoped way down) - the higher-risk option surveyed alongside
// differentiable_forward_chaining.rs's soft rule-weighting. That file
// avoided real OR/AND-module recursion as too awkward for this project's
// append-only, straight-line-per-step tape. On reflection that risk was
// overstated: NTP proof depth is bounded in every practical
// implementation anyway (unbounded unification against a cyclic KB would
// never terminate), so it doesn't need genuine Rust-level recursion -
// same as ssm_recall.rs's recurrence, a bounded depth can just be
// unrolled into ordinary Rust control flow. This unrolls to depth 2.
//
// Toy kinship KB (matches the literature's own benchmark scale - Kinship,
// Countries, UMLS are all small structured KGs, not real-corpus scale,
// so there's no "toy vs real" scale-up question here the way the memory-
// tier line had): 12 people, one base relation (`parent`, 10 hand-
// authored ground-truth edges across 3 generations), one rule template
// (`grandparent(X,Z) :- parent(X,Y), parent(Y,Z)`), no grandparent facts
// given directly - only derivable by chaining `parent` twice.
//
// Phase 1 (fact embedding): learn `parent` as a soft, embedding-based
// relation - score(a,b) = sigmoid(dot(subj_emb[a], obj_emb[b]) + bias),
// two role-specific embedding tables (not one, since "parent" isn't
// symmetric) - trained on the 10 true edges against every other ordered
// pair as a negative, deliberately compressed to d=4 dims for 12
// entities so the model can't just memorize an arbitrary lookup table.
//
// Phase 2 (the actual test - zero-shot compositional generalization, no
// further training): NTP's OR-module is existential search over the
// unbound variable Y - try every entity as a candidate, score
// grandparent(x,z) = max_y parent_score(x,y) * parent_score(y,z). AND-
// module is the product (propagating both atoms' scores, same
// "propagating minimum/product scores through logical paths" the
// literature describes). Evaluated forward-only on FROZEN phase-1
// embeddings - if this predicts true grandparent pairs correctly, the
// model composed two 1-hop facts it was never shown together into a
// 2-hop conclusion, purely through soft unification. No new tape
// primitive needed for this: the max here is over plain f32s outside
// the tape, not backpropagated through - see the closing note on why.
use scratchtape::nn::{Embedding, Rng};
use scratchtape::optim::{Adam, AdamState};
use scratchtape::tape::Tape;
use scratchtape::tensor::NdArray;

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

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reads frozen phase-1 embeddings directly - no tape involved, this is
/// the "already learned" fact, not something being trained further.
fn parent_score(subj_table: &NdArray, obj_table: &NdArray, bias: f32, a: usize, b: usize, d: usize) -> f32 {
    let dot: f32 = (0..d).map(|k| subj_table.data[a * d + k] * obj_table.data[b * d + k]).sum();
    sigmoid_scalar(dot + bias)
}

fn main() {
    let names = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi", "ivan", "judy", "karl", "liam"];
    let n = names.len();
    let idx = |name: &str| names.iter().position(|&x| x == name).unwrap();

    // 3-generation family tree, hand-authored so the true grandparent set
    // is known exactly (computed below from these edges, not asserted).
    let parent_edges: Vec<(usize, usize)> = [
        ("alice", "carol"), ("alice", "dave"),
        ("bob", "eve"), ("bob", "frank"),
        ("carol", "grace"), ("carol", "heidi"),
        ("dave", "ivan"),
        ("eve", "judy"),
        ("frank", "karl"), ("frank", "liam"),
    ]
    .iter()
    .map(|&(p, c)| (idx(p), idx(c)))
    .collect();
    let is_parent_fact = |a: usize, b: usize| parent_edges.contains(&(a, b));

    // Ground truth grandparent(x,z): exists y with parent(x,y) and
    // parent(y,z) - computed directly from the edges above, independent
    // of anything the model will learn, so phase 2 has an honest target.
    let is_true_grandparent = |x: usize, z: usize| (0..n).any(|y| is_parent_fact(x, y) && is_parent_fact(y, z));

    println!("kinship KB: {n} entities, {} parent facts", parent_edges.len());

    // Phase 1: learn `parent` as a soft, embedding-based relation.
    let d = 4;
    let mut rng = Rng::new(1);
    let mut subj_emb = Embedding::new(&mut rng, n, d);
    let mut obj_emb = Embedding::new(&mut rng, n, d);
    let mut bias = NdArray::new(vec![0.0], vec![1, 1]);
    let adam = Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let mut subj_state = AdamState::zeros_like(&subj_emb.table);
    let mut obj_state = AdamState::zeros_like(&obj_emb.table);
    let mut bias_state = AdamState::zeros_like(&bias);

    let mut subj_ids = Vec::new();
    let mut obj_ids = Vec::new();
    let mut fact_labels = Vec::new();
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue;
            }
            subj_ids.push(a);
            obj_ids.push(b);
            fact_labels.push(if is_parent_fact(a, b) { 1.0 } else { 0.0 });
        }
    }
    let n_pairs = subj_ids.len();

    let steps = 2000;
    for step in 0..steps {
        let mut tape = Tape::new();
        let subj_out = subj_emb.forward(&mut tape, &subj_ids);
        let obj_out = obj_emb.forward(&mut tape, &obj_ids);
        let prod = tape.mul(subj_out.y, obj_out.y);
        let dot = tape.sum_last_axis(prod);
        let b_var = tape.leaf(bias.clone());
        let logit = tape.add(dot, b_var);

        let neg_logit = tape.scale(logit, -1.0);
        let exp_neg = tape.exp(neg_logit);
        let one = tape.leaf(NdArray::new(vec![1.0; n_pairs], vec![n_pairs, 1]));
        let denom = tape.add(one, exp_neg);
        let p = tape.div(one, denom);

        let y = tape.leaf(NdArray::new(fact_labels.clone(), vec![n_pairs, 1]));
        let one_minus_y = tape.sub(one, y);
        let one_minus_p = tape.sub(one, p);
        let log_p = tape.log(p);
        let log_one_minus_p = tape.log(one_minus_p);
        let term1 = tape.mul(y, log_p);
        let term2 = tape.mul(one_minus_y, log_one_minus_p);
        let sum_terms = tape.add(term1, term2);
        let neg_mean = tape.scale(sum_terms, -1.0 / n_pairs as f32);
        let loss = tape.sum(neg_mean);

        tape.backward(loss);
        // Direct Adam::step on the tables (not apply_grad, which is
        // Sgd-only) - same pattern ssm_recall.rs uses for weight-tied
        // Adam. Safe here too: each step is a fresh forward+backward,
        // no reuse of a stale EmbeddingOut across multiple forwards.
        adam.step(&mut subj_emb.table, tape.grad(subj_out.table).unwrap(), &mut subj_state);
        adam.step(&mut obj_emb.table, tape.grad(obj_out.table).unwrap(), &mut obj_state);
        adam.step(&mut bias, tape.grad(b_var).unwrap(), &mut bias_state);

        if step % 200 == 0 {
            println!("  phase 1 step {step:>3}: loss = {:.4}", tape.value(loss).data[0]);
        }
    }

    // Sanity check: did phase 1 actually recover the known facts, before
    // trusting phase 2's composition of them?
    let mut fact_predictions = vec![0usize; n_pairs];
    for i in 0..n_pairs {
        let s = parent_score(&subj_emb.table, &obj_emb.table, bias.data[0], subj_ids[i], obj_ids[i], d);
        fact_predictions[i] = if s > 0.5 { 1 } else { 0 };
    }
    let fact_labels_usize: Vec<usize> = fact_labels.iter().map(|&l| l as usize).collect();
    let (fp, fr, ff1) = precision_recall_f1(&fact_labels_usize, &fact_predictions);
    println!("phase 1 result - recovering the 10 known `parent` facts themselves: precision={fp:.3} recall={fr:.3} f1={ff1:.3}");

    // Phase 2: zero-shot backward chaining. OR-module = max over the
    // existentially-quantified Y; AND-module = product of the two atoms'
    // scores. No training happens here - these are frozen phase-1
    // embeddings, this is purely inference.
    let mut labels = Vec::new();
    let mut predictions = Vec::new();
    for x in 0..n {
        for z in 0..n {
            if x == z {
                continue;
            }
            let mut best = 0.0f32;
            for y in 0..n {
                if y == x || y == z {
                    continue;
                }
                let s = parent_score(&subj_emb.table, &obj_emb.table, bias.data[0], x, y, d)
                    * parent_score(&subj_emb.table, &obj_emb.table, bias.data[0], y, z, d);
                if s > best {
                    best = s;
                }
            }
            labels.push(if is_true_grandparent(x, z) { 1 } else { 0 });
            predictions.push(if best > 0.5 { 1 } else { 0 });
        }
    }
    let true_count = labels.iter().filter(|&&l| l == 1).count();
    let (p, r, f1) = precision_recall_f1(&labels, &predictions);
    let correct = labels.iter().zip(predictions.iter()).filter(|(l, p)| l == p).count();
    println!(
        "\nphase 2 result - zero-shot grandparent(x,z) via backward chaining over frozen `parent` embeddings:"
    );
    println!(
        "  {true_count}/{} true grandparent pairs, accuracy={:.3} precision={p:.3} recall={r:.3} f1={f1:.3}",
        labels.len(),
        correct as f32 / labels.len() as f32
    );

    println!("\ntrue grandparent pairs and what the prover scored them:");
    for x in 0..n {
        for z in 0..n {
            if x != z && is_true_grandparent(x, z) {
                let mut best = 0.0f32;
                let mut best_y = 0;
                for y in 0..n {
                    if y == x || y == z {
                        continue;
                    }
                    let s = parent_score(&subj_emb.table, &obj_emb.table, bias.data[0], x, y, d)
                        * parent_score(&subj_emb.table, &obj_emb.table, bias.data[0], y, z, d);
                    if s > best {
                        best = s;
                        best_y = y;
                    }
                }
                println!("  grandparent({}, {}) via {} - score={best:.3}", names[x], names[z], names[best_y]);
            }
        }
    }
}
