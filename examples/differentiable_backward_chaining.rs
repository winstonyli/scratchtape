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
// the tape, not backpropagated through.
//
// Phase 3 (added as a follow-up - does backprop through the OR-module's
// max actually matter?): first checked whether continuing phase 1's
// single-relation training a few more steps would trivially close phase
// 2's gap on its own - at this toy scale it does (step 60 has 0.833
// grandparent recall, step 62 already hits 1.000), so there's no
// meaningful "does end-to-end help beyond more of the same" question to
// ask on this KB. The actually decisive test is different: can the model
// learn `parent` from ONLY 2-hop grandparent supervision, never shown a
// single `parent` fact directly? That needs a genuinely differentiable
// OR-module - gradient has to flow back through `max` to shape the base
// relation embeddings at all. Added `Tape::max_last_axis` (src/tape.rs)
// for this, gradient-checked like every other op here (routes the whole
// incoming gradient to whichever candidate actually won the max, zero
// elsewhere - standard max-pool backward). Fresh embeddings, trained
// end-to-end via the same OR/AND-module formula batched over the tape
// (n score columns concatenated, then max_last_axis - Tape::concat and
// Tape::max_last_axis are both first-class differentiable ops, so
// gradient reaches every candidate's embeddings, not just the winner's,
// across the whole training run). Checked afterward: does the model's
// RECOVERED parent_score, which it was never directly supervised on,
// actually match the 10 true parent facts - or did it find some other,
// non-veridical way to satisfy the observed 2-hop constraints?
//
// Phase 4: tests the literature's own named fix for phase 3's failure -
// "propagating loss across a beam of top-k proof paths" instead of
// greedy single-path max. Implemented as beam_or: a softmax-weighted
// average over candidates (needs zero new tape primitives - just
// existing softmax+mul+sum_last_axis composed together), so every near-
// winning candidate gets gradient each step, not only the current
// argmax. Result: it doesn't fix check 2 - still 0/0/0 recovering the
// true facts, and at temperature=0.5 it hallucinates MORE spurious
// relations than phase 3's hard max did (22 vs 7), not fewer. A much
// sharper temperature (0.05, close to hard max) was also tried and
// collapsed differently - it failed even the TRAINED objective (check 1
// also 0/0/0), suggesting numerical/optimization sensitivity at that
// extreme rather than a cleaner result. The honest reading: beam/soft-OR
// is a fix for greedy-max's OPTIMIZATION pathology (getting stuck in a
// bad local minimum during search) - it is not a fix for this task's
// actual problem, which is INFORMATION-theoretic: existence-only
// supervision ("some y satisfies this" - never which y) genuinely
// underdetermines the base relation among several equally-consistent
// alternatives, and no amount of smoothing the aggregation function can
// inject bridge-identity information that was never in the training
// signal to begin with.
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

fn sigmoid(tape: &mut Tape, x: scratchtape::tape::Var) -> scratchtape::tape::Var {
    let neg_x = tape.scale(x, -1.0);
    let exp_neg_x = tape.exp(neg_x);
    let one = tape.leaf(NdArray::new(vec![1.0; tape.value(exp_neg_x).data.len()], tape.value(exp_neg_x).shape.clone()));
    let denom = tape.add(one, exp_neg_x);
    tape.div(one, denom)
}

/// score(a,b) = sigmoid(dot(subj_emb[a], obj_emb[b]) + bias), batched over
/// a whole id list at once - `forward_shared` (not `forward`) on
/// pre-leafed `subj_table`/`obj_table`, since phase 3 calls this many
/// times per step against the SAME embeddings (once per OR-module
/// candidate). Plain `forward` would leaf a fresh copy each call - the
/// exact weight-tying footgun Linear/Embedding's own doc comments warn
/// about, since only the LAST leaf's gradient would then be visible.
fn pair_score(
    tape: &mut Tape,
    subj_emb: &Embedding,
    subj_table: scratchtape::tape::Var,
    obj_emb: &Embedding,
    obj_table: scratchtape::tape::Var,
    bias_var: scratchtape::tape::Var,
    a_ids: &[usize],
    b_ids: &[usize],
) -> scratchtape::tape::Var {
    let a_out = subj_emb.forward_shared(tape, a_ids, subj_table);
    let b_out = obj_emb.forward_shared(tape, b_ids, obj_table);
    let prod = tape.mul(a_out.y, b_out.y);
    let dot = tape.sum_last_axis(prod);
    let logit = tape.add(dot, bias_var);
    sigmoid(tape, logit)
}

/// The AND-module, batched over a whole query set and every candidate Y
/// at once: for each of the n entities as a candidate, computes
/// parent_score(x,y)*parent_score(y,z) (product of both atoms) as one
/// column, concatenates all n candidates' columns (Tape::concat, along
/// the last axis - exactly what it's for) into one [q, n] matrix. The
/// OR-module (how to reduce across candidates) is deliberately a
/// separate step - see max_or/beam_or below, both consume this same
/// matrix.
fn candidate_scores(
    tape: &mut Tape,
    subj_emb: &Embedding,
    subj_table: scratchtape::tape::Var,
    obj_emb: &Embedding,
    obj_table: scratchtape::tape::Var,
    bias_var: scratchtape::tape::Var,
    xs: &[usize],
    zs: &[usize],
    n: usize,
) -> scratchtape::tape::Var {
    let q = xs.len();
    let mut columns = Vec::with_capacity(n);
    for y in 0..n {
        let y_ids = vec![y; q];
        let s1 = pair_score(tape, subj_emb, subj_table, obj_emb, obj_table, bias_var, xs, &y_ids);
        let s2 = pair_score(tape, subj_emb, subj_table, obj_emb, obj_table, bias_var, &y_ids, zs);
        columns.push(tape.mul(s1, s2));
    }
    tape.concat(&columns)
}

/// OR-module v1: hard max over candidates (phase 3's version, kept for
/// comparison) - existential search over Y, but only the single winning
/// candidate gets any gradient each step.
fn max_or(tape: &mut Tape, scores: scratchtape::tape::Var) -> scratchtape::tape::Var {
    tape.max_last_axis(scores)
}

/// OR-module v2 (phase 4): a smooth "beam" relaxation of top-k proof
/// supervision - the literature's own named fix for greedy-max's local
/// minima (see differentiable_backward_chaining.rs's phase 3 doc
/// comment). Needs no new tape primitive at all: a softmax-weighted
/// average over candidates (attention over proof paths, closer to how
/// Conditional Theorem Provers replace exhaustive enumeration with
/// attention-based clause selection) is just existing softmax + mul +
/// sum_last_axis composed together - every near-winning candidate gets
/// gradient proportional to its weight, not just whichever one is
/// currently ahead. `temperature` controls how close to hard-max this
/// gets: low temperature sharpens toward max_or, high temperature
/// spreads weight more broadly across all n candidates.
fn beam_or(tape: &mut Tape, scores: scratchtape::tape::Var, temperature: f32) -> scratchtape::tape::Var {
    let scaled = tape.scale(scores, 1.0 / temperature);
    let weights = tape.softmax(scaled);
    let weighted = tape.mul(scores, weights);
    tape.sum_last_axis(weighted)
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

    // Phase 3: learn `parent` from ONLY 2-hop grandparent supervision -
    // fresh embeddings, never shown a single `parent` fact directly. This
    // is the test that actually needs gradient to flow through the OR-
    // module's max, unlike phases 1-2.
    println!("\nphase 3: learning `parent` from ONLY grandparent supervision (fresh embeddings, no direct parent facts shown)");
    let mut gp_xs = Vec::new();
    let mut gp_zs = Vec::new();
    let mut gp_labels = Vec::new();
    for x in 0..n {
        for z in 0..n {
            if x == z {
                continue;
            }
            gp_xs.push(x);
            gp_zs.push(z);
            gp_labels.push(if is_true_grandparent(x, z) { 1.0 } else { 0.0 });
        }
    }
    let q = gp_xs.len();

    let mut rng3 = Rng::new(1);
    let mut subj_emb3 = Embedding::new(&mut rng3, n, d);
    let mut obj_emb3 = Embedding::new(&mut rng3, n, d);
    let mut bias3 = NdArray::new(vec![0.0], vec![1, 1]);
    let adam3 = Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let mut subj_state3 = AdamState::zeros_like(&subj_emb3.table);
    let mut obj_state3 = AdamState::zeros_like(&obj_emb3.table);
    let mut bias_state3 = AdamState::zeros_like(&bias3);

    let steps3 = 2000;
    for step in 0..steps3 {
        let mut tape = Tape::new();
        let subj_table_var = tape.leaf(subj_emb3.table.clone());
        let obj_table_var = tape.leaf(obj_emb3.table.clone());
        let bias_var = tape.leaf(bias3.clone());

        let scores = candidate_scores(&mut tape, &subj_emb3, subj_table_var, &obj_emb3, obj_table_var, bias_var, &gp_xs, &gp_zs, n);
        let proof = max_or(&mut tape, scores);

        let one = tape.leaf(NdArray::new(vec![1.0; q], vec![q, 1]));
        let y = tape.leaf(NdArray::new(gp_labels.clone(), vec![q, 1]));
        let eps = tape.leaf(NdArray::new(vec![1e-6; q], vec![q, 1]));
        let proof_eps = tape.add(proof, eps);
        let one_minus_proof = tape.sub(one, proof);
        let one_minus_proof_eps = tape.add(one_minus_proof, eps);
        let one_minus_y = tape.sub(one, y);
        let log_p = tape.log(proof_eps);
        let log_one_minus_p = tape.log(one_minus_proof_eps);
        let term1 = tape.mul(y, log_p);
        let term2 = tape.mul(one_minus_y, log_one_minus_p);
        let sum_terms = tape.add(term1, term2);
        let neg_mean = tape.scale(sum_terms, -1.0 / q as f32);
        let loss = tape.sum(neg_mean);

        tape.backward(loss);
        adam3.step(&mut subj_emb3.table, tape.grad(subj_table_var).unwrap(), &mut subj_state3);
        adam3.step(&mut obj_emb3.table, tape.grad(obj_table_var).unwrap(), &mut obj_state3);
        adam3.step(&mut bias3, tape.grad(bias_var).unwrap(), &mut bias_state3);

        if step % 200 == 0 {
            println!("  phase 3 step {step:>4}: loss = {:.4}", tape.value(loss).data[0]);
        }
    }

    // Check 1: did phase 3 solve the objective it was actually trained on?
    let mut gp_predictions = vec![0usize; q];
    for i in 0..q {
        let mut best = 0.0f32;
        for y in 0..n {
            if y == gp_xs[i] || y == gp_zs[i] {
                continue;
            }
            let s = parent_score(&subj_emb3.table, &obj_emb3.table, bias3.data[0], gp_xs[i], y, d)
                * parent_score(&subj_emb3.table, &obj_emb3.table, bias3.data[0], y, gp_zs[i], d);
            if s > best {
                best = s;
            }
        }
        gp_predictions[i] = if best > 0.5 { 1 } else { 0 };
    }
    let gp_labels_usize: Vec<usize> = gp_labels.iter().map(|&l| l as usize).collect();
    let (gp_p, gp_r, gp_f1) = precision_recall_f1(&gp_labels_usize, &gp_predictions);
    println!("phase 3 check 1 - the trained objective (grandparent prediction): precision={gp_p:.3} recall={gp_r:.3} f1={gp_f1:.3}");

    // Check 2 (the actually interesting one): does the RECOVERED
    // parent_score match the 10 true parent facts it never saw directly?
    let mut recovered_predictions = vec![0usize; n_pairs];
    for i in 0..n_pairs {
        let s = parent_score(&subj_emb3.table, &obj_emb3.table, bias3.data[0], subj_ids[i], obj_ids[i], d);
        recovered_predictions[i] = if s > 0.5 { 1 } else { 0 };
    }
    let (rp, rr, rf1) = precision_recall_f1(&fact_labels_usize, &recovered_predictions);
    println!(
        "phase 3 check 2 - did it recover the TRUE `parent` facts as a byproduct: precision={rp:.3} recall={rr:.3} f1={rf1:.3}"
    );
    println!("  (facts it inferred as `parent` that were never directly supervised:)");
    for i in 0..n_pairs {
        if recovered_predictions[i] == 1 {
            let matches_truth = if fact_labels_usize[i] == 1 { "true parent fact" } else { "NOT a true parent fact" };
            println!("    parent({}, {}) - {matches_truth}", names[subj_ids[i]], names[obj_ids[i]]);
        }
    }

    // Phase 4: same setup as phase 3 (fresh embeddings, only grandparent
    // supervision, never shown a `parent` fact directly) but with beam_or
    // instead of max_or - does spreading gradient across every near-
    // winning candidate, not just the single argmax, actually fix what
    // phase 3 got wrong?
    println!("\nphase 4: same as phase 3, but beam_or (softmax-weighted, temperature=0.5) instead of hard max_or");
    let mut rng4 = Rng::new(1);
    let mut subj_emb4 = Embedding::new(&mut rng4, n, d);
    let mut obj_emb4 = Embedding::new(&mut rng4, n, d);
    let mut bias4 = NdArray::new(vec![0.0], vec![1, 1]);
    let adam4 = Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let mut subj_state4 = AdamState::zeros_like(&subj_emb4.table);
    let mut obj_state4 = AdamState::zeros_like(&obj_emb4.table);
    let mut bias_state4 = AdamState::zeros_like(&bias4);
    let temperature = 0.5;

    let steps4 = 2000;
    for step in 0..steps4 {
        let mut tape = Tape::new();
        let subj_table_var = tape.leaf(subj_emb4.table.clone());
        let obj_table_var = tape.leaf(obj_emb4.table.clone());
        let bias_var = tape.leaf(bias4.clone());

        let scores = candidate_scores(&mut tape, &subj_emb4, subj_table_var, &obj_emb4, obj_table_var, bias_var, &gp_xs, &gp_zs, n);
        let proof = beam_or(&mut tape, scores, temperature);

        let one = tape.leaf(NdArray::new(vec![1.0; q], vec![q, 1]));
        let y = tape.leaf(NdArray::new(gp_labels.clone(), vec![q, 1]));
        let eps = tape.leaf(NdArray::new(vec![1e-6; q], vec![q, 1]));
        let proof_eps = tape.add(proof, eps);
        let one_minus_proof = tape.sub(one, proof);
        let one_minus_proof_eps = tape.add(one_minus_proof, eps);
        let one_minus_y = tape.sub(one, y);
        let log_p = tape.log(proof_eps);
        let log_one_minus_p = tape.log(one_minus_proof_eps);
        let term1 = tape.mul(y, log_p);
        let term2 = tape.mul(one_minus_y, log_one_minus_p);
        let sum_terms = tape.add(term1, term2);
        let neg_mean = tape.scale(sum_terms, -1.0 / q as f32);
        let loss = tape.sum(neg_mean);

        tape.backward(loss);
        adam4.step(&mut subj_emb4.table, tape.grad(subj_table_var).unwrap(), &mut subj_state4);
        adam4.step(&mut obj_emb4.table, tape.grad(obj_table_var).unwrap(), &mut obj_state4);
        adam4.step(&mut bias4, tape.grad(bias_var).unwrap(), &mut bias_state4);

        if step % 200 == 0 {
            println!("  phase 4 step {step:>4}: loss = {:.4}", tape.value(loss).data[0]);
        }
    }

    let mut gp_predictions4 = vec![0usize; q];
    for i in 0..q {
        let mut best = 0.0f32;
        for y in 0..n {
            if y == gp_xs[i] || y == gp_zs[i] {
                continue;
            }
            let s = parent_score(&subj_emb4.table, &obj_emb4.table, bias4.data[0], gp_xs[i], y, d)
                * parent_score(&subj_emb4.table, &obj_emb4.table, bias4.data[0], y, gp_zs[i], d);
            if s > best {
                best = s;
            }
        }
        gp_predictions4[i] = if best > 0.5 { 1 } else { 0 };
    }
    let (gp_p4, gp_r4, gp_f14) = precision_recall_f1(&gp_labels_usize, &gp_predictions4);
    println!("phase 4 check 1 - the trained objective (grandparent prediction): precision={gp_p4:.3} recall={gp_r4:.3} f1={gp_f14:.3}");

    let mut recovered_predictions4 = vec![0usize; n_pairs];
    for i in 0..n_pairs {
        let s = parent_score(&subj_emb4.table, &obj_emb4.table, bias4.data[0], subj_ids[i], obj_ids[i], d);
        recovered_predictions4[i] = if s > 0.5 { 1 } else { 0 };
    }
    let (rp4, rr4, rf14) = precision_recall_f1(&fact_labels_usize, &recovered_predictions4);
    println!(
        "phase 4 check 2 - did it recover the TRUE `parent` facts as a byproduct: precision={rp4:.3} recall={rr4:.3} f1={rf14:.3}"
    );
    println!("  (facts it inferred as `parent` that were never directly supervised:)");
    for i in 0..n_pairs {
        if recovered_predictions4[i] == 1 {
            let matches_truth = if fact_labels_usize[i] == 1 { "true parent fact" } else { "NOT a true parent fact" };
            println!("    parent({}, {}) - {matches_truth}", names[subj_ids[i]], names[obj_ids[i]]);
        }
    }
}
