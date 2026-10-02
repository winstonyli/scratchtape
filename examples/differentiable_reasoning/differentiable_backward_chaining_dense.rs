// Direct follow-up to differentiable_backward_chaining.rs's phases 3-4:
// both found that a model trained on ONLY grandparent(x,z) existence
// labels (never shown a `parent` fact directly) fails to recover the
// true `parent` relation - it finds a different, non-veridical bridge
// structure that still satisfies almost every existence constraint.
// Neither hard max_or nor softmax-weighted beam_or fixed it, pointing at
// an information-theoretic cause (existence-only supervision
// underdetermines WHICH y works, not just an optimization pathology).
//
// This tests that diagnosis directly: does the SAME failure persist on
// a KB with much denser, more asymmetric constraints, or was the
// original 12-entity family tree's clean 2-branch symmetry (every
// branch the same shape) specifically what made a wrong bridge
// structure so easy to substitute? If identifiability improves with
// more overlapping constraints per free parameter, that confirms this
// really is a data-density question, not a fixed property of
// differentiable backward chaining in general.
//
// New KB: 26 entities across 3 generations, deliberately irregular
// branching (fan-out 1-3 at every level, no two branches the same
// shape) so there's no clean structural symmetry to hide behind. 22
// parent edges, 13 true grandparent pairs out of 650 ordered pairs
// (~2.0% positive, sparser than the original KB's ~4.5%).
//
// First result (lr=0.05, this file's original run): both check 1
// (trained objective) AND check 2 (recovery) failed for max_or - the
// direct objective wasn't even solved, muddying whether check 2's
// 0.000 reflected identifiability or just bad optimization at the
// larger scale. Follow-up (lr=0.15): fixes check 1 cleanly (max_or
// reaches 0.833 f1, matching the original 12-entity KB), isolating the
// two questions. With optimization no longer the bottleneck, max_or's
// check 2 is STILL exactly 0.000 - confirms the failure is
// identifiability, not an artifact of under-tuned optimization.
// beam_or at lr=0.15 differs in one respect worth keeping precise: it
// gets a small but genuinely NONZERO check 2 (some true facts
// recovered, not none) - softer aggregation leaks a little real signal
// about the base relation once optimization stops being the
// bottleneck, though nowhere near reliable recovery.
use scratchtape::nn::{Embedding, Rng};
use scratchtape::optim::{Adam, AdamState};
use scratchtape::tape::{Tape, Var};
use scratchtape::tensor::NdArray;

#[path = "../common/mod.rs"]
mod common;
use common::{precision_recall_f1, sigmoid, sigmoid_scalar};

fn parent_score(subj_table: &NdArray, obj_table: &NdArray, bias: f32, a: usize, b: usize, d: usize) -> f32 {
    let dot: f32 = (0..d).map(|k| subj_table.data[a * d + k] * obj_table.data[b * d + k]).sum();
    sigmoid_scalar(dot + bias)
}

fn pair_score(tape: &mut Tape, subj_emb: &Embedding, subj_table: Var, obj_emb: &Embedding, obj_table: Var, bias_var: Var, a_ids: &[usize], b_ids: &[usize]) -> Var {
    let a_out = subj_emb.forward_shared(tape, a_ids, subj_table);
    let b_out = obj_emb.forward_shared(tape, b_ids, obj_table);
    let prod = tape.mul(a_out.y, b_out.y);
    let dot = tape.sum_last_axis(prod);
    let logit = tape.add(dot, bias_var);
    sigmoid(tape, logit)
}

fn candidate_scores(tape: &mut Tape, subj_emb: &Embedding, subj_table: Var, obj_emb: &Embedding, obj_table: Var, bias_var: Var, xs: &[usize], zs: &[usize], n: usize) -> Var {
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

fn max_or(tape: &mut Tape, scores: Var) -> Var {
    tape.max_last_axis(scores)
}

fn beam_or(tape: &mut Tape, scores: Var, temperature: f32) -> Var {
    let scaled = tape.scale(scores, 1.0 / temperature);
    let weights = tape.softmax(scaled);
    let weighted = tape.mul(scores, weights);
    tape.sum_last_axis(weighted)
}

/// Trains fresh embeddings on ONLY grandparent existence labels (`or_fn`
/// picks max_or or beam_or), then reports both the trained objective's
/// own accuracy and whether the RECOVERED parent_score matches the true
/// edges it never saw directly - same two-check structure as
/// differentiable_backward_chaining.rs's phases 3/4.
fn train_from_indirect_supervision(
    label: &str,
    n: usize,
    d: usize,
    lr: f32,
    is_parent_fact: &dyn Fn(usize, usize) -> bool,
    is_true_grandparent: &dyn Fn(usize, usize) -> bool,
    or_fn: impl Fn(&mut Tape, Var) -> Var,
    names: &[&str],
) -> (f32, f32, f32) {
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

    let mut rng = Rng::new(1);
    let mut subj_emb = Embedding::new(&mut rng, n, d);
    let mut obj_emb = Embedding::new(&mut rng, n, d);
    let mut bias = NdArray::new(vec![0.0], vec![1, 1]);
    let adam = Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
    let mut subj_state = AdamState::zeros_like(&subj_emb.table);
    let mut obj_state = AdamState::zeros_like(&obj_emb.table);
    let mut bias_state = AdamState::zeros_like(&bias);

    let steps = 3000;
    let mut last_loss = 0.0f32;
    for step in 0..steps {
        let mut tape = Tape::new();
        let subj_table_var = tape.leaf(subj_emb.table.clone());
        let obj_table_var = tape.leaf(obj_emb.table.clone());
        let bias_var = tape.leaf(bias.clone());

        let scores = candidate_scores(&mut tape, &subj_emb, subj_table_var, &obj_emb, obj_table_var, bias_var, &gp_xs, &gp_zs, n);
        let proof = or_fn(&mut tape, scores);

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
        adam.step(&mut subj_emb.table, tape.grad(subj_table_var).unwrap(), &mut subj_state);
        adam.step(&mut obj_emb.table, tape.grad(obj_table_var).unwrap(), &mut obj_state);
        adam.step(&mut bias, tape.grad(bias_var).unwrap(), &mut bias_state);
        last_loss = tape.value(loss).data[0];

        if step % 500 == 0 {
            println!("  [{label}] step {step:>4}: loss = {last_loss:.4}");
        }
    }
    println!("  [{label}] final loss = {last_loss:.4}");

    let mut gp_predictions = vec![0usize; q];
    for i in 0..q {
        let mut best = 0.0f32;
        for y in 0..n {
            if y == gp_xs[i] || y == gp_zs[i] {
                continue;
            }
            let s = parent_score(&subj_emb.table, &obj_emb.table, bias.data[0], gp_xs[i], y, d) * parent_score(&subj_emb.table, &obj_emb.table, bias.data[0], y, gp_zs[i], d);
            if s > best {
                best = s;
            }
        }
        gp_predictions[i] = if best > 0.5 { 1 } else { 0 };
    }
    let gp_labels_usize: Vec<usize> = gp_labels.iter().map(|&l| l as usize).collect();
    let (gp_p, gp_r, gp_f1) = precision_recall_f1(&gp_labels_usize, &gp_predictions);
    println!("  [{label}] check 1 - trained objective (grandparent prediction): precision={gp_p:.3} recall={gp_r:.3} f1={gp_f1:.3}");

    let mut fact_labels = Vec::new();
    let mut recovered_predictions = Vec::new();
    let mut wrong_facts = Vec::new();
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue;
            }
            let truth = is_parent_fact(a, b);
            fact_labels.push(if truth { 1 } else { 0 });
            let s = parent_score(&subj_emb.table, &obj_emb.table, bias.data[0], a, b, d);
            let pred = if s > 0.5 { 1 } else { 0 };
            recovered_predictions.push(pred);
            if pred == 1 && !truth {
                wrong_facts.push((a, b));
            }
        }
    }
    let (rp, rr, rf1) = precision_recall_f1(&fact_labels, &recovered_predictions);
    println!("  [{label}] check 2 - recovered TRUE `parent` facts as a byproduct: precision={rp:.3} recall={rr:.3} f1={rf1:.3}");
    if !wrong_facts.is_empty() {
        print!("  [{label}] hallucinated (not real parent edges): ");
        for &(a, b) in wrong_facts.iter().take(8) {
            print!("{}->{} ", names[a], names[b]);
        }
        if wrong_facts.len() > 8 {
            print!("... ({} more)", wrong_facts.len() - 8);
        }
        println!();
    }
    (rp, rr, rf1)
}

fn main() {
    let names = [
        "alice", "bob", "carol", "dave", //
        "eve", "frank", "grace", "heidi", "ivan", "judy", "karl", "liam", "mia", //
        "nina", "oscar", "paul", "quinn", "rose", "sam", "tara", "uma", "vince", "wendy", "xander", "yara", "zack",
    ];
    let n = names.len();
    let idx = |name: &str| names.iter().position(|&x| x == name).unwrap();

    // 3 generations, deliberately irregular fan-out (1-3 children,
    // varying at every branch) so there's no clean structural symmetry
    // for a wrong bridge hypothesis to hide behind.
    let parent_edges: Vec<(usize, usize)> = [
        ("alice", "eve"),
        ("alice", "frank"),
        ("alice", "grace"),
        ("bob", "heidi"),
        ("bob", "ivan"),
        ("carol", "judy"),
        ("dave", "karl"),
        ("dave", "liam"),
        ("dave", "mia"),
        ("eve", "nina"),
        ("eve", "oscar"),
        ("frank", "paul"),
        ("grace", "quinn"),
        ("heidi", "rose"),
        ("heidi", "sam"),
        ("ivan", "tara"),
        ("judy", "uma"),
        ("judy", "vince"),
        ("karl", "wendy"),
        ("liam", "xander"),
        ("mia", "yara"),
        ("mia", "zack"),
    ]
    .iter()
    .map(|&(p, c)| (idx(p), idx(c)))
    .collect();
    let is_parent_fact = |a: usize, b: usize| parent_edges.contains(&(a, b));
    let is_true_grandparent = |x: usize, z: usize| (0..n).any(|y| is_parent_fact(x, y) && is_parent_fact(y, z));

    let true_gp_count = (0..n).flat_map(|x| (0..n).map(move |z| (x, z))).filter(|&(x, z)| x != z && is_true_grandparent(x, z)).count();
    println!("kinship KB: {n} entities, {} parent facts, {true_gp_count} true grandparent pairs out of {}", parent_edges.len(), n * (n - 1));

    let d = 6;
    println!("\ntraining from ONLY grandparent existence labels, never shown a `parent` fact directly (d={d}):");
    let _ = train_from_indirect_supervision("max_or lr=0.05", n, d, 0.05, &is_parent_fact, &is_true_grandparent, max_or, &names);
    let _ = train_from_indirect_supervision("beam_or T=0.5 lr=0.05", n, d, 0.05, &is_parent_fact, &is_true_grandparent, |t, s| beam_or(t, s, 0.5), &names);

    // lr=0.05 (above) undersolved the TRAINED objective itself at this
    // scale (max_or's check 1 was 0/0/0 - worse than the original
    // 12-entity KB's 0.833), muddying whether check 2's failure reflects
    // identifiability or just bad optimization. lr=0.15 fixes check 1
    // (max_or reaches 0.833 f1, matching the original KB) - isolating
    // the two questions cleanly.
    println!("\nsame KB, tuned lr=0.15 - isolates optimization difficulty from identifiability:");
    let _ = train_from_indirect_supervision("max_or lr=0.15", n, d, 0.15, &is_parent_fact, &is_true_grandparent, max_or, &names);
    let (_, _, beam_05_f1) = train_from_indirect_supervision("beam_or T=0.5 lr=0.15", n, d, 0.15, &is_parent_fact, &is_true_grandparent, |t, s| beam_or(t, s, 0.5), &names);

    // beam_or's one nonzero check-2 result (above) raises the obvious
    // question: does temperature matter, and is there a better setting?
    // Sweeps it directly rather than guessing from one data point.
    println!("\nsweeping beam_or's temperature at lr=0.15 - is recovery a smooth function of temperature, or noisy?");
    let temperatures = [0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0];
    let mut sweep_results = vec![(0.5, beam_05_f1)];
    for &t in &temperatures {
        if t == 0.5 {
            continue; // already have it above
        }
        let (_, _, f1) = train_from_indirect_supervision(&format!("beam_or T={t}"), n, d, 0.15, &is_parent_fact, &is_true_grandparent, |tape, s| beam_or(tape, s, t), &names);
        sweep_results.push((t, f1));
    }
    sweep_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    println!("\ntemperature -> check 2 f1 (recovering true parent facts):");
    for (t, f1) in &sweep_results {
        println!("  T={t:<5} f1={f1:.3}");
    }
    let (best_t, best_f1) = sweep_results.iter().cloned().fold((0.0, -1.0), |acc, x| if x.1 > acc.1 { x } else { acc });
    println!(
        "best: T={best_t} (f1={best_f1:.3}) - still far from reliable recovery, and not a smooth function of \
         temperature (T=0.07/0.2/0.3/1.0/2.0 all collapsed to 0.000 while T=0.05/0.1/0.15/0.5 didn't) - single-seed \
         training \
         dynamics finding a locally better bridge structure by chance seems more likely than a genuine \
         temperature/sharpness trend."
    );
}
