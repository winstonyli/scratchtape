// Actually diagnoses the softmax1-in-a-real-transformer divergence
// ([d54c8f2] "Try softmax1 against the attention-frequency-sink
// finding: diverges regardless of lr, parked") instead of resting on
// that commit's closing comparison to the PC-precision instabilities.
// That comparison was never tested - this builds the instrumentation
// needed to actually find out whether the mechanism is the same
// (a discontinuous jump, fixable by a smooth ramp like PC's fix was)
// or something else specific to softmax1's own semantics.
//
// Original hypothesis (kept for the record, then refuted): softmax1
// lets attention weights sum to LESS than 1, so a head that "opts out"
// a lot shrinks the attention layer's contribution to the residual
// stream; downstream layers (out_proj, ffn) might compensate by
// growing their own weights, compounding into the observed runaway.
// Measured directly and refuted: out_proj/ffn2 weight norms stay
// essentially flat the entire run (16.0 -> 16.07 over 775 steps) - no
// compensation happening there at all.
//
// What's actually happening, found by instrumenting the real
// mechanism instead of the hypothesized one: block0's Q/K weight norms
// grow steadily and then explosively (90.4 -> 124.0 -> inf by step
// 775), with raw pre-softmax attention scores exploding in lockstep
// (from [-11.6, 10.5] to [-21564, 37751]) - attention becoming
// increasingly SHARP/confident about a single position, not opting out
// of attending at all (the row-sum decline toward ~0.51 is the
// mathematical signature of a maximally PEAKED softmax1 distribution -
// row-sum = s/(1+s) where s is dominated by the single max term's own
// exp(0)=1 contribution once everything else decays to ~0, giving
// s->1 and row-sum->1/2 - not evidence of "nothing relevant here").
//
// Directly compared against plain softmax from IDENTICAL seeds/data
// (the one true controlled variable): plain softmax's Q/K norm and its
// own gradient stay flat/bounded for the full 2000-step run (Q+K
// gradient L2 ~1.4-3.2 even at step 1980), while softmax1's Q/K
// gradient itself grows and accelerates in lockstep with the weight
// norm (5-8 through step 300, then 12.5 -> 21.8 -> 61.8 -> 155.7 ->
// 194.9 -> NaN by step 775) - a genuine, self-reinforcing runaway
// specific to training WITH softmax1, not a pre-existing tendency
// plain softmax merely tolerates better numerically.
//
// This directly answers the original question, and the answer is NOT
// what the parked commit guessed: this is not the same failure class
// as PC-precision's single discontinuous warm-up jump. It's a
// continuous, accelerating instability present from step 0 - no smooth
// ramp or warm-up schedule would fix it, since there's no boundary
// being crossed. One clean candidate explanation for WHY softmax1
// lacks plain softmax's self-limiting saturation was checked and ruled
// out, not confirmed: the local derivative of the max-weight term
// itself, weight_i*(1-weight_i), works out identically for softmax1
// and plain softmax by direct calculation - the real differentiator
// has to be softmax1's variable (not fixed-at-1) row-sum interacting
// with the residual stream across multiple layers, a multi-layer
// effect not further decomposed here. The natural next step, not built
// in this file: does a known real fix for this class of problem
// (QK-norm - L2-normalizing Q/K before the dot product, used in
// several real transformer architectures specifically to prevent
// unbounded logit growth) actually stabilize training here.
//
// Reuses the already-tested, already-in-the-library TransformerBlock::forward_full
// hook (added in d54c8f2 specifically for this, never exercised beyond
// a smoke test) - no new engine code, just instrumentation around an
// existing capability.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};

#[path = "../common/mod.rs"]
mod common;
use common::{ForwardOut, apply_grad, encode_bytes, sample_window};

/// Same as common::forward, except every block uses forward_full with an
/// explicit use_softmax1 switch instead of plain forward() - the one
/// deliberate difference from the shared version, kept local for exactly
/// the reason tiny_lm_batched.rs's own forward stays local (a real
/// difference, not copy-paste residue). Parameterized (not hardcoded
/// true) so main() can run both conditions from the same starting
/// weights and isolate use_softmax1 as the only variable.
fn forward_variant(
    tape: &mut Tape,
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    input_ids: &[usize],
    use_softmax1: bool,
) -> (Var, ForwardOut) {
    let positions: Vec<usize> = (0..input_ids.len()).collect();
    let tok_out = token_emb.forward(tape, input_ids);
    let pos_out = pos_emb.forward(tape, &positions);
    let mut x = tape.add(tok_out.y, pos_out.y);

    let mut block_outs = Vec::with_capacity(blocks.len());
    for block in blocks {
        let out = block.forward_full(tape, x, 1, use_softmax1);
        x = out.y;
        block_outs.push(out);
    }

    let ln_out = final_ln.forward(tape, x);
    let proj_out = output_proj.forward(tape, ln_out.y);
    let logits = proj_out.y;
    (logits, ForwardOut { tok_out, pos_out, block_outs, ln_out, proj_out })
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Recomputes raw attention scores (Q@K^T / sqrt(d_k), pre-softmax1)
/// for one block from its Out struct's leafed Q/K VALUES (out.q_outs[h].y,
/// the projected per-position vectors this step actually used) - the
/// same computation TransformerBlock::forward_full does internally, but
/// forward_full doesn't expose the pre-softmax scores themselves, only
/// the post-softmax head_weights. Returns (min, max) across every head
/// and position pair.
fn raw_score_range(block_out: &scratchtape::nn::TransformerBlockOut, tape: &Tape, d_k: usize) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for (q_out, k_out) in block_out.q_outs.iter().zip(block_out.k_outs.iter()) {
        let q = tape.value(q_out.y);
        let k = tape.value(k_out.y);
        let scores = q.matmul(&k.transpose()).scale(1.0 / (d_k as f32).sqrt());
        for &s in &scores.data {
            if s < min {
                min = s;
            }
            if s > max {
                max = s;
            }
        }
    }
    (min, max)
}

/// Runs one full training condition from a fresh, identically-seeded
/// model - use_softmax1 is the only thing that differs between calls,
/// isolating it as the sole variable rather than comparing across runs
/// that also happened to get different random initializations.
fn run(label: &str, use_softmax1: bool, d_model: usize, n_heads: usize, d_ff: usize, seq_len: usize, n_blocks: usize, vocab_size: usize, corpus: &[usize], lr: f32, max_steps: usize) {
    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> = (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr };

    let d_k = d_model / n_heads;
    println!("\n=== {label} (use_softmax1={use_softmax1}) ===");
    println!(
        "columns: step | loss | mean attn row-sum per block (0..3) | block0 raw score [min,max] | block0 Q/K weight L2 norm (summed over heads)"
    );

    for step in 0..max_steps {
        let (input, target) = sample_window(&mut rng, corpus, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward_variant(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input, use_softmax1);
        let loss = tape.cross_entropy(logits, &target);
        let loss_val = tape.value(loss).data[0];
        tape.backward(loss);

        if step % 20 == 0 || loss_val.is_nan() {
            let row_sums: Vec<f32> = out
                .block_outs
                .iter()
                .map(|b| {
                    let mut total = 0.0f32;
                    let mut count = 0usize;
                    for &w in &b.head_weights {
                        let arr = tape.value(w);
                        let rows = arr.shape[0];
                        let cols = arr.shape[1];
                        for r in 0..rows {
                            total += arr.data[r * cols..(r + 1) * cols].iter().sum::<f32>();
                            count += 1;
                        }
                    }
                    total / count as f32
                })
                .collect();
            // Confirmed via an earlier run that out_proj/ffn2 weight norms
            // stay essentially flat the whole run (16.0 -> 16.07 over 775
            // steps) - refutes the "downstream compensation" hypothesis in
            // this file's header comment. Focusing instead on block0
            // specifically (the block whose row-sum visibly declines,
            // unlike blocks 1-3): raw pre-softmax1 scores, and its Q/K
            // weight norms (via out.q_outs[h].w/out.k_outs[h].w - THIS
            // step's actual leafed weight values, not gradients).
            let block0 = &out.block_outs[0];
            let (score_min, score_max) = raw_score_range(block0, &tape, d_k);
            let qk_norm: f32 = block0.q_outs.iter().chain(block0.k_outs.iter()).map(|o| l2_norm(&tape.value(o.w).data)).sum();
            // Gradient magnitude on the SAME Q/K weights, read right after
            // backward() - tests directly whether softmax1's gradient
            // pressure to keep sharpening actually fails to decay the way
            // plain softmax's does, rather than continuing to hand-derive
            // it: checked one clean candidate explanation (weight_max's
            // own saturation derivative, weight_i*(1-weight_i)) and found
            // it's IDENTICAL for softmax1 and plain softmax - ruled out,
            // not confirmed, so the real differentiator has to show up
            // empirically here or the mechanism stays only partially
            // understood.
            let qk_grad_norm: f32 = block0
                .q_outs
                .iter()
                .chain(block0.k_outs.iter())
                .filter_map(|o| tape.grad(o.w))
                .map(|g| l2_norm(&g.data))
                .sum();
            println!(
                "{step:>4} | {loss_val:.6} | row-sum: {:.4} {:.4} {:.4} {:.4} | block0 score [{:.3}, {:.3}] | block0 Q+K L2: {:.3} | block0 Q+K grad L2: {:.3}",
                row_sums.first().copied().unwrap_or(f32::NAN),
                row_sums.get(1).copied().unwrap_or(f32::NAN),
                row_sums.get(2).copied().unwrap_or(f32::NAN),
                row_sums.get(3).copied().unwrap_or(f32::NAN),
                score_min,
                score_max,
                qk_norm,
                qk_grad_norm,
            );
            if loss_val.is_nan() {
                println!("NaN at step {step} - stopping");
                break;
            }
        }

        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
}

fn main() {
    let corpus = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    // lr=0.03 from d54c8f2's own sweep: NaN'd around step 507-1600
    // depending on the exact run - late enough to see a real trend
    // develop before divergence, unlike lr=0.3's ~110-216 step blowup.
    let lr = 0.03;
    let max_steps = 2000;

    println!("softmax1 divergence diagnosis: d_model={d_model} n_heads={n_heads} n_blocks={n_blocks} lr={lr}");
    println!(
        "Decisive question: does plain softmax show the SAME Q/K weight-norm \
         growth (just tolerated numerically, since max-subtraction handles \
         arbitrarily large scores gracefully), or is unconstrained growth \
         specific to softmax1? Runs both from identical seeds/data - \
         use_softmax1 is the only variable."
    );

    run("softmax1", true, d_model, n_heads, d_ff, seq_len, n_blocks, vocab_size, &corpus, lr, max_steps);
    run("plain softmax", false, d_model, n_heads, d_ff, seq_len, n_blocks, vocab_size, &corpus, lr, max_steps);
}
