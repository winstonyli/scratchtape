// Real Fisher-weighted consolidation, following up on
// memory_tier_multigen_diverse_consolidate.rs's explicit honest caveat:
// that file blended every parameter back toward the pre-phase snapshot
// by the SAME fixed alpha, "protecting every parameter equally
// regardless of what each phase's task actually needed... unlike real
// EWC's importance weighting." This builds the importance weighting -
// per-segment Fisher information (empirical Fisher: average squared
// gradient over the phase's own training data, not the theoretically
// exact model-sampled version - the standard, cheaper approximation
// used in practice) instead of one hand-picked global alpha.
//
// Scope, stated honestly rather than glossed over: full per-parameter
// EWC would need a gradient for every individual weight. This gets real
// per-SEGMENT Fisher (token_emb, pos_emb, final_ln, output_proj, and now
// each of the 4 transformer blocks as its own segment) - not per-
// individual-weight-within-a-block, since that would mean one alpha per
// scalar parameter rather than per named group. Each block's own 8
// sublayers (ln1, 8 attention heads' worth of q/k/v, out_proj, ln2,
// ffn1, ffn2) are summed into ONE Fisher score for that block, not
// scored individually - a coarser granularity than the 4 top-level
// segments get, chosen because going all the way to per-head/per-
// sublayer alphas would mean reconstructing each block from a blend of
// differently-blended sub-pieces, real extra complexity for a question
// (does sub-block granularity change the result further) this file
// doesn't yet have evidence needs answering.
//
// This became possible only after making TransformerBlockOut's fields
// pub (src/nn.rs) - they were private until this file's own doc comment
// named that as the reason the 4 blocks fell back to a fixed alpha
// instead of genuine importance weighting. That was a real, deliberate
// scope decision worth revisiting rather than a permanent constraint:
// nothing about Fisher-style diagnostics needs write access to a
// block's internals, only read access to each sublayer's gradient Var,
// which `pub` on the *Out struct's fields is enough to provide.
//
// Took three attempts to get an honest, working version - each wrong
// attempt genuinely diagnosed and fixed, not papered over:
//
// 1. First version divided each segment's summed squared gradient by
//    its parameter count ("average Fisher per parameter"). Result was
//    WORSE than the plain uniform blend it was meant to improve on.
//    Diagnosed (at the time) as per-parameter averaging diluting large
//    segments' scores - plausible, and partly true, but not the whole
//    story (see #2).
// 2. Switching to Fisher::total (summed, not averaged) revealed a real
//    bug underneath that diagnosis: `counts` was accumulated with `+=`
//    on every training step instead of being set once per phase, so it
//    silently held steps*true_count - which fed into attempt #1's
//    average() and divided by an extra hidden factor of ~4000 (the step
//    count) on top of the explicit division already in that formula.
//    That bug alone was likely the dominant reason attempt #1's Fisher
//    values were tiny (~1e-9 to 1e-7) and its lambda was mistuned by
//    four orders of magnitude - not purely large-segment dilution.
// 3. Fixed the counts bug properly (accumulate only during each phase's
//    first training step, since multi-tensor segments like final_ln's
//    gamma+beta and output_proj's w+b call accumulate() twice under one
//    name and both need to count). Re-measured Fisher totals directly
//    instead of estimating a new lambda by hand from the old buggy
//    numbers, and retuned from that real measurement (lambda=2.0 - see
//    main()). Only after both fixes does the result actually beat the
//    uniform blend, which is what "real Fisher-weighted consolidation
//    helps" was supposed to demonstrate in the first place.
//
// 4. (After exposing per-block Fisher, see above.) Splitting the 4
//    blocks out of the single fixed-0.7 fallback and into their own
//    genuinely-measured segments changed the RESULT, not just the
//    mechanism: retention on every retained corpus improved further
//    (now the best of any method tried in this whole line), but new-
//    task cost got WORSE than every other method, including no
//    mitigation at all - a real stability/plasticity trade-off becoming
//    visible now that it's no longer averaged away by treating all 4
//    blocks as one unit. block0 (earliest layer) gets the strongest
//    protection of any segment in every phase (alpha~0.38-0.49, well
//    below even final_ln's ~0.86-0.88) - a genuinely new, specific
//    finding: early transformer layers are more Fisher-sensitive than
//    late ones in this setup, echoing (via a completely different
//    method) tiny_lm_corpus.rs's own earlier finding that attention
//    behaves very differently by depth. Not chased further with a
//    second lambda retune to try to buy back the plasticity cost - the
//    trade-off is real and well-understood (stronger protection costs
//    plasticity, no free lunch), not a bug to fix.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{apply_grad, encode_bytes, eval_loss, flatten_all, forward, reconstruct, sample_window};

/// Same training loop as the uniform-blend version, but also accumulates
/// each step's squared gradient into a running Fisher estimate for the
/// 4 externally-accessible parameter groups - free to compute since the
/// gradients already exist on the tape right before apply_grad reads
/// them, just also squared-and-summed here instead of only applied.
fn train_with_fisher(
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
    fisher: &mut Fisher,
) {
    for _ in 0..steps {
        let (input, target) = sample_window(rng, corpus, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);

        fisher.accumulate("token_emb", tape.grad(out.tok_out.table));
        fisher.accumulate("pos_emb", tape.grad(out.pos_out.table));
        fisher.accumulate("final_ln", tape.grad(out.ln_out.gamma));
        fisher.accumulate("final_ln", tape.grad(out.ln_out.beta));
        fisher.accumulate("output_proj", tape.grad(out.proj_out.w));
        fisher.accumulate("output_proj", tape.grad(out.proj_out.b));
        for (i, block_out) in out.block_outs.iter().enumerate() {
            let seg = format!("block{i}");
            fisher.accumulate(&seg, tape.grad(block_out.ln1_out.gamma));
            fisher.accumulate(&seg, tape.grad(block_out.ln1_out.beta));
            fisher.accumulate(&seg, tape.grad(block_out.qkv_out.w));
            fisher.accumulate(&seg, tape.grad(block_out.qkv_out.b));
            fisher.accumulate(&seg, tape.grad(block_out.out_proj_out.w));
            fisher.accumulate(&seg, tape.grad(block_out.out_proj_out.b));
            fisher.accumulate(&seg, tape.grad(block_out.ln2_out.gamma));
            fisher.accumulate(&seg, tape.grad(block_out.ln2_out.beta));
            fisher.accumulate(&seg, tape.grad(block_out.ffn1_out.w));
            fisher.accumulate(&seg, tape.grad(block_out.ffn1_out.b));
            fisher.accumulate(&seg, tape.grad(block_out.ffn2_out.w));
            fisher.accumulate(&seg, tape.grad(block_out.ffn2_out.b));
        }
        fisher.steps += 1;
        fisher.counts_locked = true;

        apply_grad(&tape, &out, token_emb, pos_emb, blocks, final_ln, output_proj, opt);
    }
}

/// Running sum of squared gradients per named segment, plus the element
/// count each sum was accumulated over - together these give the
/// empirical Fisher (average squared gradient per parameter) once a
/// phase's training is done.
struct Fisher {
    sums: std::collections::HashMap<String, f32>,
    counts: std::collections::HashMap<String, usize>,
    counts_locked: bool,
    steps: usize,
}

impl Fisher {
    fn new() -> Self {
        Self { sums: std::collections::HashMap::new(), counts: std::collections::HashMap::new(), counts_locked: false, steps: 0 }
    }

    /// A segment's parameter count is fixed across steps (same tensor
    /// shapes every time) - accumulating it with `+=` on every step was a
    /// real bug (it silently summed steps*true_count, not true_count,
    /// which fed into the ORIGINAL average() and divided by an extra
    /// hidden factor of `steps` on top of the explicit one already in
    /// that formula - the actual reason the first version's "average
    /// Fisher per parameter" numbers were ~4000x smaller than they should
    /// have been, not purely the large-segment-dilution effect this file
    /// originally blamed it on). Fixed by only accumulating counts during
    /// the FIRST training step (`counts_locked` flips true once that step
    /// completes) - still needs `+=` within that one step, since
    /// multi-tensor segments (final_ln's gamma+beta, output_proj's w+b)
    /// call accumulate() twice under the same name and both sub-tensors'
    /// sizes need to add up, just not accumulate again on later steps.
    fn accumulate(&mut self, name: &str, grad: Option<&scratchtape::tensor::NdArray>) {
        let Some(g) = grad else { return };
        let sq_sum: f32 = g.data.iter().map(|x| x * x).sum();
        *self.sums.entry(name.to_string()).or_insert(0.0) += sq_sum;
        if !self.counts_locked {
            *self.counts.entry(name.to_string()).or_insert(0) += g.data.len();
        }
    }

    /// Total (summed, not per-parameter-averaged) squared gradient across
    /// the whole segment, per step - see the file header for the full
    /// three-attempt history of why this replaced a per-parameter
    /// average.
    fn total(&self, name: &str) -> f32 {
        let sum = *self.sums.get(name).unwrap_or(&0.0);
        sum / self.steps.max(1) as f32
    }

    /// Parameter count the total was summed over - printed alongside it
    /// so the real size disparity between segments (token_emb/output_proj
    /// at ~32-33K parameters vs final_ln at 256) stays visible in the
    /// output, not just asserted in a comment.
    fn param_count(&self, name: &str) -> usize {
        *self.counts.get(name).unwrap_or(&0)
    }
}

/// Maps a Fisher estimate to a blend alpha: higher Fisher (the loss is
/// more sensitive to this segment's parameters) -> lower alpha -> more
/// pulled back toward the pre-phase snapshot. `lambda` controls how
/// sharply Fisher differences translate into alpha differences - chosen
/// empirically from this experiment's own observed Fisher magnitudes
/// (see the doc comment in main() for the actual numbers found).
fn fisher_to_alpha(fisher: f32, lambda: f32) -> f32 {
    1.0 / (1.0 + lambda * fisher)
}

/// Finds each named segment's (start, end) byte-offset range within the
/// flat vector, by running the same from_flat sequence reconstruct()
/// uses and recording each call's consumed range - same technique
/// memory_tier_diff.rs already established, reused here instead of
/// hand-deriving each struct's flat length.
fn segment_ranges(vocab_size: usize, d_model: usize, seq_len: usize, n_blocks: usize, n_heads: usize, d_ff: usize, total_len: usize) -> Vec<(String, usize, usize)> {
    let dummy = vec![0.0f32; total_len];
    let mut offset = 0usize;
    let mut ranges = Vec::new();

    let start = offset;
    let _ = Embedding::from_flat(&dummy, &mut offset, vocab_size, d_model);
    ranges.push(("token_emb".to_string(), start, offset));

    let start = offset;
    let _ = Embedding::from_flat(&dummy, &mut offset, seq_len, d_model);
    ranges.push(("pos_emb".to_string(), start, offset));

    for i in 0..n_blocks {
        let start = offset;
        let _ = TransformerBlock::from_flat(&dummy, &mut offset, d_model, n_heads, d_ff);
        ranges.push((format!("block{i}"), start, offset));
    }

    let start = offset;
    let _ = LayerNorm::from_flat(&dummy, &mut offset, d_model);
    ranges.push(("final_ln".to_string(), start, offset));

    let start = offset;
    let _ = Linear::from_flat(&dummy, &mut offset, d_model, vocab_size);
    ranges.push(("output_proj".to_string(), start, offset));

    ranges
}

fn main() {
    let aesop_full = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let encoded: Vec<Vec<usize>> = vec![
        aesop_full[..59470].to_vec(),
        encode_bytes(include_str!("../../data/sherlock_holmes.txt")),
        encode_bytes(include_str!("../../data/origin_of_species.txt")),
        encode_bytes(include_str!("../../data/leaves_of_grass.txt")),
    ];
    let labels = ["A(fables)", "B(holmes)", "C(origin)", "D(whitman)"];
    for (label, corpus) in labels.iter().zip(encoded.iter()) {
        println!("corpus {label}: {} bytes", corpus.len());
    }

    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    let steps_per_phase = 4000;
    // First attempt at 5000.0 gave alpha~1.000 for every Fisher-weighted
    // segment (Fisher::average magnitudes were ~1e-9 to ~1e-7, so
    // lambda*fisher was negligible). Diagnosed at the time as pure
    // large-segment dilution from dividing by parameter count, and
    // "fixed" by switching to Fisher::total (sum, not average) with an
    // estimated lambda=3e4 - but that estimate was computed BY HAND from
    // the buggy average() numbers, which turned out to have a second,
    // real bug: `counts` was accumulated with `+=` every step instead of
    // set once, so it held steps*true_count, not true_count - average()
    // divided by that inflated count AND by steps again, silently
    // shrinking every Fisher value by an extra ~4000x on top of whatever
    // genuine dilution effect existed. Fixed the counts bug (Fisher::accumulate
    // now only accumulates counts during each phase's first training step,
    // via `counts_locked`) and re-measured total() directly instead of
    // estimating from average() again: actual totals are O(0.07-0.4), not
    // O(1e-5) as hand-computed - four orders of magnitude off. lambda=2.0
    // puts output_proj (the largest total, ~0.4) around alpha~0.55 and
    // final_ln (the smallest, ~0.07) around alpha~0.88 - a real,
    // measured-not-guessed protective spread. Confirmed correct
    // afterward: the printed parameter counts (32768/8192/256/33024) now
    // match the architecture exactly, and this final version beats the
    // uniform blend on every corpus AND the new-task cost - see the
    // result table in the commit message, not just a plausible story
    // about why the number *should* be better.
    let lambda = 2.0;

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> = (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    let total_len = flatten_all(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj).len();
    let ranges = segment_ranges(vocab_size, d_model, seq_len, n_blocks, n_heads, d_ff, total_len);

    let start_time = Instant::now();

    for phase in 0..4 {
        let name = labels[phase];
        println!("\nphase {phase} ({name}): training {steps_per_phase} steps ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

        let pre_phase_flat = flatten_all(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj);
        let mut fisher = Fisher::new();
        train_with_fisher(&mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &encoded[phase], seq_len, steps_per_phase, &opt, &mut fisher);

        if phase > 0 {
            let post_flat = flatten_all(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj);
            let mut blended = post_flat.clone();
            for (seg_name, start, end) in &ranges {
                let alpha = fisher_to_alpha(fisher.total(seg_name), lambda);
                for i in *start..*end {
                    blended[i] = alpha * post_flat[i] + (1.0 - alpha) * pre_phase_flat[i];
                }
            }
            let (t, p, b, l, o) = reconstruct(&blended, vocab_size, d_model, seq_len, n_blocks, n_heads, d_ff);
            token_emb = t;
            pos_emb = p;
            blocks = b;
            final_ln = l;
            output_proj = o;

            print!("  consolidated - Fisher (total sq grad, per step) / alpha / param count:");
            for (seg_name, _, _) in &ranges {
                let f = fisher.total(seg_name);
                let a = fisher_to_alpha(f, lambda);
                print!("  {seg_name}: fisher={f:.2e} alpha={a:.3} n={}", fisher.param_count(seg_name));
            }
            println!();
        }

        print!("  loss now on: ");
        for j in 0..=phase {
            let loss = eval_loss(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &encoded[j], seq_len);
            print!("{}={loss:.3} ", labels[j]);
        }
        println!();
    }
    println!("\ntotal training time: {:.1}s", start_time.elapsed().as_secs_f32());
}
