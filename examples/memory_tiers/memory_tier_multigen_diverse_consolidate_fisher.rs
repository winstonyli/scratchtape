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
// EWC would need a gradient for every individual weight. This gets
// real per-parameter Fisher for token_emb, pos_emb, final_ln and
// output_proj - their *Out structs expose the actual parameter Vars
// (table/w/b/gamma/beta are all `pub`). TransformerBlock's internals
// (q/k/v heads, ffn, its own layernorms) are private - TransformerBlockOut
// only exposes `y` and `head_weights`, not the sub-layer Vars - so the 4
// blocks fall back to the old fixed-alpha uniform blend. A real,
// meaningful partial upgrade (4 of the model's parameter groups now get
// genuine importance weighting instead of none), not a claim of full EWC.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::Tape;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{apply_grad, encode_bytes, eval_loss, forward, sample_window};

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
        fisher.steps += 1;

        apply_grad(&tape, &out, token_emb, pos_emb, blocks, final_ln, output_proj, opt);
    }
}

/// Running sum of squared gradients per named segment, plus the element
/// count each sum was accumulated over - together these give the
/// empirical Fisher (average squared gradient per parameter) once a
/// phase's training is done.
struct Fisher {
    sums: std::collections::HashMap<&'static str, f32>,
    counts: std::collections::HashMap<&'static str, usize>,
    steps: usize,
}

impl Fisher {
    fn new() -> Self {
        Self { sums: std::collections::HashMap::new(), counts: std::collections::HashMap::new(), steps: 0 }
    }

    fn accumulate(&mut self, name: &'static str, grad: Option<&scratchtape::tensor::NdArray>) {
        let Some(g) = grad else { return };
        let sq_sum: f32 = g.data.iter().map(|x| x * x).sum();
        *self.sums.entry(name).or_insert(0.0) += sq_sum;
        *self.counts.entry(name).or_insert(0) += g.data.len();
    }

    /// Average squared gradient per parameter per step - the empirical
    /// Fisher estimate for this segment.
    fn average(&self, name: &str) -> f32 {
        let sum = *self.sums.get(name).unwrap_or(&0.0);
        let count = *self.counts.get(name).unwrap_or(&1) as f32;
        sum / count / self.steps.max(1) as f32
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

fn flatten_all(token_emb: &Embedding, pos_emb: &Embedding, blocks: &[TransformerBlock], final_ln: &LayerNorm, output_proj: &Linear) -> Vec<f32> {
    let mut flat = token_emb.to_flat();
    flat.extend(pos_emb.to_flat());
    for block in blocks {
        flat.extend(block.to_flat());
    }
    flat.extend(final_ln.to_flat());
    flat.extend(output_proj.to_flat());
    flat
}

fn reconstruct(flat: &[f32], vocab_size: usize, d_model: usize, seq_len: usize, n_blocks: usize, n_heads: usize, d_ff: usize) -> (Embedding, Embedding, Vec<TransformerBlock>, LayerNorm, Linear) {
    let mut offset = 0usize;
    let token_emb = Embedding::from_flat(flat, &mut offset, vocab_size, d_model);
    let pos_emb = Embedding::from_flat(flat, &mut offset, seq_len, d_model);
    let blocks = (0..n_blocks).map(|_| TransformerBlock::from_flat(flat, &mut offset, d_model, n_heads, d_ff)).collect();
    let final_ln = LayerNorm::from_flat(flat, &mut offset, d_model);
    let output_proj = Linear::from_flat(flat, &mut offset, d_model, vocab_size);
    assert_eq!(offset, flat.len(), "flat vector had leftover/missing floats - architecture mismatch");
    (token_emb, pos_emb, blocks, final_ln, output_proj)
}

/// Finds each named segment's (start, end) byte-offset range within the
/// flat vector, by running the same from_flat sequence reconstruct()
/// uses and recording each call's consumed range - same technique
/// memory_tier_diff.rs already established, reused here instead of
/// hand-deriving each struct's flat length.
fn segment_ranges(vocab_size: usize, d_model: usize, seq_len: usize, n_blocks: usize, n_heads: usize, d_ff: usize, total_len: usize) -> Vec<(&'static str, usize, usize)> {
    let dummy = vec![0.0f32; total_len];
    let mut offset = 0usize;
    let mut ranges = Vec::new();

    let start = offset;
    let _ = Embedding::from_flat(&dummy, &mut offset, vocab_size, d_model);
    ranges.push(("token_emb", start, offset));

    let start = offset;
    let _ = Embedding::from_flat(&dummy, &mut offset, seq_len, d_model);
    ranges.push(("pos_emb", start, offset));

    for _ in 0..n_blocks {
        let start = offset;
        let _ = TransformerBlock::from_flat(&dummy, &mut offset, d_model, n_heads, d_ff);
        ranges.push(("blocks", start, offset));
    }

    let start = offset;
    let _ = LayerNorm::from_flat(&dummy, &mut offset, d_model);
    ranges.push(("final_ln", start, offset));

    let start = offset;
    let _ = Linear::from_flat(&dummy, &mut offset, d_model, vocab_size);
    ranges.push(("output_proj", start, offset));

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
    let blocks_alpha = 0.7; // unchanged fallback for the 4 opaque block segments
    // First attempt at 5000.0 gave alpha~1.000 for every Fisher-weighted
    // segment (Fisher magnitudes here are ~1e-9 to ~1e-7, so lambda*fisher
    // was negligible - effectively no protection at all, silently
    // reducing to the no-replay baseline for 4 of 5 segment groups).
    // Retuned from that real measurement: 1e7 puts final_ln (the largest
    // Fisher, ~1e-7) around alpha~0.5-0.6 - real, comparable-magnitude
    // protection to the blocks' fixed 0.7 - while token_emb/pos_emb/
    // output_proj (Fisher ~1e-9, two orders of magnitude smaller) stay
    // around alpha~0.96-0.99, correctly reflecting that per-parameter
    // sensitivity is diluted across a much larger table.
    let lambda = 1.0e7;

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
            for &(seg_name, start, end) in &ranges {
                let alpha = if seg_name == "blocks" { blocks_alpha } else { fisher_to_alpha(fisher.average(seg_name), lambda) };
                for i in start..end {
                    blended[i] = alpha * post_flat[i] + (1.0 - alpha) * pre_phase_flat[i];
                }
            }
            let (t, p, b, l, o) = reconstruct(&blended, vocab_size, d_model, seq_len, n_blocks, n_heads, d_ff);
            token_emb = t;
            pos_emb = p;
            blocks = b;
            final_ln = l;
            output_proj = o;

            println!(
                "  consolidated - Fisher (avg sq grad/param): token_emb={:.2e} pos_emb={:.2e} final_ln={:.2e} output_proj={:.2e}",
                fisher.average("token_emb"),
                fisher.average("pos_emb"),
                fisher.average("final_ln"),
                fisher.average("output_proj"),
            );
            println!(
                "  alpha used: token_emb={:.3} pos_emb={:.3} final_ln={:.3} output_proj={:.3} blocks={:.3} (fixed)",
                fisher_to_alpha(fisher.average("token_emb"), lambda),
                fisher_to_alpha(fisher.average("pos_emb"), lambda),
                fisher_to_alpha(fisher.average("final_ln"), lambda),
                fisher_to_alpha(fisher.average("output_proj"), lambda),
                blocks_alpha,
            );
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
