// A second consolidation mechanism to compare against replay, on the same
// diverse-corpus setup memory_tier_multigen_diverse.rs and its budget
// sweep used. No trait unifies this with replay - the two don't share a
// call shape (replay hooks into the training loop's sampling; this one
// runs only between phases and never touches training), and there's only
// one consumer of each so far. Reuses the checkpoint flatten/reconstruct
// machinery memory_tier_save.rs/memory_tier_diff.rs already built instead
// of adding a new blend() to every layer type in nn.rs.
//
// Mechanism (an EWC-lite stand-in, not real EWC - no per-parameter Fisher
// weighting, just a uniform pull-back): before each phase (from phase 1
// onward - phase 0 has nothing yet worth protecting, same reasoning
// catastrophic_forgetting.rs uses for starting the replay buffer empty),
// snapshot the flat parameter vector. Train the phase normally, no
// replay. Then blend the post-training weights back toward the
// pre-phase snapshot: new = alpha*trained + (1-alpha)*pre_phase. alpha=1
// reduces to plain sequential training (memory_tier_multigen_diverse_no_replay.rs);
// smaller alpha protects old weights harder at the direct cost of how
// much of this phase's training actually sticks.
use scratchtape::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng, TransformerBlock, TransformerBlockOut};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};
use std::time::Instant;

fn encode_bytes(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
}

fn sample_window(rng: &mut Rng, corpus: &[usize], seq_len: usize) -> (Vec<usize>, Vec<usize>) {
    let max_start = corpus.len() - seq_len - 1;
    let start = (rng.next_f32() * max_start as f32) as usize;
    (corpus[start..start + seq_len].to_vec(), corpus[start + 1..start + seq_len + 1].to_vec())
}

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

fn eval_loss(
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    corpus: &[usize],
    seq_len: usize,
) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    let mut start = 0;
    while start + seq_len + 1 <= corpus.len() {
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

fn train(
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
) {
    for _ in 0..steps {
        let (input, target) = sample_window(rng, corpus, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, token_emb, pos_emb, blocks, final_ln, output_proj, opt);
    }
}

fn flatten_all(
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
) -> Vec<f32> {
    let mut flat = token_emb.to_flat();
    flat.extend(pos_emb.to_flat());
    for block in blocks {
        flat.extend(block.to_flat());
    }
    flat.extend(final_ln.to_flat());
    flat.extend(output_proj.to_flat());
    flat
}

fn reconstruct(
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

/// new=trained weights, old=pre-phase snapshot. alpha is the fraction of
/// THIS phase's training that survives; (1-alpha) is pulled back toward
/// what the model already knew.
fn blend(new: &[f32], old: &[f32], alpha: f32) -> Vec<f32> {
    new.iter().zip(old.iter()).map(|(n, o)| alpha * n + (1.0 - alpha) * o).collect()
}

fn main() {
    // Identical corpus construction to memory_tier_multigen_diverse.rs -
    // see its doc comment for sourcing.
    let aesop_full = encode_bytes(include_str!("../data/aesops_fables.txt"));
    let encoded: Vec<Vec<usize>> = vec![
        aesop_full[..59470].to_vec(),
        encode_bytes(include_str!("../data/sherlock_holmes.txt")),
        encode_bytes(include_str!("../data/origin_of_species.txt")),
        encode_bytes(include_str!("../data/leaves_of_grass.txt")),
    ];
    let labels = ["A(fables)", "B(holmes)", "C(origin)", "D(whitman)"];
    for (label, corpus) in labels.iter().zip(encoded.iter()) {
        println!("corpus {label}: {} bytes", corpus.len());
    }

    // Same architecture and step budget as memory_tier_multigen_diverse.rs.
    let (d_model, n_heads, d_ff, seq_len, n_blocks) = (128, 8, 256, 64, 4);
    let vocab_size = 256;
    let steps_per_phase = 4000;
    let alpha = 0.7; // 70% of each phase's training survives the pull-back

    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> =
        (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    let start_time = Instant::now();

    for phase in 0..4 {
        let name = labels[phase];
        println!("\nphase {phase} ({name}): training {steps_per_phase} steps ({:.1}s elapsed)", start_time.elapsed().as_secs_f32());

        let pre_phase_flat = flatten_all(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj);
        train(&mut rng, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &encoded[phase], seq_len, steps_per_phase, &opt);

        if phase > 0 {
            let post_flat = flatten_all(&token_emb, &pos_emb, &blocks, &final_ln, &output_proj);
            let blended = blend(&post_flat, &pre_phase_flat, alpha);
            let (t, p, b, l, o) = reconstruct(&blended, vocab_size, d_model, seq_len, n_blocks, n_heads, d_ff);
            token_emb = t;
            pos_emb = p;
            blocks = b;
            final_ln = l;
            output_proj = o;
            println!("  consolidated: blended {:.0}% this phase's training with the pre-phase snapshot", alpha * 100.0);
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
