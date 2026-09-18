use engine::nn::{Embedding, EmbeddingOut, LayerNorm, LayerNormOut, Linear, LinearOut, Rng, TransformerBlock, TransformerBlockOut};
use engine::optim::Sgd;
use engine::tape::{Tape, Var};
use std::thread;
use std::time::Instant;

fn encode_bytes(text: &str) -> Vec<usize> {
    text.bytes().map(|b| b as usize).collect()
}

fn decode_bytes(ids: &[usize]) -> String {
    let bytes: Vec<u8> = ids.iter().map(|&i| i as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn sample_window(rng: &mut Rng, corpus: &[usize], seq_len: usize) -> (Vec<usize>, Vec<usize>) {
    let max_start = corpus.len() - seq_len - 1;
    let start = (rng.next_f32() * max_start as f32) as usize;
    (corpus[start..start + seq_len].to_vec(), corpus[start + 1..start + seq_len + 1].to_vec())
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

/// Mean cross-entropy over `n_windows` random windows, no gradient update -
/// pure evaluation. Called on the held-out region to measure generalization
/// (loss on text the model never trained on), and separately on the train
/// region for a directly comparable in-sample number.
fn eval_loss(rng: &mut Rng, token_emb: &Embedding, pos_emb: &Embedding, blocks: &[TransformerBlock], final_ln: &LayerNorm, output_proj: &Linear, corpus: &[usize], seq_len: usize, n_windows: usize) -> f32 {
    let mut total = 0.0;
    for _ in 0..n_windows {
        let (input, target) = sample_window(rng, corpus, seq_len);
        let mut tape = Tape::new();
        let (logits, _) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &input);
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
/// grad_accum, no periodic eval, no generation) - only the trained token
/// embedding table is needed from these runs, used purely to re-derive
/// `assignments` at a different init/sampling seed for the cross-seed
/// stability check below.
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
) -> Embedding {
    let mut rng = Rng::new(seed);
    let mut token_emb = Embedding::new(&mut rng, vocab_size, d_model);
    let mut pos_emb = Embedding::new(&mut rng, seq_len, d_model);
    let mut blocks: Vec<TransformerBlock> =
        (0..n_blocks).map(|_| TransformerBlock::new(&mut rng, d_model, n_heads, d_ff)).collect();
    let mut final_ln = LayerNorm::new(d_model);
    let mut output_proj = Linear::new(&mut rng, d_model, vocab_size);
    let opt = Sgd { lr: 0.3 };

    for _ in 0..=steps {
        let (input, target) = sample_window(&mut rng, train, seq_len);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
    }
    token_emb
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
) -> (Embedding, Vec<f32>, Vec<String>) {
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

    (token_emb, grad_accum, log)
}

fn main() {
    // Real, diverse public-domain prose (not a repeated stress-test corpus
    // like tiny_lm_scaled.rs) - Project Gutenberg ebook #21, "Three Hundred
    // Aesop's Fables" (George Fyler Townsend translation), Gutenberg
    // boilerplate header/footer and the trailing alphabetical index
    // stripped, fetched verbatim rather than reproduced from memory (same
    // accuracy-risk reasoning as every other corpus choice in this project).
    let full_text = include_str!("../data/aesops_fables.txt");
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
    let (primary, seed2_emb, seed3_emb) = thread::scope(|scope| {
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

    let (token_emb, grad_accum, primary_log) = primary;
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
        filtered_bytes.iter().map(|&b| token_emb.table.data[b * d_model..b * d_model + d_model].to_vec()).collect();

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
    for emb in [&seed2_emb, &seed3_emb] {
        let rows: Vec<Vec<f32>> =
            filtered_bytes.iter().map(|&b| emb.table.data[b * d_model..b * d_model + d_model].to_vec()).collect();
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
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let agree_count = (0..n_runs).filter(|&r| same_cluster(r, i, j)).count();
            unanimous[i][j] = agree_count == 0 || agree_count == n_runs;
            majority_same[i][j] = agree_count * 2 > n_runs;
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
                if majority_same[i][j] && component[j] == usize::MAX {
                    component[j] = next_component;
                    stack.push(j);
                }
            }
        }
        next_component += 1;
    }
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
}
