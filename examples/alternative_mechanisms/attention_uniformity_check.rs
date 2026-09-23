// Follow-up to attention_null_baseline.rs: that file showed the per-head
// "top sink target" metric reproduces from EXACTLY uniform attention, so
// the reported '\u{bc}' sink under softmax1+QK-norm was a measurement
// artifact. What it could not say is how far the TRAINED model's attention
// actually is from uniform, and how much of each head's argmax is just the
// null's. This trains the tiny_lm_corpus.rs seed-1 model under both
// conditions at the same 64000 steps - softmax1+QK-norm (the promoted
// mechanism) and plain softmax (the control) - saves each checkpoint so
// later analysis needs no retrain (a checkpoint file present = skip
// training), then measures per (block, head), over the same fixed windows
// the sink metric uses (window_seed 777):
//   - KL(row / row-mass || uniform over visible keys), mean over rows -
//     0 means exactly uniform.
//   - peak factor: max normalized weight x visible-key count - 1.0 is
//     uniform, larger is peakier. QK-norm caps it near e^0.5 =~ 1.65 per
//     row by construction (scores are +/-0.25).
//   - row mass: sum of weights (softmax1 lets it fall below 1).
//   - argmax agreement with the null: fraction of query bytes whose top
//     key under this head equals the top key under uniform attention on
//     the same windows.
// Training reproduces train_with_diagnostics' rng consumption exactly
// (including the eval_loss draws every 1600 steps) so the seed-1 models
// match tiny_lm_corpus.rs's.
//
// Result (32-head means): softmax1+QK-norm KL=0.0043, peak=1.16, mass=0.94,
// null-argmax agreement 0.93 - effectively a causal mean-pool; its sink
// tables are the null's. Plain softmax KL=2.24, peak=21.4, agreement 0.36 -
// genuinely selective, so its per-head sinks are mostly real content.
// Held-out loss at the same 64000 steps: 2.026 (softmax1+QK-norm) vs 1.738
// (plain), plain lower at all 40 evals after init. Root cause: QK-norm here
// has no learnable scale, so logits are capped at +/-0.25; the published
// form (Henry et al. 2020) learns that scale.
use scratchtape::nn::{Embedding, LayerNorm, Linear, Rng, TransformerBlock};
use scratchtape::optim::Sgd;
use scratchtape::tape::{Tape, Var};
use std::collections::HashMap;
use std::thread;
use std::time::Instant;

#[path = "../common/mod.rs"]
mod common;
use common::{ForwardOut, apply_grad, encode_bytes, flatten_all, reconstruct, sample_window};

const D_MODEL: usize = 128;
const N_HEADS: usize = 8;
const D_FF: usize = 256;
const SEQ_LEN: usize = 64;
const N_BLOCKS: usize = 4;
const VOCAB: usize = 256;

struct Model {
    token_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    final_ln: LayerNorm,
    output_proj: Linear,
}

fn forward(
    tape: &mut Tape,
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    input_ids: &[usize],
    softmax1: bool,
    qknorm: bool,
) -> (Var, ForwardOut) {
    let positions: Vec<usize> = (0..input_ids.len()).collect();
    let tok_out = token_emb.forward(tape, input_ids);
    let pos_out = pos_emb.forward(tape, &positions);
    let mut x = tape.add(tok_out.y, pos_out.y);
    let mut block_outs = Vec::with_capacity(blocks.len());
    for block in blocks {
        let out = block.forward_full(tape, x, 1, softmax1, qknorm);
        x = out.y;
        block_outs.push(out);
    }
    let ln_out = final_ln.forward(tape, x);
    let proj_out = output_proj.forward(tape, ln_out.y);
    (proj_out.y, ForwardOut { tok_out, pos_out, block_outs, ln_out, proj_out })
}

fn eval_loss(
    rng: &mut Rng,
    token_emb: &Embedding,
    pos_emb: &Embedding,
    blocks: &[TransformerBlock],
    final_ln: &LayerNorm,
    output_proj: &Linear,
    corpus: &[usize],
    softmax1: bool,
    qknorm: bool,
) -> f32 {
    let mut total = 0.0;
    for _ in 0..20 {
        let (input, target) = sample_window(rng, corpus, SEQ_LEN);
        let mut tape = Tape::new();
        let (logits, _) = forward(&mut tape, token_emb, pos_emb, blocks, final_ln, output_proj, &input, softmax1, qknorm);
        let loss = tape.cross_entropy(logits, &target);
        total += tape.value(loss).data[0];
    }
    total / 20.0
}

/// Mirrors train_with_diagnostics (seed 1), minus generation. Also returns
/// grad_accum, which the noise filter needs to reproduce filtered_bytes.
fn train(train_set: &[usize], held_out: &[usize], steps: usize, softmax1: bool, qknorm: bool, log: &mut Vec<String>) -> (Model, Vec<f32>) {
    let mut rng = Rng::new(1);
    let mut token_emb = Embedding::new(&mut rng, VOCAB, D_MODEL);
    let mut pos_emb = Embedding::new(&mut rng, SEQ_LEN, D_MODEL);
    let mut blocks: Vec<TransformerBlock> = (0..N_BLOCKS).map(|_| TransformerBlock::new(&mut rng, D_MODEL, N_HEADS, D_FF)).collect();
    let mut final_ln = LayerNorm::new(D_MODEL);
    let mut output_proj = Linear::new(&mut rng, D_MODEL, VOCAB);
    let opt = Sgd { lr: 0.3 };
    let mut grad_accum = vec![0.0f32; VOCAB];
    let start = Instant::now();
    for step in 0..=steps {
        let (input, target) = sample_window(&mut rng, train_set, SEQ_LEN);
        let mut tape = Tape::with_capacity(2000);
        let (logits, out) = forward(&mut tape, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, &input, softmax1, qknorm);
        let loss = tape.cross_entropy(logits, &target);
        tape.backward(loss);
        let tok_grad = tape.grad(out.tok_out.table).unwrap();
        for b in 0..VOCAB {
            grad_accum[b] += tok_grad.data[b * D_MODEL..(b + 1) * D_MODEL].iter().map(|g| g.abs()).sum::<f32>();
        }
        apply_grad(&tape, &out, &mut token_emb, &mut pos_emb, &mut blocks, &mut final_ln, &mut output_proj, &opt);
        if step % 1600 == 0 {
            let tr = eval_loss(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, train_set, softmax1, qknorm);
            let ho = eval_loss(&mut rng, &token_emb, &pos_emb, &blocks, &final_ln, &output_proj, held_out, softmax1, qknorm);
            log.push(format!("step {step:>5}: eval train = {tr:.4}, eval held-out = {ho:.4} ({:.1}s)", start.elapsed().as_secs_f32()));
        }
    }
    (Model { token_emb, pos_emb, blocks, final_ln, output_proj }, grad_accum)
}

fn save(path: &str, m: &Model, grad_accum: &[f32]) {
    let flat = flatten_all(&m.token_emb, &m.pos_emb, &m.blocks, &m.final_ln, &m.output_proj);
    let join = |v: &[f32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
    std::fs::write(path, format!("{}\n{}\n", join(grad_accum), join(&flat))).unwrap();
}

fn load(path: &str) -> Option<(Model, Vec<f32>)> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    let parse = |l: &str| l.split_whitespace().map(|x| x.parse::<f32>().unwrap()).collect::<Vec<f32>>();
    let grad_accum = parse(lines.next()?);
    let flat = parse(lines.next()?);
    let (token_emb, pos_emb, blocks, final_ln, output_proj) = reconstruct(&flat, VOCAB, D_MODEL, SEQ_LEN, N_BLOCKS, N_HEADS, D_FF);
    Some((Model { token_emb, pos_emb, blocks, final_ln, output_proj }, grad_accum))
}

/// Per query byte, the key byte with the highest mean weight (None if the
/// query byte never appeared) - same reduction as tiny_lm_corpus.rs's
/// attention_graph_all_heads + per-head sink summary.
fn argmax_targets(sum: &[f32], count: &[u32], n: usize) -> Vec<Option<usize>> {
    (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| count[i * n + j] > 0)
                .max_by(|&a, &b| {
                    let ma = sum[i * n + a] / count[i * n + a] as f32;
                    let mb = sum[i * n + b] / count[i * n + b] as f32;
                    ma.partial_cmp(&mb).unwrap()
                })
        })
        .collect()
}

fn analyze(name: &str, m: &Model, train_set: &[usize], filtered: &[usize], softmax1: bool, qknorm: bool) -> Vec<String> {
    let byte_index: HashMap<usize, usize> = filtered.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let n = filtered.len();
    let heads = N_BLOCKS * N_HEADS;
    let mut wsum = vec![0.0f32; heads * n * n];
    let mut count = vec![0u32; n * n];
    let mut null_sum = vec![0.0f32; n * n];
    let null_plus = if softmax1 { 2.0 } else { 1.0 };
    let (mut kl, mut peak, mut mass) = (vec![0.0f64; heads], vec![0.0f64; heads], vec![0.0f64; heads]);
    let mut rows = 0u64;
    let mut rng = Rng::new(777);
    let mut used = 0;
    let windows: usize = std::env::var("WINDOWS").ok().and_then(|v| v.parse().ok()).unwrap_or(20000);
    for _ in 0..windows {
        let (window, _) = sample_window(&mut rng, train_set, SEQ_LEN);
        if window.iter().any(|b| !byte_index.contains_key(b)) {
            continue;
        }
        used += 1;
        let idxs: Vec<usize> = window.iter().map(|b| byte_index[b]).collect();
        let mut tape = Tape::new();
        let (_, out) = forward(&mut tape, &m.token_emb, &m.pos_emb, &m.blocks, &m.final_ln, &m.output_proj, &window, softmax1, qknorm);
        for qi in 0..SEQ_LEN {
            for ki in 0..SEQ_LEN {
                let cell = idxs[qi] * n + idxs[ki];
                count[cell] += 1;
                if ki <= qi {
                    null_sum[cell] += 1.0 / (qi as f32 + null_plus);
                }
            }
        }
        for b in 0..N_BLOCKS {
            for h in 0..N_HEADS {
                let hid = b * N_HEADS + h;
                let w = &tape.value(out.block_outs[b].head_weights[h]).data;
                for qi in 0..SEQ_LEN {
                    for ki in 0..SEQ_LEN {
                        wsum[hid * n * n + idxs[qi] * n + idxs[ki]] += w[qi * SEQ_LEN + ki];
                    }
                }
                for qi in 1..SEQ_LEN {
                    let row = &w[qi * SEQ_LEN..qi * SEQ_LEN + qi + 1];
                    let row_mass: f32 = row.iter().sum();
                    let visible = (qi + 1) as f32;
                    let (mut row_kl, mut row_max) = (0.0f32, 0.0f32);
                    for &x in row {
                        let p = x / row_mass;
                        if p > 0.0 {
                            row_kl += p * (p * visible).ln();
                        }
                        row_max = row_max.max(p);
                    }
                    kl[hid] += row_kl as f64;
                    peak[hid] += (row_max * visible) as f64;
                    mass[hid] += row_mass as f64;
                }
            }
        }
        rows += (SEQ_LEN - 1) as u64;
    }

    let null_targets = argmax_targets(&null_sum, &count, n);
    let describe = |t: &[Option<usize>]| {
        let (mut self_count, mut covered) = (0, 0);
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for (i, top) in t.iter().enumerate() {
            let Some(top) = top else { continue };
            covered += 1;
            if *top == i {
                self_count += 1;
            } else {
                *votes.entry(*top).or_insert(0) += 1;
            }
        }
        let mut ranked: Vec<(usize, usize)> = votes.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let sink = ranked.first().map(|&(j, c)| format!("{:?}({c})", (filtered[j] as u8) as char)).unwrap_or_else(|| "-".to_string());
        (self_count as f32 / covered as f32, sink)
    };
    let (null_self, null_sink) = describe(&null_targets);
    let mut lines = vec![format!("[{name}] {used} windows used; null (uniform attention): self rate {null_self:.2}, top sink {null_sink}")];
    let (mut agree_all, mut kl_all, mut peak_all, mut mass_all) = (0.0, 0.0, 0.0, 0.0);
    for b in 0..N_BLOCKS {
        for h in 0..N_HEADS {
            let hid = b * N_HEADS + h;
            let targets = argmax_targets(&wsum[hid * n * n..(hid + 1) * n * n], &count, n);
            let (self_rate, sink) = describe(&targets);
            let pairs: Vec<(usize, usize)> = targets.iter().zip(null_targets.iter()).filter_map(|(a, c)| Some(((*a)?, (*c)?))).collect();
            let agree = pairs.iter().filter(|(a, c)| a == c).count() as f64 / pairs.len() as f64;
            let (k, p, ms) = (kl[hid] / rows as f64, peak[hid] / rows as f64, mass[hid] / rows as f64);
            lines.push(format!(
                "  block {b} head {h}: KL={k:.5} peak={p:.3} mass={ms:.3} self={self_rate:.2} sink={sink} null-argmax-agreement={agree:.2}"
            ));
            agree_all += agree;
            kl_all += k;
            peak_all += p;
            mass_all += ms;
        }
    }
    let hn = heads as f64;
    lines.push(format!("  mean over {heads} heads: KL={:.5} peak={:.3} mass={:.3} null-argmax-agreement={:.2}", kl_all / hn, peak_all / hn, mass_all / hn, agree_all / hn));
    lines
}

fn main() {
    let full_encoded = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let split = (full_encoded.len() as f32 * 0.9) as usize;
    let (train_set, held_out) = full_encoded.split_at(split);
    // STEPS / WINDOWS env overrides exist for smoke-testing the pipeline in
    // seconds; the real run uses the defaults.
    let steps: usize = std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(64000);

    let conditions = [("softmax1+qknorm", true, true), ("plain", false, false)];
    let results: Vec<(Vec<String>, Vec<String>)> = thread::scope(|scope| {
        let handles: Vec<_> = conditions
            .iter()
            .map(|&(name, softmax1, qknorm)| {
                scope.spawn(move || {
                    let path = format!("attention_uniformity_{}.txt", name.replace('+', "_"));
                    let mut log = Vec::new();
                    let (model, grad_accum) = match load(&path) {
                        Some(loaded) => {
                            log.push(format!("loaded checkpoint {path}, skipped training"));
                            loaded
                        }
                        None => {
                            let (m, g) = train(train_set, held_out, steps, softmax1, qknorm, &mut log);
                            save(&path, &m, &g);
                            log.push(format!("saved checkpoint {path}"));
                            (m, g)
                        }
                    };
                    let mut distinct: Vec<usize> = train_set.to_vec();
                    distinct.sort_unstable();
                    distinct.dedup();
                    let mut by_accum = distinct.clone();
                    by_accum.sort_by(|&a, &b| grad_accum[a].partial_cmp(&grad_accum[b]).unwrap());
                    let noise_floor = grad_accum[by_accum[distinct.len() / 5]];
                    let filtered: Vec<usize> = distinct.iter().copied().filter(|&b| grad_accum[b] >= noise_floor).collect();
                    log.push(format!("{} of {} bytes pass the noise filter", filtered.len(), distinct.len()));
                    let analysis = analyze(name, &model, train_set, &filtered, softmax1, qknorm);
                    (log, analysis)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for ((name, _, _), (log, analysis)) in conditions.iter().zip(results.iter()) {
        println!("== {name} ==");
        for l in log {
            println!("{l}");
        }
        for l in analysis {
            println!("{l}");
        }
        println!();
    }
}
