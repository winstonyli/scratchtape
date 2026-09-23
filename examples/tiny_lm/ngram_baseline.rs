// Count-based baseline for tiny_lm_corpus.rs: how well does a byte n-gram
// model do on the same Aesop split, in the same units (nats/byte)?
// humble-cortex's whole audit turned on "does the model beat the simplest
// baseline appropriate to the task" - this project's tiny LMs never had
// one.
//
// Interpolated Kneser-Ney (Chen & Goodman 1998): the top order uses raw
// counts, lower orders use continuation counts (how many distinct bytes
// precede an n-gram), each order discounted by Ney's D = n1/(n1 + 2 n2),
// bottoming out in a uniform distribution over 256 bytes so no byte ever
// gets zero probability. Contexts never seen in training back off whole.
//
// Scored two ways on held-out:
//   - windowed: non-overlapping 64-byte windows, context reset at each
//     window start - position t sees only the t+1 bytes before it in its
//     window, exactly the transformer's view (input = corpus[s..s+64],
//     target = corpus[s+1..s+65]).
//   - stream: unlimited left context (only the last N-1 bytes matter).
// The transformer's logged held-out loss averages 20 random windows per
// eval and swings ~+/-0.2 between evals, so compare against its range, not
// one number.
//
// Result: windowed held-out CE falls from 3.266 (unigram) to 1.655 at
// order 7, then flattens (orders 6-10 all 1.655-1.657; stream context
// 1.635). tiny_lm_corpus.rs's plain-softmax model logged 1.738 at step
// 64000 and averaged 1.868 over its last 8 evals (seed 1) - the count
// model is better. ~25s to fit and score, no training.
use std::collections::HashMap;

#[path = "../common/mod.rs"]
mod common;
use common::encode_bytes;

const V: usize = 256;
const SEQ_LEN: usize = 64;

/// Per order k (n-gram length): counts keyed by the full k-gram, and per
/// (k-1)-byte context: (total count, number of distinct next bytes).
struct Level {
    counts: HashMap<Vec<usize>, f64>,
    ctx: HashMap<Vec<usize>, (f64, f64)>,
    d: f64,
}

impl Level {
    fn from_counts(counts: HashMap<Vec<usize>, f64>) -> Self {
        let mut ctx: HashMap<Vec<usize>, (f64, f64)> = HashMap::new();
        let (mut n1, mut n2) = (0.0, 0.0);
        for (gram, &c) in &counts {
            let e = ctx.entry(gram[..gram.len() - 1].to_vec()).or_insert((0.0, 0.0));
            e.0 += c;
            e.1 += 1.0;
            if c == 1.0 {
                n1 += 1.0;
            } else if c == 2.0 {
                n2 += 1.0;
            }
        }
        let d = if n1 + n2 > 0.0 { n1 / (n1 + 2.0 * n2) } else { 0.5 };
        Self { counts, ctx, d }
    }
}

struct KneserNey {
    raw: Vec<Level>,  // raw[k-1]: order-k raw counts (used at the top)
    cont: Vec<Level>, // cont[k-1]: order-k continuation counts (used below)
}

impl KneserNey {
    fn fit(train: &[usize], max_order: usize) -> Self {
        let mut raw = Vec::new();
        let mut cont = Vec::new();
        for k in 1..=max_order {
            let mut counts: HashMap<Vec<usize>, f64> = HashMap::new();
            let mut preceders: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
            for i in 0..=train.len() - k {
                let gram = train[i..i + k].to_vec();
                *counts.entry(gram.clone()).or_insert(0.0) += 1.0;
                if i > 0 {
                    let p = preceders.entry(gram).or_default();
                    if !p.contains(&train[i - 1]) {
                        p.push(train[i - 1]);
                    }
                }
            }
            let cont_counts = preceders.into_iter().map(|(g, p)| (g, p.len() as f64)).collect();
            raw.push(Level::from_counts(counts));
            cont.push(Level::from_counts(cont_counts));
        }
        Self { raw, cont }
    }

    /// P(w | ctx), ctx = the last (k-1) bytes; `top` selects raw counts.
    fn prob(&self, ctx: &[usize], w: usize, top: bool) -> f64 {
        let k = ctx.len() + 1;
        let level = if top { &self.raw[k - 1] } else { &self.cont[k - 1] };
        let lower = || if ctx.is_empty() { 1.0 / V as f64 } else { self.prob(&ctx[1..], w, false) };
        let Some(&(total, distinct)) = level.ctx.get(ctx) else {
            return lower();
        };
        let mut gram = ctx.to_vec();
        gram.push(w);
        let c = level.counts.get(&gram).copied().unwrap_or(0.0);
        (c - level.d).max(0.0) / total + level.d * distinct / total * lower()
    }
}

fn main() {
    let full = encode_bytes(include_str!("../../data/aesops_fables.txt"));
    let split = (full.len() as f32 * 0.9) as usize;
    let (train, held_out) = full.split_at(split);
    println!("train {} bytes, held-out {} bytes", train.len(), held_out.len());

    let max_order = 10;
    let kn = KneserNey::fit(train, max_order);

    // Runnable check: every distribution must sum to 1 over all 256 bytes,
    // for seen, partly-seen and unseen contexts, at every order.
    for ctx in [&train[1000..1009], &held_out[500..509], &[0x7fusize, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f][..]] {
        for k in 1..=max_order {
            let c = &ctx[ctx.len() - (k - 1)..];
            let total: f64 = (0..V).map(|w| kn.prob(c, w, true)).sum();
            assert!((total - 1.0).abs() < 1e-9, "order {k}: probabilities sum to {total}");
        }
    }

    let ce = |n: usize, windowed: bool| -> f64 {
        let mut total = 0.0;
        let mut count = 0usize;
        let starts: Vec<usize> = if windowed { (0..held_out.len() - SEQ_LEN).step_by(SEQ_LEN).collect() } else { vec![0] };
        for &s in &starts {
            let end = if windowed { s + SEQ_LEN } else { held_out.len() - 1 };
            for t in s..end {
                let ctx_start = if windowed { s } else { 0 };
                let lo = (t + 1).saturating_sub(n - 1).max(ctx_start);
                total -= kn.prob(&held_out[lo..=t], held_out[t + 1], true).ln();
                count += 1;
            }
        }
        total / count as f64
    };

    println!("order | held-out CE windowed (nats/byte) | stream | train-set D (raw top order)");
    let mut best = (0, f64::INFINITY);
    for n in 1..=max_order {
        let w = ce(n, true);
        if w < best.1 {
            best = (n, w);
        }
        println!("  {n:>2} | {w:.4} | {:.4} | {:.3}", ce(n, false), kn.raw[n - 1].d);
    }
    println!("best windowed: order {} at {:.4} nats/byte", best.0, best.1);
}
