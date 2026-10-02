// Cross-tier evaluation (docs/tiers_design.md): a trained byte-LM checkpoint on the GPU, with extra tiers
// applied to its next-byte distribution on the CPU, scored in held-out nats/byte like every other number here.
//
//   tier_eval <checkpoint> [corpus=novels6] [d_model=256] [heads=8] [d_ff=512] [blocks=4] [tap=<block index>]
//             [expect=<CE to reproduce>] [stride=<score every n-th held-out window>] [offset=<first window, < stride>] [store=<train windows in the memory>]
//             [threads=<search threads>] [tier=<spec> ...]
//
// A `tier=<spec>` is one or more tier parts joined by '+', applied in order to each position's distribution; each
// spec is scored on its own, next to the plain model. Parts (add new ones by implementing `Tier` and a line in
// `build_tier`):
//   lexicon:<eps>   the symbolic CPU tier: a word list from the TRAIN split masks next bytes that cannot continue
//                   or end a known word; p' = (1-eps) * renorm(masked p) + eps * p. It only constrains a position
//                   once the window has shown a word boundary (the start of a window may cut a word).
//   knn:<k>:<lambda>:<temp>
//                   the memory tier (kNN-LM style): keys are `tap` hidden states of `store` evenly spaced TRAIN
//                   windows run through the frozen model, values their next bytes; the k nearest keys (squared L2)
//                   give p_knn(b) ~ sum exp(-(d - d_nearest)/temp) over neighbours with value b (k <= 64);
//                   p' = (1-lambda) * p + lambda * p_knn.
// `tap` picks which block's output the tiers get as the position's hidden state (default: the last block, i.e. the
// final residual stream before the final LayerNorm). With `stride` > 1 the model alone is scored on the same
// subsample, so tiers are always compared to the plain model on identical positions (Gate 0's `expect` needs stride 1). `offset` picks which of the `stride` interleaved subsamples, so tuning and
// testing can use disjoint windows.
//
// Gate 0: the CE computed here from the logits must match the device's own row losses (1e-4) and, with
// `expect=`, the number recorded for the checkpoint (to its 4 printed digits).
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::tape::{Config, DeviceTape, model_forward};
use scratchtape::gpu_step::{DeviceParams, read};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::time::Duration;

#[path = "../common/mod.rs"]
mod common;
use common::split_corpus;

const SEQ_LEN: usize = 64;
const VOCAB: usize = 256;
/// Windows per forward launch.
const CHUNK: usize = 32;
/// Neighbours kept per query by the memory tier.
const KMAX: usize = 64;

/// What a tier sees at one position: the row within the chunk last passed to `prepare`, the window's input bytes
/// up to and including this one, the byte it predicts, and the tapped hidden state (empty unless a tier asked).
struct Ctx<'a> {
    row: usize,
    window: &'a [usize],
    target: usize,
    #[allow(dead_code)] // for tiers that read the hidden state directly rather than via a store
    hidden: &'a [f32],
}

trait Tier {
    fn needs_hidden(&self) -> bool {
        false
    }
    /// Called once per forward chunk, before its rows are adjusted, with the tapped hidden states (if needed).
    fn prepare(&mut self, _chunk: usize, _hidden: &[f32], _rows: usize) {}
    /// Rewrites the next-byte distribution `probs` (sums to 1) in place.
    fn adjust(&mut self, ctx: &Ctx, probs: &mut [f64]);
    /// A line of tier-specific statistics, printed after a scoring pass.
    fn report(&self) -> String {
        String::new()
    }
}

fn is_letter(b: usize) -> bool {
    (b as u8).is_ascii_alphabetic() || b == b'\'' as usize
}

/// Every maximal run of letters/apostrophes, in order.
fn word_list(data: &[usize]) -> Vec<Vec<u8>> {
    let (mut out, mut word) = (vec![], vec![]);
    for &b in data.iter().chain(std::iter::once(&(b' ' as usize))) {
        if is_letter(b) {
            word.push(b as u8);
        } else if !word.is_empty() {
            out.push(std::mem::take(&mut word));
        }
    }
    out
}

/// Word list from the train split; masks next bytes that cannot continue or end a known word.
struct Lexicon {
    prefixes: HashSet<Vec<u8>>,
    words: HashSet<Vec<u8>>,
    eps: f64,
    positions: usize,
    constrained: usize,
    top1_rejected: usize,
    top1_fixed: usize,
}

impl Lexicon {
    fn new(train: &[usize], eps: f64) -> Lexicon {
        let (mut prefixes, mut words) = (HashSet::new(), HashSet::new());
        for word in word_list(train) {
            for n in 1..=word.len() {
                prefixes.insert(word[..n].to_vec());
            }
            words.insert(word);
        }
        Lexicon { prefixes, words, eps, positions: 0, constrained: 0, top1_rejected: 0, top1_fixed: 0 }
    }
}

impl Tier for Lexicon {
    fn adjust(&mut self, ctx: &Ctx, probs: &mut [f64]) {
        self.positions += 1;
        // The current word so far, only if a boundary precedes it inside the window.
        let Some(boundary) = ctx.window.iter().rposition(|&b| !is_letter(b)) else { return };
        let word: Vec<u8> = ctx.window[boundary + 1..].iter().map(|&b| b as u8).collect();
        if word.is_empty() {
            return;
        }
        let complete = self.words.contains(&word);
        let mut buf = word.clone();
        let allowed = |b: usize, buf: &mut Vec<u8>| -> bool {
            if is_letter(b) {
                buf.push(b as u8);
                let ok = self.prefixes.contains(buf.as_slice());
                buf.pop();
                ok
            } else {
                complete
            }
        };
        let mask: Vec<bool> = (0..VOCAB).map(|b| allowed(b, &mut buf)).collect();
        let mass: f64 = (0..VOCAB).filter(|&b| mask[b]).map(|b| probs[b]).sum();
        if mass <= 0.0 {
            return;
        }
        self.constrained += 1;
        let argmax = |p: &[f64]| (0..VOCAB).max_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap()).unwrap();
        let top = argmax(probs);
        if !mask[top] {
            self.top1_rejected += 1;
            let masked: Vec<f64> = (0..VOCAB).map(|b| if mask[b] { probs[b] } else { 0.0 }).collect();
            if argmax(&masked) == ctx.target && top != ctx.target {
                self.top1_fixed += 1;
            }
        }
        for b in 0..VOCAB {
            probs[b] = (1.0 - self.eps) * (if mask[b] { probs[b] / mass } else { 0.0 }) + self.eps * probs[b];
        }
    }

    fn report(&self) -> String {
        format!(
            "lexicon ({} words): constrained {}/{} positions; model's top-1 rejected at {} of them, the next-best allowed byte was then right {} times",
            self.words.len(),
            self.constrained,
            self.positions,
            self.top1_rejected,
            self.top1_fixed
        )
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0f32; 8];
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for j in 0..8 {
            s[j] += ca[j] * cb[j];
        }
    }
    s.iter().sum()
}

/// The datastore: `tap` hidden states of train positions and the bytes that followed them. Shared by all knn specs,
/// which reuse one neighbour search per chunk.
struct Store {
    keys: Vec<f32>,
    norms: Vec<f32>,
    vals: Vec<u8>,
    d: usize,
    threads: usize,
    /// (chunk searched, per row the KMAX nearest (squared distance, value), ascending)
    found: RefCell<(Option<usize>, Vec<(f32, u8)>)>,
}

impl Store {
    fn prepare(&self, chunk: usize, hidden: &[f32], rows: usize) {
        if self.found.borrow().0 == Some(chunk) {
            return;
        }
        let mut out = vec![(f32::INFINITY, 0u8); rows * KMAX];
        let per = rows.div_ceil(self.threads);
        let (keys, norms, vals, d) = (&self.keys, &self.norms, &self.vals, self.d);
        std::thread::scope(|s| {
            for (ti, out) in out.chunks_mut(per * KMAX).enumerate() {
                s.spawn(move || {
                    for (i, best) in out.chunks_mut(KMAX).enumerate() {
                        let r = ti * per + i;
                        let q = &hidden[r * d..(r + 1) * d];
                        let qn = dot(q, q);
                        let mut worst = f32::INFINITY;
                        for (j, key) in keys.chunks_exact(d).enumerate() {
                            let dist = norms[j] - 2.0 * dot(q, key);
                            if dist < worst {
                                let mut p = KMAX - 1;
                                while p > 0 && best[p - 1].0 > dist {
                                    best[p] = best[p - 1];
                                    p -= 1;
                                }
                                best[p] = (dist, vals[j]);
                                worst = best[KMAX - 1].0;
                            }
                        }
                        for b in best.iter_mut() {
                            b.0 = (b.0 + qn).max(0.0);
                        }
                    }
                });
            }
        });
        *self.found.borrow_mut() = (Some(chunk), out);
    }
}

struct Knn {
    store: Rc<Store>,
    k: usize,
    lambda: f64,
    temp: f64,
    positions: usize,
    nearest: f64,
    kth: f64,
}

impl Tier for Knn {
    fn needs_hidden(&self) -> bool {
        true
    }
    fn prepare(&mut self, chunk: usize, hidden: &[f32], rows: usize) {
        self.store.prepare(chunk, hidden, rows);
    }
    fn adjust(&mut self, ctx: &Ctx, probs: &mut [f64]) {
        let found = self.store.found.borrow();
        let nb = &found.1[ctx.row * KMAX..ctx.row * KMAX + self.k];
        let mut knn = [0f64; VOCAB];
        let mut total = 0.0;
        for &(dist, v) in nb {
            let w = (-((dist - nb[0].0) as f64) / self.temp).exp();
            knn[v as usize] += w;
            total += w;
        }
        for b in 0..VOCAB {
            probs[b] = (1.0 - self.lambda) * probs[b] + self.lambda * knn[b] / total;
        }
        self.positions += 1;
        self.nearest += nb[0].0 as f64;
        self.kth += nb[self.k - 1].0 as f64;
    }
    fn report(&self) -> String {
        let n = self.positions.max(1) as f64;
        format!("knn ({} keys): mean squared distance to nearest {:.1}, to k-th {:.1}", self.store.vals.len(), self.nearest / n, self.kth / n)
    }
}

/// One spec, e.g. `lexicon:0.05` or `lexicon:0.05+knn:16:0.2:50`, into its tiers.
fn build_tier(spec: &str, train: &[usize], store: &Option<Rc<Store>>) -> Vec<Box<dyn Tier>> {
    spec.split('+')
        .map(|part| {
            let (kind, arg) = part.split_once(':').unwrap_or((part, ""));
            match kind {
                "lexicon" => Box::new(Lexicon::new(train, arg.parse().unwrap_or_else(|_| panic!("lexicon:<eps>, got {part}")))) as Box<dyn Tier>,
                "knn" => {
                    let a: Vec<f64> = arg.split(':').map(|x| x.parse().unwrap_or_else(|_| panic!("knn:<k>:<lambda>:<temp>, got {part}"))).collect();
                    assert!(a.len() == 3 && a[0] >= 1.0 && a[0] as usize <= KMAX, "knn:<k>:<lambda>:<temp> with 1 <= k <= {KMAX}, got {part}");
                    let store = store.clone().expect("a knn tier needs the store");
                    Box::new(Knn { store, k: a[0] as usize, lambda: a[1], temp: a[2], positions: 0, nearest: 0.0, kth: 0.0 }) as Box<dyn Tier>
                }
                _ => panic!("unknown tier {kind} in {spec}"),
            }
        })
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exp: Vec<f64> = logits.iter().map(|&l| (l as f64 - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|e| e / sum).collect()
}

/// One forward launch over the windows of `data` starting at `starts`.
struct Forward {
    ids: Vec<usize>,
    targets: Vec<usize>,
    logits: Vec<f32>,
    row_loss: Vec<f32>,
    hidden: Vec<f32>,
}

fn forward(dev: &DeviceParams, cfg: &Config, data: &[usize], starts: &[usize], tap: usize, want_hidden: bool) -> Forward {
    let (mut ids, mut targets) = (vec![], vec![]);
    for &s in starts {
        ids.extend(&data[s..s + SEQ_LEN]);
        targets.extend(&data[s + 1..s + SEQ_LEN + 1]);
    }
    let mut dt = DeviceTape::new(dev);
    let (outs, logits, loss) = model_forward(&mut dt, cfg, &ids, &targets, starts.len());
    let (logits, row_loss) = (read(dt.value(logits)), read(dt.row_losses(loss)));
    let hidden = if want_hidden { read(dt.value(outs[tap])) } else { vec![] };
    Forward { ids, targets, logits, row_loss, hidden }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: tier_eval <checkpoint> [key=value ...] (see the file header)");
    let ckpt = &args[0];
    let mut opt = std::collections::HashMap::new();
    let mut specs: Vec<String> = vec![];
    for a in &args[1..] {
        let (k, v) = a.split_once('=').unwrap_or_else(|| panic!("expected key=value, got {a}"));
        match k {
            "tier" => specs.push(v.to_string()),
            "corpus" | "d_model" | "heads" | "d_ff" | "blocks" | "tap" | "expect" | "stride" | "offset" | "store" | "threads" => {
                assert!(opt.insert(k, v.to_string()).is_none(), "{k} given twice")
            }
            _ => panic!("unknown key {k}"),
        }
    }
    let get = |k: &str, d: &str| opt.get(k).cloned().unwrap_or(d.to_string());
    let size = |k: &str, d: &str| -> usize { get(k, d).parse().unwrap() };
    let cfg = Config { vocab: VOCAB, d: size("d_model", "256"), heads: size("heads", "8"), d_ff: size("d_ff", "512"), t: SEQ_LEN, n_blocks: size("blocks", "4"), softmax1: false };
    let tap = size("tap", &(cfg.n_blocks - 1).to_string());
    assert!(tap < cfg.n_blocks, "tap must be a block index below {}", cfg.n_blocks);
    let stride = size("stride", "1");
    assert!(stride >= 1 && (stride == 1 || !opt.contains_key("expect")), "stride >= 1, and `expect` needs stride 1");
    let offset = size("offset", "0");
    assert!(offset < stride, "offset must be below stride");
    let corpus = get("corpus", "novels6");

    let flat: Vec<f32> = std::fs::read_to_string(ckpt).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(flat.len(), cfg.len(), "{ckpt} has {} parameters, this config needs {}", flat.len(), cfg.len());
    let (train, held_out) = split_corpus(&corpus);
    let _lease = gpu_lease::hold(Kind::Shared, "scratchtape tier_eval", Duration::from_secs(3600));
    let dev = DeviceParams::upload(&flat);

    // Held-out words the train split has never shown: the ceiling on what a train-split lexicon can know.
    let train_words: HashSet<Vec<u8>> = word_list(&train).into_iter().collect();
    let held_words = word_list(&held_out);
    let oov: Vec<&Vec<u8>> = held_words.iter().filter(|w| !train_words.contains(*w)).collect();
    let oov_bytes: usize = oov.iter().map(|w| w.len()).sum();
    println!(
        "held-out words: {} tokens, {} ({:.2}%) not in the train lexicon, covering {:.2}% of held-out bytes",
        held_words.len(),
        oov.len(),
        100.0 * oov.len() as f64 / held_words.len() as f64,
        100.0 * oov_bytes as f64 / held_out.len() as f64
    );

    let store = if specs.iter().any(|s| s.contains("knn")) {
        let want = size("store", "15000");
        let total = (train.len() - 1) / SEQ_LEN;
        let starts: Vec<usize> = (0..want.min(total)).map(|i| i * total / want.min(total) * SEQ_LEN).collect();
        let (mut keys, mut vals) = (vec![], vec![]);
        for chunk in starts.chunks(CHUNK) {
            let f = forward(&dev, &cfg, &train, chunk, tap, true);
            keys.extend(f.hidden);
            vals.extend(f.targets.iter().map(|&t| t as u8));
        }
        let norms = keys.chunks_exact(cfg.d).map(|k| dot(k, k)).collect();
        println!("memory: {} keys of {} floats from {} train windows (tap = block {tap})", vals.len(), cfg.d, starts.len());
        Some(Rc::new(Store { keys, norms, vals, d: cfg.d, threads: size("threads", "6"), found: RefCell::new((None, vec![])) }))
    } else {
        None
    };

    let mut tiers: Vec<Vec<Box<dyn Tier>>> = specs.iter().map(|s| build_tier(s, &train, &store)).collect();
    let need_hidden = tiers.iter().flatten().any(|t| t.needs_hidden());
    let (mut ce_model, mut ce_device) = (0.0f64, 0.0f64);
    let mut ce_tier = vec![0.0f64; tiers.len()];
    let starts: Vec<usize> = (offset * SEQ_LEN..held_out.len() - SEQ_LEN).step_by(SEQ_LEN * stride).collect();
    for (ci, chunk) in starts.chunks(CHUNK).enumerate() {
        let f = forward(&dev, &cfg, &held_out, chunk, tap, need_hidden);
        let rows = f.ids.len();
        for tier in tiers.iter_mut().flatten() {
            tier.prepare(ci, &f.hidden, rows);
        }
        for r in 0..rows {
            let t = r % SEQ_LEN;
            let window = &f.ids[r - t..=r];
            let probs = softmax(&f.logits[r * VOCAB..(r + 1) * VOCAB]);
            ce_model -= probs[f.targets[r]].ln();
            ce_device += f.row_loss[r] as f64;
            let hid = if need_hidden { &f.hidden[r * cfg.d..(r + 1) * cfg.d] } else { &[][..] };
            let ctx = Ctx { row: r, window, target: f.targets[r], hidden: hid };
            for (i, spec_tiers) in tiers.iter_mut().enumerate() {
                let mut p = probs.clone();
                for tier in spec_tiers.iter_mut() {
                    tier.adjust(&ctx, &mut p);
                }
                ce_tier[i] -= p[f.targets[r]].max(1e-300).ln();
            }
        }
    }
    let n = (starts.len() * SEQ_LEN) as f64;
    let (ce_model, ce_device) = (ce_model / n, ce_device / n);
    println!("checkpoint {ckpt}: {} windows (stride {stride}, offset {offset}), {} positions; tap = block {tap}", starts.len(), n as usize);
    println!("model alone: CE {ce_model:.4} (device row losses {ce_device:.4})");
    assert!((ce_model - ce_device).abs() < 1e-4, "gate 0: CE from logits {ce_model:.5} vs device row losses {ce_device:.5}");
    if let Some(expect) = opt.get("expect") {
        let expect: f64 = expect.parse().unwrap();
        assert!((ce_model - expect).abs() < 6e-5, "gate 0: reproduced {ce_model:.5}, recorded {expect}");
        println!("gate 0 passed: reproduces the recorded {expect}");
    }
    for (i, spec) in specs.iter().enumerate() {
        println!("tier {spec}: CE {:.4} ({:+.4} vs model)", ce_tier[i] / n, ce_tier[i] / n - ce_model);
        for tier in &tiers[i] {
            let report = tier.report();
            if !report.is_empty() {
                println!("  {report}");
            }
        }
    }
}
