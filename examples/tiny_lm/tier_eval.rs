// Cross-tier evaluation (docs/tiers_design.md): a trained byte-LM checkpoint on the GPU, with extra tiers
// applied to its next-byte distribution on the CPU, scored in held-out nats/byte like every other number here.
//
//   tier_eval <checkpoint> [corpus=novels6] [d_model=256] [heads=8] [d_ff=512] [blocks=4] [tap=<block index>]
//             [expect=<CE to reproduce>] [tier=<spec> ...]
//
// A `tier=<spec>` is one or more tier parts joined by '+', applied in order to each position's distribution; each
// spec is scored on its own, next to the plain model. Parts (add new ones by implementing `Tier` and a line in
// `build_tier`):
//   lexicon:<eps>   the symbolic CPU tier: a word list from the TRAIN split masks next bytes that cannot continue
//                   or end a known word; p' = (1-eps) * renorm(masked p) + eps * p. It only constrains a position
//                   once the window has shown a word boundary (the start of a window may cut a word).
// `tap` picks which block's output the tiers get as the position's hidden state (default: the last block, i.e. the
// final residual stream before the final LayerNorm); no current tier reads it, the memory tier will.
//
// Gate 0: the CE computed here from the logits must match the device's own row losses (1e-4) and, with
// `expect=`, the number recorded for the checkpoint (to its 4 printed digits).
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::tape::{Config, DeviceTape, model_forward};
use scratchtape::gpu_step::{DeviceParams, read};
use std::collections::HashSet;
use std::time::Duration;

#[path = "../common/mod.rs"]
mod common;
use common::split_corpus;

const SEQ_LEN: usize = 64;
const VOCAB: usize = 256;
/// Held-out windows per forward launch.
const CHUNK: usize = 32;

/// What a tier sees at one position: the window's input bytes up to and including this one, the byte it
/// predicts, and the tapped hidden state (empty unless a tier asked for it).
struct Ctx<'a> {
    window: &'a [usize],
    target: usize,
    hidden: &'a [f32],
}

trait Tier {
    fn needs_hidden(&self) -> bool {
        false
    }
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
        let mut word: Vec<u8> = vec![];
        for &b in train.iter().chain(std::iter::once(&(b' ' as usize))) {
            if is_letter(b) {
                word.push(b as u8);
            } else if !word.is_empty() {
                for n in 1..=word.len() {
                    prefixes.insert(word[..n].to_vec());
                }
                words.insert(std::mem::take(&mut word));
            }
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

/// One spec, e.g. `lexicon:0.05` or `lexicon:0.05+knn:...`, into its tiers.
fn build_tier(spec: &str, train: &[usize]) -> Vec<Box<dyn Tier>> {
    spec.split('+')
        .map(|part| {
            let (kind, arg) = part.split_once(':').unwrap_or((part, ""));
            match kind {
                "lexicon" => Box::new(Lexicon::new(train, arg.parse().unwrap_or_else(|_| panic!("lexicon:<eps>, got {part}")))) as Box<dyn Tier>,
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
            "corpus" | "d_model" | "heads" | "d_ff" | "blocks" | "tap" | "expect" => assert!(opt.insert(k, v.to_string()).is_none(), "{k} given twice"),
            _ => panic!("unknown key {k}"),
        }
    }
    let get = |k: &str, d: &str| opt.get(k).cloned().unwrap_or(d.to_string());
    let size = |k: &str, d: &str| -> usize { get(k, d).parse().unwrap() };
    let cfg = Config { vocab: VOCAB, d: size("d_model", "256"), heads: size("heads", "8"), d_ff: size("d_ff", "512"), t: SEQ_LEN, n_blocks: size("blocks", "4"), softmax1: false };
    let tap = size("tap", &(cfg.n_blocks - 1).to_string());
    assert!(tap < cfg.n_blocks, "tap must be a block index below {}", cfg.n_blocks);
    let corpus = get("corpus", "novels6");

    let flat: Vec<f32> = std::fs::read_to_string(ckpt).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(flat.len(), cfg.len(), "{ckpt} has {} parameters, this config needs {}", flat.len(), cfg.len());
    let (train, held_out) = split_corpus(&corpus);
    let _lease = gpu_lease::hold(Kind::Shared, "scratchtape tier_eval", Duration::from_secs(3600));
    let dev = DeviceParams::upload(&flat);

    let mut tiers: Vec<Vec<Box<dyn Tier>>> = specs.iter().map(|s| build_tier(s, &train)).collect();
    let need_hidden = tiers.iter().flatten().any(|t| t.needs_hidden());
    let (mut ce_model, mut ce_device) = (0.0f64, 0.0f64);
    let mut ce_tier = vec![0.0f64; tiers.len()];
    let starts: Vec<usize> = (0..held_out.len() - SEQ_LEN).step_by(SEQ_LEN).collect();
    for chunk in starts.chunks(CHUNK) {
        let (mut ids, mut targets) = (vec![], vec![]);
        for &s in chunk {
            ids.extend(&held_out[s..s + SEQ_LEN]);
            targets.extend(&held_out[s + 1..s + SEQ_LEN + 1]);
        }
        let mut dt = DeviceTape::new(&dev);
        let (outs, logits, loss) = model_forward(&mut dt, &cfg, &ids, &targets, chunk.len());
        let rows = ids.len();
        let (logits, row_loss) = (read(dt.value(logits)), read(dt.row_losses(loss)));
        let hidden = if need_hidden { read(dt.value(outs[tap])) } else { vec![] };
        for r in 0..rows {
            let t = r % SEQ_LEN;
            let window = &ids[r - t..=r];
            let probs = softmax(&logits[r * VOCAB..(r + 1) * VOCAB]);
            ce_model -= probs[targets[r]].ln();
            ce_device += row_loss[r] as f64;
            let hid = if need_hidden { &hidden[r * cfg.d..(r + 1) * cfg.d] } else { &[][..] };
            let ctx = Ctx { window, target: targets[r], hidden: hid };
            for (i, spec_tiers) in tiers.iter_mut().enumerate() {
                let mut p = probs.clone();
                for tier in spec_tiers.iter_mut() {
                    tier.adjust(&ctx, &mut p);
                }
                ce_tier[i] -= p[targets[r]].max(1e-300).ln();
            }
        }
    }
    let n = (starts.len() * SEQ_LEN) as f64;
    let (ce_model, ce_device) = (ce_model / n, ce_device / n);
    println!("checkpoint {ckpt}: {} windows, {} positions; tap = block {tap}", starts.len(), n as usize);
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
