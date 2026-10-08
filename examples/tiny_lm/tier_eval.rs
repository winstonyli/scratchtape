// Cross-tier evaluation (docs/tiers_design.md): a trained byte-LM checkpoint on the GPU, with extra tiers
// applied to its next-byte distribution on the CPU, scored in held-out nats/byte like every other number here.
//
//   tier_eval <checkpoint> [corpus=novels6] [d_model=256] [heads=8] [d_ff=512] [blocks=4] [tap=<block index>]
//             [expect=<CE to reproduce>] [stride=<score every n-th held-out window>] [offset=<first window, < stride>] [part=<i>/<n>] [store_part=<i>/<n>] [store=<windows in the memory>] [store_from=train|held|both] [memory=flat|causal|online] [recent=<windows>] [warm=<t0>] [dump=<path>]
//             [tier=<spec> ...]
//
// A `tier=<spec>` is one or more tier parts joined by '+', applied in order to each position's distribution; each
// spec is scored on its own, next to the plain model. Parts (add new ones by implementing `Tier` and a line in
// `build_tier`):
//   lexicon:<eps>   the symbolic CPU tier: a word list from the TRAIN split masks next bytes that cannot continue
//                   or end a known word; p' = (1-eps) * renorm(masked p) + eps * p. It only constrains a position
//                   once the window has shown a word boundary (the start of a window may cut a word).
//   words:<order>:<lambda>
//                   a second CPU tier, word statistics from the TRAIN split: inside a word, the words that extend
//                   the current prefix (order 0: all train words by frequency; order 1: words that followed the
//                   previous word, backing off to order 0 where that has none) give a next-letter / end-of-word
//                   distribution q (the end mass is spread over non-letter bytes in the model's own proportions);
//                   p' = (1-lambda) * p + lambda * q.
//                   Flags after a third ':' : `h` reads CONTEXT (256) bytes of preceding text instead of only the 64-byte window,
//                   so a word cut by the window's start is seen whole (a CPU-side working memory the model lacks); `f` also
//                   predicts the FIRST letter of the next word (between words) from the words that followed the previous
//                   word. `lexicon:<eps>:h` takes the `h` flag too.
//   knn:<k>:<lambda>:<temp>
//                   the memory tier (kNN-LM style): keys are `tap` hidden states of `store` evenly spaced TRAIN
//                   windows run through the frozen model, values their next bytes; the k nearest keys (squared L2)
//                   give p_knn(b) ~ sum exp(-(d - d_nearest)/temp) over neighbours with value b (k <= 64);
//                   p' = (1-lambda) * p + lambda * p_knn.
// `dump=<path>` writes six f32 per scored position of the last spec's knn tier (chunk, p_target, k_target, entropy, d0, dk) for `gate_fit`.
// `warm=<t0>` scores each byte only at window position >= t0 (windows then start every 64 - t0 bytes), so the model
// always has at least t0 bytes of context: the plain protocol (warm 0) gives the early positions of every window almost
// none, which a tier reading longer context (the `h` flag) can exploit; `warm=32` is the control for that.
// `memory=online` is the same idea written as it is read: the memory starts as the train keys and each scored chunk's
// scored positions are appended after scoring, so a query sees the past text of its own book up to the previous chunk
// (32 windows of write latency) and nothing else; it needs stride 1. `recent=<n>` (causal) limits a query to the n
// windows before its own.
// `memory=causal` makes the memory a past-only in-document one: the train keys plus ALL held-out windows (in order), where
// a query in window w sees the train keys and only those held-out windows before w in the same book (`store` counts
// train windows; 0 for no train keys). `stride`/`offset`/`part` still choose what is scored.
// `store_from` chooses the memory's contents: `train` (default; `store` evenly spaced train windows), `held` (every
// held-out window NOT scored, i.e. whose index is not offset mod stride, which needs stride > 1: text the model never
// trained on, but from the same books as the scored windows, some of it adjacent), or `both`.
// `part=i/n` scores only the i-th of n equal contiguous parts of the held-out text, and `store_part=i/n` limits the
// held-out windows that `store_from=held|both` may use to that part. The held-out text is the last 10% of each book in
// turn, so parts 0/2 and 1/2 of novels6 are (mostly) different books: a memory from the other part is unseen by the model
// AND from other documents, separating "never trained on" from "adjacent to the scored text".
// `tap` picks which block's output the tiers get as the position's hidden state (default: the last block, i.e. the
// final residual stream before the final LayerNorm). With `stride` > 1 the model alone is scored on the same
// subsample, so tiers are always compared to the plain model on identical positions (Gate 0's `expect` needs stride 1). `offset` picks which of the `stride` interleaved subsamples, so tuning and
// testing can use disjoint windows.
//
// Gate 0: the CE computed here from the logits must match the device's own row losses (1e-4) and, with
// `expect=`, the number recorded for the checkpoint (to its 4 printed digits).
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::knn::{KMAX, KnnStore};
use scratchtape::gpu_step::tape::{Config, DeviceTape, model_forward};
use scratchtape::gpu_step::{DeviceParams, read};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::time::Duration;

#[path = "../common/mod.rs"]
mod common;
use common::{held_out_docs, split_corpus};

const SEQ_LEN: usize = 64;
const VOCAB: usize = 256;
/// Windows per forward launch.
const CHUNK: usize = 32;
/// Bytes of preceding text the CPU tiers may read with the `h` flag (the model itself sees only its 64-byte window).
const CONTEXT: usize = 256;

/// What a tier sees at one position: the row within the chunk last passed to `prepare`, the window's input bytes
/// up to and including this one, the byte it predicts, and the tapped hidden state (empty unless a tier asked).
struct Ctx<'a> {
    row: usize,
    window: &'a [usize],
    /// Up to CONTEXT bytes of the text up to and including this position, reaching back past the window's start.
    context: &'a [usize],
    target: usize,
    #[allow(dead_code)] // for tiers that read the hidden state directly rather than via a store
    hidden: &'a [f32],
}

trait Tier {
    fn needs_hidden(&self) -> bool {
        false
    }
    /// Called once per forward chunk, before its rows are adjusted, with the tapped hidden states (if needed) and the
    /// chunk's window starts in the held-out text.
    fn prepare(&mut self, _chunk: usize, _hidden: &[f32], _rows: usize, _starts: &[usize]) {}
    /// Rewrites the next-byte distribution `probs` (sums to 1) in place.
    fn adjust(&mut self, ctx: &Ctx, probs: &mut [f64]);
    /// A line of tier-specific statistics, printed after a scoring pass.
    fn report(&self) -> String {
        String::new()
    }
    /// Writes this tier's per-position dump (the `dump=` option) to `path`; tiers without one do nothing.
    fn dump(&self, _path: &str, _last: bool) {}
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
    hist: bool,
    positions: usize,
    constrained: usize,
    top1_rejected: usize,
    top1_fixed: usize,
}

impl Lexicon {
    fn new(train: &[usize], eps: f64, hist: bool) -> Lexicon {
        let (mut prefixes, mut words) = (HashSet::new(), HashSet::new());
        for word in word_list(train) {
            for n in 1..=word.len() {
                prefixes.insert(word[..n].to_vec());
            }
            words.insert(word);
        }
        Lexicon { prefixes, words, eps, hist, positions: 0, constrained: 0, top1_rejected: 0, top1_fixed: 0 }
    }
}

impl Tier for Lexicon {
    fn adjust(&mut self, ctx: &Ctx, probs: &mut [f64]) {
        self.positions += 1;
        // The current word so far, only if a boundary precedes it inside the window.
        let seq = if self.hist { ctx.context } else { ctx.window };
        let Some(boundary) = seq.iter().rposition(|&b| !is_letter(b)) else { return };
        let word: Vec<u8> = seq[boundary + 1..].iter().map(|&b| b as u8).collect();
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

/// Words (letters/apostrophes) from the train split with counts, as sorted lists for prefix lookup.
struct Words {
    order: usize,
    lambda: f64,
    hist: bool,
    first: bool,
    unigram: Vec<(Vec<u8>, u32)>,
    after: std::collections::HashMap<Vec<u8>, Vec<(Vec<u8>, u32)>>,
    positions: usize,
    applied: usize,
    applied_bigram: usize,
    applied_first: usize,
}

fn sorted_counts(m: std::collections::HashMap<Vec<u8>, u32>) -> Vec<(Vec<u8>, u32)> {
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort();
    v
}

impl Words {
    fn new(train: &[usize], order: usize, lambda: f64, hist: bool, first: bool) -> Words {
        use std::collections::HashMap;
        assert!(order <= 1, "words:<order 0|1>:<lambda>");
        let list = word_list(train);
        let mut uni: HashMap<Vec<u8>, u32> = HashMap::new();
        for w in &list {
            *uni.entry(w.clone()).or_default() += 1;
        }
        let mut bi: HashMap<Vec<u8>, HashMap<Vec<u8>, u32>> = HashMap::new();
        if order == 1 {
            for pair in list.windows(2) {
                *bi.entry(pair[0].clone()).or_default().entry(pair[1].clone()).or_default() += 1;
            }
        }
        Words {
            order,
            lambda,
            hist,
            first,
            unigram: sorted_counts(uni),
            after: bi.into_iter().map(|(k, v)| (k, sorted_counts(v))).collect(),
            positions: 0,
            applied: 0,
            applied_bigram: 0,
            applied_first: 0,
        }
    }

    /// Over the list's words that extend `prefix`: counts of the next letter, of words ending exactly there, and
    /// the total.
    fn extend(list: &[(Vec<u8>, u32)], prefix: &[u8]) -> ([f64; VOCAB], f64, f64) {
        let mut letters = [0f64; VOCAB];
        let (mut end, mut total) = (0.0, 0.0);
        let from = list.partition_point(|(w, _)| w.as_slice() < prefix);
        for (w, c) in list[from..].iter().take_while(|(w, _)| w.starts_with(prefix)) {
            total += *c as f64;
            if w.len() == prefix.len() {
                end += *c as f64;
            } else {
                letters[w[prefix.len()] as usize] += *c as f64;
            }
        }
        (letters, end, total)
    }
}

impl Tier for Words {
    fn adjust(&mut self, ctx: &Ctx, probs: &mut [f64]) {
        self.positions += 1;
        let seq = if self.hist { ctx.context } else { ctx.window };
        let Some(boundary) = seq.iter().rposition(|&b| !is_letter(b)) else { return };
        let prefix: Vec<u8> = seq[boundary + 1..].iter().map(|&b| b as u8).collect();
        if prefix.is_empty() && !self.first {
            return;
        }
        // The word before the current one: the last letter run ending at or before the boundary.
        let before = &seq[..=boundary];
        let prev: Vec<u8> = match before.iter().rposition(|&b| is_letter(b)) {
            Some(end) => {
                let start = before[..=end].iter().rposition(|&b| !is_letter(b)).map_or(0, |i| i + 1);
                before[start..=end].iter().map(|&b| b as u8).collect()
            }
            None => vec![],
        };
        let mut found = None;
        if self.order == 1 && !prev.is_empty() {
            if let Some(list) = self.after.get(&prev) {
                let e = Words::extend(list, &prefix);
                if e.2 > 0.0 {
                    found = Some(e);
                    self.applied_bigram += 1;
                }
            }
        }
        let (letters, end, total) = found.unwrap_or_else(|| Words::extend(&self.unigram, &prefix));
        if prefix.is_empty() {
            // Between words: the model keeps its split between letters and other bytes; the words that followed the
            // previous word redistribute the letter part over first letters.
            let letter_mass: f64 = (0..VOCAB).filter(|&b| is_letter(b)).map(|b| probs[b]).sum();
            if total <= 0.0 || letter_mass <= 0.0 {
                return;
            }
            self.applied_first += 1;
            for b in (0..VOCAB).filter(|&b| is_letter(b)) {
                probs[b] = (1.0 - self.lambda) * probs[b] + self.lambda * letter_mass * letters[b] / total;
            }
            return;
        }
        let non_letter: f64 = (0..VOCAB).filter(|&b| !is_letter(b)).map(|b| probs[b]).sum();
        if total <= 0.0 || non_letter <= 0.0 {
            return;
        }
        self.applied += 1;
        for b in 0..VOCAB {
            let q = if is_letter(b) { letters[b] / total } else { end / total * probs[b] / non_letter };
            probs[b] = (1.0 - self.lambda) * probs[b] + self.lambda * q;
        }
    }

    fn report(&self) -> String {
        format!(
            "words (order {}{}{}): applied inside words at {}, between words at {} of {} positions ({} from the bigram list)",
            self.order,
            if self.hist { ", long context" } else { "" },
            if self.first { ", first letters" } else { "" },
            self.applied,
            self.applied_first,
            self.positions,
            self.applied_bigram
        )
    }
}

/// The datastore (`gpu_step::knn`): `tap` hidden states of train positions and the bytes that followed them.
/// Shared by all knn specs, which reuse one neighbour search per chunk.
struct Store {
    gpu: RefCell<KnnStore>,
    causal: Option<Causal>,
    online: Option<RefCell<Online>>,
    /// Rows at window positions below this are never scored, so they are not searched.
    warm: usize,
    /// (chunk searched, per row the KMAX nearest (squared distance, value), ascending)
    found: RefCell<(Option<usize>, Vec<(f32, u8)>)>,
}

/// Makes the memory causal: the first `n_train` keys (train windows) are always visible, and of the held-out windows
/// (all of them, in order, after the train keys) a query in window w sees only those before w in its own book.
struct Causal {
    n_train: usize,
    /// First held-out window of the book that window w belongs to.
    first_window: Vec<usize>,
    /// A query sees at most this many windows before its own (`recent=`; usize::MAX for the whole book so far).
    recent: usize,
}

/// A memory written as the text is read: after each scored chunk its scored positions are appended, so a query sees the
/// train keys and, from its own book, everything written before its chunk (a write latency of one chunk).
struct Online {
    n_train: usize,
    /// Where each book's held-out text starts (bytes), and the first key written from it (None until then).
    docs: Vec<usize>,
    doc_first_key: Vec<Option<usize>>,
}

impl Store {
    fn prepare(&self, chunk: usize, hidden: &[f32], rows: usize, starts: &[usize]) {
        if self.found.borrow().0 == Some(chunk) {
            return;
        }
        let mut ranges: Option<(Vec<(u32, u32)>, u32)> = self.causal.as_ref().map(|c| {
            let r = (0..rows)
                .map(|r| {
                    let w = starts[r / SEQ_LEN] / SEQ_LEN;
                    let first = c.first_window[w].max(w.saturating_sub(c.recent));
                    ((c.n_train + first * SEQ_LEN) as u32, (c.n_train + w * SEQ_LEN) as u32)
                })
                .collect();
            (r, c.n_train as u32)
        });
        if let Some(on) = &self.online {
            let on = on.borrow();
            let hi = self.gpu.borrow().len() as u32;
            let r = (0..rows)
                .map(|r| {
                    let d = on.docs.partition_point(|&b| b <= starts[r / SEQ_LEN]) - 1;
                    (on.doc_first_key[d].map_or(hi, |k| k as u32), hi)
                })
                .collect();
            ranges = Some((r, on.n_train as u32));
        }
        // Only the scored rows (window position >= warm) are searched; the others keep (infinity, 0) entries.
        let scored: Vec<usize> = (0..rows).filter(|r| r % SEQ_LEN >= self.warm).collect();
        let d = hidden.len() / rows;
        let q: Vec<f32> = scored.iter().flat_map(|&r| hidden[r * d..(r + 1) * d].iter().copied()).collect();
        let rg = ranges.as_ref().map(|(r, a)| (scored.iter().map(|&i| r[i]).collect::<Vec<_>>(), *a));
        let part = self.gpu.borrow().search_masked(&q, scored.len(), rg.as_ref().map(|(r, a)| (r.as_slice(), *a)));
        let mut found = vec![(f32::INFINITY, 0u8); rows * KMAX];
        for (j, &r) in scored.iter().enumerate() {
            found[r * KMAX..(r + 1) * KMAX].copy_from_slice(&part[j * KMAX..(j + 1) * KMAX]);
        }
        *self.found.borrow_mut() = (Some(chunk), found);
    }

    /// Online memory only: appends the chunk's scored positions (window position >= warm) as new keys.
    fn write_chunk(&self, hidden: &[f32], targets: &[usize], starts: &[usize], warm: usize, d: usize) {
        let Some(on) = &self.online else { return };
        let (mut on, mut gpu) = (on.borrow_mut(), self.gpu.borrow_mut());
        for (j, &s) in starts.iter().enumerate() {
            let doc = on.docs.partition_point(|&b| b <= s) - 1;
            if on.doc_first_key[doc].is_none() {
                on.doc_first_key[doc] = Some(gpu.len());
            }
            let rows = j * SEQ_LEN + warm..(j + 1) * SEQ_LEN;
            // `len()` counts keys not yet uploaded, so no flush per window (that made one 32-key device tile per window).
            gpu.add(&hidden[rows.start * d..rows.end * d], &targets[rows].iter().map(|&t| t as u8).collect::<Vec<_>>());
        }
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
    dump: Option<Vec<f32>>,
    chunk: usize,
}

impl Tier for Knn {
    fn needs_hidden(&self) -> bool {
        true
    }
    fn prepare(&mut self, chunk: usize, hidden: &[f32], rows: usize, starts: &[usize]) {
        self.chunk = chunk;
        self.store.prepare(chunk, hidden, rows, starts);
    }
    fn adjust(&mut self, ctx: &Ctx, probs: &mut [f64]) {
        let found = self.store.found.borrow();
        let nb = &found.1[ctx.row * KMAX..ctx.row * KMAX + self.k];
        // A causal memory can have fewer than k keys to offer (the start of a book): use those, or leave p alone.
        let nb = &nb[..nb.iter().take_while(|n| n.0.is_finite()).count()];
        let p_t = probs[ctx.target];
        let entropy = -probs.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f64>();
        if nb.is_empty() {
            if let Some(d) = &mut self.dump {
                d.extend([self.chunk as f32, p_t as f32, p_t as f32, entropy as f32, 0.0, 0.0]);
            }
            return;
        }
        let mut knn = [0f64; VOCAB];
        let mut total = 0.0;
        for &(dist, v) in nb {
            let w = (-((dist - nb[0].0) as f64) / self.temp).exp();
            knn[v as usize] += w;
            total += w;
        }
        if let Some(d) = &mut self.dump {
            d.extend([self.chunk as f32, p_t as f32, (knn[ctx.target] / total) as f32, entropy as f32, nb[0].0, nb[nb.len() - 1].0]);
        }
        for b in 0..VOCAB {
            probs[b] = (1.0 - self.lambda) * probs[b] + self.lambda * knn[b] / total;
        }
        self.positions += 1;
        self.nearest += nb[0].0 as f64;
        self.kth += nb[nb.len() - 1].0 as f64;
    }
    fn report(&self) -> String {
        let n = self.positions.max(1) as f64;
        format!("knn ({} keys): mean squared distance to nearest {:.1}, to k-th {:.1}", self.store.gpu.borrow().len(), self.nearest / n, self.kth / n)
    }
    fn dump(&self, path: &str, last: bool) {
        if let Some(d) = &self.dump {
            let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
            let tmp = format!("{path}.tmp");
            std::fs::write(&tmp, bytes).unwrap_or_else(|e| panic!("writing {tmp}: {e}"));
            std::fs::rename(&tmp, path).unwrap_or_else(|e| panic!("renaming {tmp} to {path}: {e}"));
            if last {
                println!("dump: {} rows to {path}", d.len() / 6);
            }
        }
    }
}

/// One spec, e.g. `lexicon:0.05` or `lexicon:0.05+knn:16:0.2:50`, into its tiers.
fn build_tier(spec: &str, train: &[usize], store: &Option<Rc<Store>>, dump: bool) -> Vec<Box<dyn Tier>> {
    spec.split('+')
        .map(|part| {
            let (kind, arg) = part.split_once(':').unwrap_or((part, ""));
            match kind {
                "lexicon" => {
                    let (eps, flags) = arg.split_once(':').unwrap_or((arg, ""));
                    let eps = eps.parse().unwrap_or_else(|_| panic!("lexicon:<eps>[:h], got {part}"));
                    Box::new(Lexicon::new(train, eps, flags.contains('h'))) as Box<dyn Tier>
                }
                "words" => {
                    let a: Vec<&str> = arg.split(':').collect();
                    assert!(a.len() >= 2 && a.len() <= 3, "words:<order>:<lambda>[:<flags h f>], got {part}");
                    let flags = a.get(2).copied().unwrap_or("");
                    Box::new(Words::new(train, a[0].parse().unwrap(), a[1].parse().unwrap(), flags.contains('h'), flags.contains('f'))) as Box<dyn Tier>
                }
                "knn" => {
                    let a: Vec<f64> = arg.split(':').map(|x| x.parse().unwrap_or_else(|_| panic!("knn:<k>:<lambda>:<temp>, got {part}"))).collect();
                    assert!(a.len() == 3 && a[0] >= 1.0 && a[0] as usize <= KMAX, "knn:<k>:<lambda>:<temp> with 1 <= k <= {KMAX}, got {part}");
                    let store = store.clone().expect("a knn tier needs the store");
                    Box::new(Knn { store, k: a[0] as usize, lambda: a[1], temp: a[2], positions: 0, nearest: 0.0, kth: 0.0, dump: dump.then(Vec::new), chunk: 0 }) as Box<dyn Tier>
                }
                _ => panic!("unknown tier {kind} in {spec}"),
            }
        })
        .collect()
}

fn from_is_train(v: &str) -> bool {
    v == "train"
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

fn forward(dev: &DeviceParams, cfg: &Config, data: &[usize], starts: &[usize], tap: usize, want_hidden: bool, want_logits: bool) -> Forward {
    let (mut ids, mut targets) = (vec![], vec![]);
    for &s in starts {
        ids.extend(&data[s..s + SEQ_LEN]);
        targets.extend(&data[s + 1..s + SEQ_LEN + 1]);
    }
    let mut dt = DeviceTape::new(dev);
    let (outs, logits, loss) = model_forward(&mut dt, cfg, &ids, &targets, starts.len());
    let (logits, row_loss) = if want_logits { (read(dt.value(logits)), read(dt.row_losses(loss))) } else { (vec![], vec![]) };
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
            "corpus" | "d_model" | "heads" | "d_ff" | "blocks" | "tap" | "expect" | "stride" | "offset" | "store" | "store_from" | "part" | "store_part" | "memory" | "warm"
            | "recent" | "dump" => {
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
    let warm = size("warm", "0");
    assert!(warm < SEQ_LEN && (warm == 0 || !opt.contains_key("expect")), "warm < {SEQ_LEN}, and `expect` needs warm 0");
    let unit = SEQ_LEN - warm;
    let offset = size("offset", "0");
    assert!(offset < stride, "offset must be below stride");
    let part = |k: &str| -> (usize, usize) {
        let v = get(k, "0/1");
        let (i, n) = v.split_once('/').unwrap_or_else(|| panic!("{k}=<i>/<n>, got {v}"));
        let (i, n): (usize, usize) = (i.parse().unwrap(), n.parse().unwrap());
        assert!(n >= 1 && i < n, "{k}: need i < n");
        (i, n)
    };
    let (part_i, part_n) = part("part");
    let (spart_i, spart_n) = part("store_part");
    if opt.contains_key("dump") {
        assert!(specs.last().is_some_and(|s| s.rsplit('+').next().is_some_and(|p| p.starts_with("knn:"))), "dump needs the last tier= spec to end in knn");
    }
    let corpus = get("corpus", "novels6");

    let flat: Vec<f32> = std::fs::read_to_string(ckpt).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(flat.len(), cfg.len(), "{ckpt} has {} parameters, this config needs {}", flat.len(), cfg.len());
    let (train, held_out) = split_corpus(&corpus);
    let _lease = gpu_lease::hold(Kind::Shared, "scratchtape tier_eval", Duration::from_secs(4 * 3600));
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
        let (causal, online) = match get("memory", "flat").as_str() {
            "flat" => (false, false),
            "causal" => (true, false),
            "online" => (false, true),
            m => panic!("memory is flat, causal or online, got {m}"),
        };
        assert!(!online || (stride == 1 && from_is_train(&get("store_from", "train"))), "memory=online needs stride 1 and store_from=train");
        let from = if causal { "both".to_string() } else { get("store_from", "train") };
        assert!(["train", "held", "both"].contains(&from.as_str()), "store_from is train, held or both");
        assert!(from == "train" || causal || stride > 1, "store_from=held needs stride > 1 so some held-out windows are left unscored");
        let mut gpu = KnnStore::new(cfg.d, 16384);
        let (mut n_train, mut n_held) = (0, 0);
        let build = std::time::Instant::now();
        let mut t_fwd = std::time::Duration::ZERO;
        let mut load = |data: &[usize], starts: &[usize]| {
            // Only the hidden states are needed (no logits or losses), so launches can be twice as large.
            for chunk in starts.chunks(2 * CHUNK) {
                let t0 = std::time::Instant::now();
                let f = forward(&dev, &cfg, data, chunk, tap, true, false);
                t_fwd += t0.elapsed();
                gpu.add(&f.hidden, &f.targets.iter().map(|&t| t as u8).collect::<Vec<_>>());
            }
        };
        if from != "held" && want > 0 {
            let total = (train.len() - 1) / SEQ_LEN;
            let starts: Vec<usize> = (0..want.min(total)).map(|i| i * total / want.min(total) * SEQ_LEN).collect();
            load(&train, &starts);
            n_train = starts.len();
        }
        if from != "train" {
            let windows = (held_out.len() - 1) / SEQ_LEN;
            let unscored: Vec<usize> = (0..windows).filter(|i| causal || (i % stride != offset && i * spart_n / windows == spart_i)).map(|i| i * SEQ_LEN).collect();
            let starts: Vec<usize> = if causal { unscored } else { (0..want.min(unscored.len())).map(|i| unscored[i * unscored.len() / want.min(unscored.len())]).collect() };
            load(&held_out, &starts);
            n_held = starts.len();
        }
        gpu.finish();
        println!(
            "memory build: {:.1}s = forward + hidden readback {:.1}s, host copy + norms {:.1}s, upload {:.1}s, other {:.1}s",
            build.elapsed().as_secs_f64(),
            t_fwd.as_secs_f64(),
            gpu.flush_time[0].as_secs_f64(),
            gpu.flush_time[1].as_secs_f64(),
            (build.elapsed() - t_fwd - gpu.flush_time[0] - gpu.flush_time[1]).as_secs_f64()
        );
        println!(
            "memory: {} keys of {} floats from {n_train} train + {n_held} unscored held-out windows{} (tap = block {tap}), searched on the GPU in {}",
            gpu.len(),
            cfg.d,
            if online { ", then written online as scored" } else { "" },
            if gpu.is_f16() { "f16 on matrix cores (KNN_F32=1 for exact f32)" } else { "f32" }
        );
        let n_train_keys = n_train * SEQ_LEN;
        let online = online.then(|| {
            let docs = held_out_docs(&corpus);
            RefCell::new(Online { n_train: n_train_keys, doc_first_key: vec![None; docs.len()], docs })
        });
        let causal = causal.then(|| {
            let docs = held_out_docs(&corpus);
            let windows = (held_out.len() - 1) / SEQ_LEN;
            let first_window = (0..windows).map(|w| docs[docs.partition_point(|&b| b <= w * SEQ_LEN) - 1].div_ceil(SEQ_LEN)).collect();
            Causal { n_train: n_train_keys, first_window, recent: size("recent", &usize::MAX.to_string()) }
        });
        Some(Rc::new(Store { gpu: RefCell::new(gpu), causal, online, warm, found: RefCell::new((None, vec![])) }))
    } else {
        None
    };

    let dump_spec = opt.contains_key("dump").then(|| specs.len() - 1);
    let mut tiers: Vec<Vec<Box<dyn Tier>>> = specs.iter().enumerate().map(|(i, s)| build_tier(s, &train, &store, dump_spec == Some(i))).collect();
    let need_hidden = tiers.iter().flatten().any(|t| t.needs_hidden());
    let (mut ce_model, mut ce_device) = (0.0f64, 0.0f64);
    let mut ce_tier = vec![0.0f64; tiers.len()];
    let n_win = (held_out.len() - 1) / SEQ_LEN;
    let starts: Vec<usize> = (offset * unit..held_out.len() - SEQ_LEN).step_by(unit * stride).filter(|s| s / SEQ_LEN * part_n / n_win == part_i).collect();
    // Wall time per phase: forward launch + readback, tier prepare (kNN search + readback), per-row scoring, online writes.
    let mut phase = [std::time::Duration::ZERO; 4];
    let t_all = std::time::Instant::now();
    let (mut t_lap, mut paused) = (std::time::Instant::now(), std::time::Duration::ZERO);
    for (ci, chunk) in starts.chunks(CHUNK).enumerate() {
        // Yield to another job's exclusive lease, and log speed per 20 chunks (paused time excluded) so a slow stretch reads as contention.
        let t_p = std::time::Instant::now();
        gpu_lease::pause_while_exclusive();
        paused += t_p.elapsed();
        if ci > 0 && ci % 20 == 0 {
            eprintln!("speed: chunks {}..{} {:.2} s/chunk ({:.0} s paused)", ci - 20, ci, (t_lap.elapsed() - paused).as_secs_f64() / 20.0, paused.as_secs_f64());
            (t_lap, paused) = (std::time::Instant::now(), std::time::Duration::ZERO);
            if let Some(path) = opt.get("dump") {
                for tier in tiers.last().unwrap() {
                    tier.dump(path, false);
                }
            }
        }
        let t0 = std::time::Instant::now();
        let f = forward(&dev, &cfg, &held_out, chunk, tap, need_hidden, true);
        let rows = f.ids.len();
        phase[0] += t0.elapsed();
        let t0 = std::time::Instant::now();
        for tier in tiers.iter_mut().flatten() {
            tier.prepare(ci, &f.hidden, rows, chunk);
        }
        phase[1] += t0.elapsed();
        let t0 = std::time::Instant::now();
        for r in 0..rows {
            let t = r % SEQ_LEN;
            if t < warm {
                continue;
            }
            let window = &f.ids[r - t..=r];
            let abs = chunk[r / SEQ_LEN] + t;
            let context = &held_out[(abs + 1).saturating_sub(CONTEXT)..=abs];
            let probs = softmax(&f.logits[r * VOCAB..(r + 1) * VOCAB]);
            ce_model -= probs[f.targets[r]].ln();
            ce_device += f.row_loss[r] as f64;
            let hid = if need_hidden { &f.hidden[r * cfg.d..(r + 1) * cfg.d] } else { &[][..] };
            let ctx = Ctx { row: r, window, context, target: f.targets[r], hidden: hid };
            for (i, spec_tiers) in tiers.iter_mut().enumerate() {
                let mut p = probs.clone();
                for tier in spec_tiers.iter_mut() {
                    tier.adjust(&ctx, &mut p);
                }
                ce_tier[i] -= p[f.targets[r]].max(1e-300).ln();
            }
        }
        phase[2] += t0.elapsed();
        let t0 = std::time::Instant::now();
        if let Some(st) = &store {
            st.write_chunk(&f.hidden, &f.targets, chunk, warm, cfg.d);
        }
        phase[3] += t0.elapsed();
    }
    let total = t_all.elapsed().as_secs_f64();
    let pct = |d: std::time::Duration| format!("{:.1}s ({:.0}%)", d.as_secs_f64(), 100.0 * d.as_secs_f64() / total);
    println!(
        "timing: {total:.1}s over {} chunks: forward {}, tier prepare {}, row scoring {}, online writes {}",
        starts.chunks(CHUNK).count(),
        pct(phase[0]),
        pct(phase[1]),
        pct(phase[2]),
        pct(phase[3])
    );
    let n = (starts.len() * unit) as f64;
    let (ce_model, ce_device) = (ce_model / n, ce_device / n);
    println!("checkpoint {ckpt}: {} windows (stride {stride}, offset {offset}, part {part_i}/{part_n}), {} positions; tap = block {tap}", starts.len(), n as usize);
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
    if let Some(path) = opt.get("dump") {
        for tier in tiers.last().unwrap() {
            tier.dump(path, true);
        }
    }
}
