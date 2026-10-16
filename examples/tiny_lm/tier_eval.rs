// Cross-tier evaluation (docs/tiers_design.md): a trained byte-LM checkpoint on the GPU, with extra tiers
// applied to its next-byte distribution on the CPU, scored in held-out nats/byte like every other number here.
//
//   tier_eval <checkpoint> [corpus=novels6] [d_model=256] [heads=8] [d_ff=512] [blocks=4] [tap=<block index>]
//             [expect=<CE to reproduce>] [stride=<score every n-th held-out window>] [offset=<first window, < stride>] [part=<i>/<n>] [store_part=<i>/<n>] [store=<windows in the memory>] [store_from=train|held|both] [memory=flat|causal|online] [recent=<windows>] [warm=<t0>] [whiten=<alpha>] [classes=1] [survey=1] [dyn=<lr>] [dyn_every=<windows>] [dyn_steps=<k>] [dyn_rms=<beta>] [dyn_pull=<f>] [dyn_keys=adapted|frozen] [dyn_layers=emb,b0..,head] [dyn_control=1] [dump=<path>] [wdump=<path>]
//             [tier=<spec> ...] [generate=<len> [prompt=<held-out byte>] [temp=0.8] [seed=7]]
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
//   wrap:<lambda>   the line-wrap tier: where the 64-byte window shows no newline but the 256-byte context does, the model's mass on newline
//                   and space is re-split by the train text's newline rate at that column (after a letter / after another byte).
//   knn:<k>:<lambda>:<temp>
//                   the memory tier (kNN-LM style): keys are `tap` hidden states of `store` evenly spaced TRAIN
//                   windows run through the frozen model, values their next bytes; the k nearest keys (squared L2)
//                   give p_knn(b) ~ sum exp(-(d - d_nearest)/temp) over neighbours with value b (k <= 64);
//                   p' = (1-lambda) * p + lambda * p_knn.
// `dump=<path>` writes six f32 per scored position of the last spec's knn tier (chunk, p_target, k_target, entropy, d0, dk) for `gate_fit`.
// `wdump=<path>` (with `dump=`) writes eight f32 per scored position of the last spec's words tier (chunk, applied, p_in_target, q_target, entropy_in, ln(1+count mass), bigram, prefix_len) for `words_gate_fit`.
// `warm=<t0>` scores each byte only at window position >= t0 (windows then start every 64 - t0 bytes), so the model
// always has at least t0 bytes of context: the plain protocol (warm 0) gives the early positions of every window almost
// none, which a tier reading longer context (the `h` flag) can exploit; `warm=32` is the control for that.
// `memory=online` is the same idea written as it is read: the memory starts as the train keys and each chunk's scored
// positions are appended before its search, and a window's queries see the keys of the windows before it in their own
// book, as the causal memory does (nothing else); it needs stride 1. An online memory searches the train keys and the in-document keys separately and keeps
// the k nearest of each (the in-document search scans only the tail tiles), instead of the k nearest of the two together.
// `recent=<n>` (causal) limits a query to the n windows before its own.
// `memory=causal` makes the memory a past-only in-document one: the train keys plus ALL held-out windows (in order), where
// a query in window w sees the train keys and only those held-out windows before w in the same book (`store` counts
// train windows; 0 for no train keys). `stride`/`offset`/`part` still choose what is scored.
// `classes=1` prints the loss by position class (letter or not before, kind of byte predicted): where the model's CE sits and which tier gains where.
// `dyn_every=<windows>` (default 32) scores, writes the online memory and adapts in chunks of that many windows; `dyn_steps=<k>` (default 1) takes k steps per chunk.
// `dyn_keys=frozen` takes the hidden states the tiers use (memory queries, in-document keys) from an extra forward of the unadapted checkpoint, so they live in the same space as the train keys; the probabilities still come from the adapted model.
// `dyn_nn=<m>` adds retrieval-driven test-time training (Hardt & Sun 2023, "nearest neighbours"): after each chunk, a step on the m train windows whose
// embedding (mean frozen-model hidden state over the scored rows) is nearest to the chunk's windows (m/windows each). `dyn_nn_mode=add` (default; after the
// chunk's own step), `only` (instead of it) or `random` (m random train windows: the control). `dyn_nn_stride=<n>` (default 8) embeds every n-th memory-corpus
// window; `dyn_nn_lr` is the neighbour step's learning rate (default dyn). Needs `dyn=<lr>`.
// `dyn_layers=<comma list of emb, b0.., head>` adapts only those parameters (host-side step, as dyn_rms).
// `dyn_rms=<beta>` divides each gradient by a bias-corrected running RMS (decay beta; lr is then per-parameter step size, ~1e-4); `dyn_pull=<f>` moves each parameter the fraction f back towards the checkpoint after each step (both: host-side prototype, `Adapter`).
// `dyn=<lr>` is dynamic evaluation (with tiers, the memory's train keys stay from the frozen model while queries and in-document keys come from the adapted one): after each chunk is scored, one SGD step at that lr on the same chunk (text already scored), parameters reset to the checkpoint at each book boundary. `dyn_control=1` takes each step from a chunk half the held-out text away instead (it must not help).
// `survey=1` (needs a tier spec; reads the LAST spec) splits the CE by what the target is: letters by whether the word is in the train lexicon, was seen earlier in the same book, or is new, and by position in the word; other bytes by kind. It prints where the stack's CE sits and a ceiling for an in-document word cache.
// `whiten=<alpha>` (default off) centres the memory's keys and queries and multiplies them by Sigma^(-alpha/2) (Sigma: the covariance of
// 128k train positions, total variance kept); alpha 0.5 gave the best stack (docs/tiers_design.md).
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
use scratchtape::gpu_step::knn::{DOC, KMAX, KnnStore};
use scratchtape::gpu_step::tape::{Config, DeviceTape, model_forward};
use scratchtape::gpu_step::matmul::{Epilogue, MatRef, matmul_f32};
use scratchtape::gpu_step::{DeviceParams, Handle, client, read_many, upload_f32};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Duration;

#[path = "../common/mod.rs"]
mod common;
use common::{held_out_docs, split_corpus};

/// Window length of the model under test: set once from `seq=` (default 64) before anything reads it.
static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(64);
fn seq_len() -> usize {
    SEQ.load(std::sync::atomic::Ordering::Relaxed)
}
const VOCAB: usize = 256;
/// Windows per forward launch.
fn chunk_windows() -> usize {
    32 * 64 / seq_len() // keeps windows * heads * seq under the 65535 launch limit
}
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
    /// Writes this tier's words dump (the `wdump=` option) to `path`; tiers without one do nothing.
    fn wdump(&self, _path: &str, _last: bool) {}
}

fn is_letter(b: usize) -> bool {
    (b as u8).is_ascii_alphabetic() || b == b'\'' as usize
}

/// Position classes for the `classes=1` loss breakdown: whether the byte being read is a letter, and what kind of byte is predicted.
/// Column buckets for the line-end table: bytes since the last newline (the last bucket also holds "none in 256 bytes").
const COLUMN_EDGES: [usize; 7] = [20, 40, 56, 64, 68, 71, 74];
const CLASS_NAMES: [&str; 10] = [
    "after non-letter -> lower", "after non-letter -> UPPER", "after non-letter -> space", "after non-letter -> newline", "after non-letter -> other",
    "after letter -> lower", "after letter -> UPPER", "after letter -> space", "after letter -> newline", "after letter -> other",
];

fn class_of(input: usize, target: usize) -> usize {
    let kind = match target as u8 {
        b'a'..=b'z' => 0,
        b'A'..=b'Z' => 1,
        b' ' => 2,
        b'\n' => 3,
        _ => 4,
    };
    kind + 5 * is_letter(input) as usize
}

/// One scored position of `survey=1`: (input position, model CE, stack CE, the stack's three likeliest next bytes with probabilities).
type SurveyRow = (usize, f64, f64, [(u8, f32); 3]);

/// `survey=1`: the last spec's CE by what the target byte is. A letter target belongs to a word that is in the train lexicon,
/// or is not but occurred earlier in the same book, or is new to the book, and is split by how many letters of the word
/// precede it; other targets are split by kind. Also the ceiling for an in-document word cache: every "seen earlier" letter
/// target brought down to the mean CE of in-lexicon targets at the same word position (a bound, not a prediction).
/// For word-initial targets in the lexicon it also prints the CE by train frequency of the word and by how far back its
/// previous occurrence in the book is. `rows` are as `SurveyRow`; `n` is the number of scored positions.
fn survey_report(held_out: &[usize], docs: &[usize], lexicon: &std::collections::HashMap<Vec<u8>, u32>, rows: &[SurveyRow], n: f64) -> std::collections::HashMap<(String, &'static str), (usize, f64, f64)> {
    use std::collections::HashMap;
    let doc_of = |pos: usize| docs.partition_point(|&b| b <= pos) - 1;
    let word_at = |t: usize| -> (usize, usize) {
        let lo = docs[doc_of(t)];
        let (mut s, mut e) = (t, t);
        while s > lo && is_letter(held_out[s - 1]) {
            s -= 1;
        }
        while e < held_out.len() && is_letter(held_out[e]) {
            e += 1;
        }
        (s, e)
    };
    // Start of every occurrence of each word in each book, in order.
    let mut occ: HashMap<(usize, Vec<u8>), Vec<usize>> = HashMap::new();
    let mut i = 0;
    while i < held_out.len() {
        if is_letter(held_out[i]) && (i == 0 || !is_letter(held_out[i - 1]) || docs[doc_of(i)] == i) {
            let (s, e) = word_at(i);
            occ.entry((doc_of(s), held_out[s..e].iter().map(|&b| b as u8).collect())).or_default().push(s);
            i = e.max(i + 1);
        } else {
            i += 1;
        }
    }
    const LETTER_KINDS: [&str; 3] = ["in lexicon", "not in lexicon, seen earlier in book", "not in lexicon, new to book"];
    const POS: [&str; 3] = ["1st letter", "2nd-3rd letter", "4th+ letter"];
    let other_kind = |b: u8| match b {
        b' ' => "space",
        b'\n' => "newline",
        b'0'..=b'9' => "digit",
        b'.' | b',' | b';' | b':' | b'!' | b'?' | b'-' => "punctuation . , ; : ! ? -",
        b'"' | b'(' | b')' | b'[' | b']' | b'_' | b'*' => "quote / bracket / marks",
        _ => "other",
    };
    // key: (kind label, position label) -> (count, model CE sum, stack CE sum)
    let mut table: HashMap<(String, &'static str), (usize, f64, f64)> = HashMap::new();
    let mut word_starts: Vec<usize> = vec![]; // rows whose target starts a word in the lexicon
    let mut recency = [[(0usize, 0.0f64); 4]; 3]; // [train count <20, <200, more][previous occurrence <=500 bytes back, <=5000, farther, none in the book] -> (count, stack CE sum)
    for (ri, &(abs, cm, cs, _)) in rows.iter().enumerate() {
        let t = abs + 1;
        let target = held_out[t];
        let (kind, pos) = if is_letter(target) {
            let (s, e) = word_at(t);
            let word: Vec<u8> = held_out[s..e].iter().map(|&b| b as u8).collect();
            let kind = if lexicon.contains_key(&word) {
                0
            } else if occ[&(doc_of(s), word)][0] < s {
                1
            } else {
                2
            };
            (LETTER_KINDS[kind].to_string(), POS[match t - s { 0 => 0, 1 | 2 => 1, _ => 2 }])
        } else {
            (other_kind(target as u8).to_string(), "")
        };
        if kind == LETTER_KINDS[0] && pos == POS[0] {
            word_starts.push(ri);
            let (s, e) = word_at(abs + 1);
            let word: Vec<u8> = held_out[s..e].iter().map(|&b| b as u8).collect();
            let starts = &occ[&(doc_of(s), word.clone())];
            let back = starts.partition_point(|&o| o < s).checked_sub(1).map(|j| s - starts[j]);
            let by_recency = match back { Some(d) if d <= 500 => 0, Some(d) if d <= 5000 => 1, Some(_) => 2, None => 3 };
            let count = lexicon[&word];
            let by_freq = if count < 20 { 0 } else if count < 200 { 1 } else { 2 };
            let cell = &mut recency[by_freq][by_recency];
            *cell = (cell.0 + 1, cell.1 + cs);
        }
        let cell = table.entry((kind, pos)).or_default();
        *cell = (cell.0 + 1, cell.1 + cm, cell.2 + cs);
    }
    let total_stack: f64 = rows.iter().map(|r| r.2).sum();
    println!("survey: stack CE by target kind (positions {}, stack CE {:.4}); contrib = 1e-3 nats of overall CE", rows.len(), total_stack / n);
    println!("{:44} {:16} {:>7} {:>7} {:>8} {:>8} {:>8}", "kind", "word position", "share%", "modelCE", "stackCE", "contrib", "of CE%");
    let mut keys: Vec<_> = table.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        let (c, m, s) = table[k];
        println!("{:44} {:16} {:7.2} {:7.3} {:8.3} {:8.2} {:8.1}", k.0, k.1, 100.0 * c as f64 / n, m / c as f64, s / c as f64, 1e3 * s / n, 100.0 * s / total_stack);
    }
    println!("cache ceiling (seen-earlier letters brought down to the in-lexicon mean stack CE at the same word position):");
    let mut gain = 0.0;
    for pos in &POS {
        let lex = table.get(&(LETTER_KINDS[0].to_string(), *pos));
        let seen = table.get(&(LETTER_KINDS[1].to_string(), *pos));
        if let (Some(&(lc, _, ls)), Some(&(sc, _, ss))) = (lex, seen) {
            let g = (ss - sc as f64 * ls / lc as f64).max(0.0);
            println!("  {pos:16} {:>6} positions, mean stack CE {:.3} vs in-lexicon {:.3}: ceiling {:.2}e-3", sc, ss / sc as f64, ls / lc as f64, 1e3 * g / n);
            gain += g;
        }
    }
    println!("  total ceiling {:.2}e-3 nats ({:.2}% of the stack CE)", 1e3 * gain / n, 100.0 * gain / total_stack);
    println!("word-initial letters in the lexicon: stack CE by train count of the word and distance to its previous occurrence in the book (positions, mean CE, contrib 1e-3):");
    for (fi, fname) in ["train count <20", "train count 20-199", "train count >=200"].iter().enumerate() {
        for (ri, rname) in ["prev <=500 B back", "prev <=5000 B back", "prev farther", "none earlier in book"].iter().enumerate() {
            let (c, s) = recency[fi][ri];
            if c > 0 {
                println!("  {fname:20} {rname:22} {c:7} {:7.3} {:8.2}", s / c as f64, 1e3 * s / n);
            }
        }
    }
    // A fixed-stride sample of word-initial targets, with the stack's three likeliest bytes: for reading by hand.
    println!("sample of word-initial letters in the lexicon (context | target word: stack top3, stack CE, model CE):");
    let show = |b: usize| if b == b'\n' as usize { '~' } else { b as u8 as char };
    for &ri in word_starts.iter().step_by((word_starts.len() / 60).max(1)) {
        let (abs, cm, cs, top) = rows[ri];
        let (s, e) = word_at(abs + 1);
        let ctx: String = held_out[(abs + 1).saturating_sub(60)..=abs].iter().map(|&b| show(b)).collect();
        let word: String = held_out[s..e].iter().map(|&b| show(b)).collect();
        let guess: Vec<String> = top.iter().map(|&(b, q)| format!("{}:{:.2}", show(b as usize), q)).collect();
        println!("  {ctx:>60} | {word:<12} {} {cs:.2} {cm:.2}", guess.join(" "));
    }
    table
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

/// The train split's words with counts, as sorted lists for prefix lookup: all words, and the words after each word (for the bigram).
struct WordLists {
    unigram: Vec<(Vec<u8>, u32)>,
    after: std::collections::HashMap<Vec<u8>, Vec<(Vec<u8>, u32)>>,
}

impl WordLists {
    fn new(train: &[usize], bigrams: bool) -> WordLists {
        use std::collections::HashMap;
        let list = word_list(train);
        let mut uni: HashMap<Vec<u8>, u32> = HashMap::new();
        for w in &list {
            *uni.entry(w.clone()).or_default() += 1;
        }
        let mut bi: HashMap<Vec<u8>, HashMap<Vec<u8>, u32>> = HashMap::new();
        if bigrams {
            for pair in list.windows(2) {
                *bi.entry(pair[0].clone()).or_default().entry(pair[1].clone()).or_default() += 1;
            }
        }
        WordLists { unigram: sorted_counts(uni), after: bi.into_iter().map(|(k, v)| (k, sorted_counts(v))).collect() }
    }
}

/// The word list masks next bytes that cannot continue or end a known word.
struct Lexicon {
    lists: Rc<WordLists>,
    eps: f64,
    hist: bool,
    positions: usize,
    constrained: usize,
    top1_rejected: usize,
    top1_fixed: usize,
}

impl Lexicon {
    fn new(lists: Rc<WordLists>, eps: f64, hist: bool) -> Lexicon {
        Lexicon { lists, eps, hist, positions: 0, constrained: 0, top1_rejected: 0, top1_fixed: 0 }
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
        // A letter may follow if some word extends the prefix by it; any other byte only if the prefix is a whole word.
        let (letters, end, _) = Words::extend(&self.lists.unigram, &word);
        let mask: Vec<bool> = (0..VOCAB).map(|b| if is_letter(b) { letters[b] > 0.0 } else { end > 0.0 }).collect();
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
            self.lists.unigram.len(),
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
    lists: Rc<WordLists>,
    positions: usize,
    applied: usize,
    applied_bigram: usize,
    applied_first: usize,
    dump: Option<Vec<f32>>,
    chunk: usize,
}

fn sorted_counts(m: std::collections::HashMap<Vec<u8>, u32>) -> Vec<(Vec<u8>, u32)> {
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort();
    v
}

impl Words {
    fn new(lists: Rc<WordLists>, order: usize, lambda: f64, hist: bool, first: bool) -> Words {
        assert!(order <= 1, "words:<order 0|1>:<lambda>");
        Words { order, lambda, hist, first, lists, positions: 0, applied: 0, applied_bigram: 0, applied_first: 0, dump: None, chunk: 0 }
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

    /// The mixing step: `Some((q_target, count mass, bigram list used, prefix length))` when the word lists changed
    /// `probs` inside a word, `None` otherwise.
    fn mix(&mut self, ctx: &Ctx, probs: &mut [f64]) -> Option<(f64, f64, bool, usize)> {
        self.positions += 1;
        let seq = if self.hist { ctx.context } else { ctx.window };
        let Some(boundary) = seq.iter().rposition(|&b| !is_letter(b)) else { return None };
        let prefix: Vec<u8> = seq[boundary + 1..].iter().map(|&b| b as u8).collect();
        if prefix.is_empty() && !self.first {
            return None;
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
            if let Some(list) = self.lists.after.get(&prev) {
                let e = Words::extend(list, &prefix);
                if e.2 > 0.0 {
                    found = Some(e);
                    self.applied_bigram += 1;
                }
            }
        }
        let bigram_found = found.is_some();
        let (letters, end, total) = found.unwrap_or_else(|| Words::extend(&self.lists.unigram, &prefix));
        if prefix.is_empty() {
            // Between words: the model keeps its split between letters and other bytes; the words that followed the
            // previous word redistribute the letter part over first letters.
            let letter_mass: f64 = (0..VOCAB).filter(|&b| is_letter(b)).map(|b| probs[b]).sum();
            if total <= 0.0 || letter_mass <= 0.0 {
                return None;
            }
            self.applied_first += 1;
            for b in (0..VOCAB).filter(|&b| is_letter(b)) {
                probs[b] = (1.0 - self.lambda) * probs[b] + self.lambda * letter_mass * letters[b] / total;
            }
            return None;
        }
        let non_letter: f64 = (0..VOCAB).filter(|&b| !is_letter(b)).map(|b| probs[b]).sum();
        if total <= 0.0 || non_letter <= 0.0 {
            return None;
        }
        self.applied += 1;
        let mut q_t = 0.0;
        for b in 0..VOCAB {
            let q = if is_letter(b) { letters[b] / total } else { end / total * probs[b] / non_letter };
            if b == ctx.target {
                q_t = q;
            }
            probs[b] = (1.0 - self.lambda) * probs[b] + self.lambda * q;
        }
        Some((q_t, total, bigram_found, prefix.len()))
    }
}

impl Tier for Words {
    fn adjust(&mut self, ctx: &Ctx, probs: &mut [f64]) {
        if self.dump.is_none() {
            self.mix(ctx, probs);
            return;
        }
        let p_in = probs[ctx.target];
        let h_in = -probs.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f64>();
        let applied = self.mix(ctx, probs);
        let row = match applied {
            Some((q, total, bigram, prefix_len)) => {
                [self.chunk as f32, 1.0, p_in as f32, q as f32, h_in as f32, (1.0 + total).ln() as f32, bigram as u8 as f32, prefix_len as f32]
            }
            None => [self.chunk as f32, 0.0, p_in as f32, p_in as f32, h_in as f32, 0.0, 0.0, 0.0],
        };
        self.dump.as_mut().unwrap().extend(row);
    }

    fn prepare(&mut self, chunk: usize, _hidden: &[f32], _rows: usize, _starts: &[usize]) {
        self.chunk = chunk;
    }

    fn wdump(&self, path: &str, last: bool) {
        if let Some(d) = &self.dump {
            let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
            let tmp = format!("{path}.tmp");
            std::fs::write(&tmp, bytes).unwrap_or_else(|e| panic!("writing {tmp}: {e}"));
            std::fs::rename(&tmp, path).unwrap_or_else(|e| panic!("renaming {tmp} to {path}: {e}"));
            if last {
                println!("wdump: {} rows to {path}", d.len() / 8);
            }
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

/// Bytes since the last newline in `seq` (0: the last byte is a newline), None if it holds none.
fn column(seq: &[usize]) -> Option<usize> {
    seq.iter().rposition(|&b| b == b'\n' as usize).map(|i| seq.len() - 1 - i)
}

/// The line-wrap tier (KR&R): the books are hard-wrapped at ~72 bytes, longer than the model's 64-byte window, so where
/// the window shows no newline the model does not know the column. From the train text, the rate of a newline (rather than a
/// space) at each column, separately after a letter and after another byte; where the window has no newline but the
/// 256-byte context has one, the newline/space split of the model's mass on those two bytes is mixed with that rate.
struct Wrap {
    lambda: f64,
    /// [after a letter][column]: (newlines, spaces) that followed.
    counts: [Vec<(u32, u32)>; 2],
    positions: usize,
    applied: usize,
}

const WRAP_COLUMNS: usize = 128;

impl Wrap {
    fn new(train: &[usize], lambda: f64) -> Wrap {
        let mut counts = [vec![(0u32, 0u32); WRAP_COLUMNS], vec![(0u32, 0u32); WRAP_COLUMNS]];
        let mut last_nl: Option<usize> = None;
        for j in 0..train.len() - 1 {
            if train[j] == b'\n' as usize {
                last_nl = Some(j);
            }
            let Some(c) = last_nl.map(|i| j - i) else { continue };
            let cell = &mut counts[is_letter(train[j]) as usize][c.min(WRAP_COLUMNS - 1)];
            match train[j + 1] as u8 {
                b'\n' => cell.0 += 1,
                b' ' => cell.1 += 1,
                _ => {}
            }
        }
        Wrap { lambda, counts, positions: 0, applied: 0 }
    }

    /// P(newline | newline or space follows) at this column, smoothed towards one half.
    fn newline_rate(&self, after_letter: bool, column: usize) -> f64 {
        let (nl, sp) = self.counts[after_letter as usize][column.min(WRAP_COLUMNS - 1)];
        (nl as f64 + 0.5) / (nl as f64 + sp as f64 + 1.0)
    }
}

impl Tier for Wrap {
    fn adjust(&mut self, ctx: &Ctx, probs: &mut [f64]) {
        self.positions += 1;
        if ctx.window.contains(&(b'\n' as usize)) {
            return;
        }
        let Some(c) = column(ctx.context) else { return };
        let mass = probs[b'\n' as usize] + probs[b' ' as usize];
        if mass <= 0.0 {
            return;
        }
        self.applied += 1;
        let q = self.newline_rate(is_letter(*ctx.window.last().unwrap()), c);
        probs[b'\n' as usize] = (1.0 - self.lambda) * probs[b'\n' as usize] + self.lambda * mass * q;
        probs[b' ' as usize] = (1.0 - self.lambda) * probs[b' ' as usize] + self.lambda * mass * (1.0 - q);
    }
    fn report(&self) -> String {
        format!("wrap: applied (window shows no newline, context does) at {} of {} positions", self.applied, self.positions)
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
    /// Per searched chunk, per row the KMAX nearest (squared distance, value), ascending; online, the train list then
    /// the in-document list.
    found: RefCell<std::collections::HashMap<usize, Vec<(f32, u16)>>>,
    /// `ivf=<cells>:<probes>`: the train keys are searched approximately (`KnnStore::search_ivf`), probing this many cells.
    nprobe: Option<usize>,
    /// `ivf_check=<n>`: on the first batch, report how many of the exact 128 nearest train keys the approximate search finds
    /// for n queries.
    check: Cell<usize>,
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

/// A memory written as the text is read: each chunk's scored positions are appended before its search, and a query sees
/// the train keys and, from its own book, only the keys of earlier windows.
struct Online {
    n_train: usize,
    /// Where each book's held-out text starts (bytes), and the first key written from it (None until then).
    docs: Vec<usize>,
    doc_first_key: Vec<Option<usize>>,
}

impl Store {
    /// The online memory searches its train keys and its in-document keys separately, KMAX nearest each.
    fn split(&self) -> bool {
        self.online.is_some()
    }

    /// Entries per row in `found`.
    fn stride(&self) -> usize {
        if self.split() { 2 * KMAX } else { KMAX }
    }

    /// Searches the queries of several chunks together (bigger batches search faster per query) and files each chunk's
    /// neighbour lists under its id in `found`. Chunks must be written (`write_chunk`) before they are searched.
    fn search_batch(&self, items: &[Pending]) {
        if items.is_empty() {
            return;
        }
        let d = items[0].hidden.len() / items[0].rows;
        let (mut q, mut ranges, mut who): (Vec<f32>, Vec<(u32, u32)>, Vec<(usize, usize)>) = (vec![], vec![], vec![]);
        for (i, it) in items.iter().enumerate() {
            // Only the scored rows (window position >= warm) are searched; the others keep (infinity, 0) entries.
            for r in (0..it.rows).filter(|r| r % seq_len() >= self.warm) {
                q.extend_from_slice(&it.hidden[r * d..(r + 1) * d]);
                who.push((i, r));
                let w = it.starts[r / seq_len()];
                if let Some(c) = &self.causal {
                    let (w, first) = (w / seq_len(), c.first_window[w / seq_len()]);
                    let first = first.max(w.saturating_sub(c.recent));
                    ranges.push(((c.n_train + first * seq_len()) as u32, (c.n_train + w * seq_len()) as u32));
                }
                if let Some(on) = &self.online {
                    let on = on.borrow();
                    let doc = on.docs.partition_point(|&b| b <= w) - 1;
                    let hi = (it.base + r / seq_len() * (seq_len() - self.warm)) as u32;
                    ranges.push((on.doc_first_key[doc].map_or(hi, |k| k as u32), hi));
                }
            }
        }
        let n = who.len();
        let gpu = self.gpu.borrow();
        let stride = self.stride();
        let mut lists = vec![(f32::INFINITY, 0u16); n * stride];
        if self.split() {
            // Per-source lists: the KMAX nearest train keys, then the KMAX nearest in-document keys (the last keys, so only
            // the tail tiles are scanned).
            let n_train = self.online.as_ref().unwrap().borrow().n_train as u32;
            let train = match self.nprobe {
                Some(p) => gpu.search_ivf(&q, n, p),
                None => gpu.search_tagged(&q, n, Some((&vec![(0, 0); n], n_train))),
            };
            if let (Some(_), m @ 1..) = (self.nprobe, self.check.replace(0).min(n)) {
                let exact = gpu.search_tagged(&q[..m * d], m, Some((&vec![(0, 0); m], n_train)));
                let hits: usize = (0..m).map(|j| (0..128).filter(|&i| train[j * KMAX + i].0 <= exact[j * KMAX + 127].0 * (1.0 + 2e-3)).count()).sum();
                println!("ivf recall: {:.4} of the exact 128 nearest train keys, over {m} queries", hits as f64 / (128 * m) as f64);
            }
            let doc = gpu.search_tail(&q, n, &ranges, n_train as usize);
            for j in 0..n {
                lists[j * stride..j * stride + KMAX].copy_from_slice(&train[j * KMAX..(j + 1) * KMAX]);
                lists[j * stride + KMAX..(j + 1) * stride].copy_from_slice(&doc[j * KMAX..(j + 1) * KMAX]);
            }
        } else {
            let mask = self.causal.as_ref().map(|c| (ranges.as_slice(), c.n_train as u32));
            lists = gpu.search_tagged(&q, n, mask);
        }
        drop(gpu);
        let mut found = self.found.borrow_mut();
        for it in items {
            found.insert(it.chunk, vec![(f32::INFINITY, 0u16); it.rows * stride]);
        }
        for (j, &(i, r)) in who.iter().enumerate() {
            found.get_mut(&items[i].chunk).unwrap()[r * stride..(r + 1) * stride].copy_from_slice(&lists[j * stride..(j + 1) * stride]);
        }
    }

    /// Online memory only: appends the chunk's scored positions (window position >= warm) as new keys; returns the index of
    /// the first one.
    fn write_chunk(&self, hidden: &[f32], targets: &[usize], starts: &[usize], warm: usize, d: usize) -> usize {
        let Some(on) = &self.online else { return 0 };
        let (mut on, mut gpu) = (on.borrow_mut(), self.gpu.borrow_mut());
        let base = gpu.len();
        for (j, &s) in starts.iter().enumerate() {
            let doc = on.docs.partition_point(|&b| b <= s) - 1;
            if on.doc_first_key[doc].is_none() {
                on.doc_first_key[doc] = Some(gpu.len());
            }
            let rows = j * seq_len() + warm..(j + 1) * seq_len();
            // `len()` counts keys not yet uploaded, so no flush per window (that made one 32-key device tile per window).
            gpu.add_doc(&hidden[rows.start * d..rows.end * d], &targets[rows].iter().map(|&t| t as u8).collect::<Vec<_>>());
        }
        base
    }
}

/// A chunk waiting for its neighbour search: its id, hidden states ([rows, d]), window starts, and (online memory) the
/// index of the first key written from it.
struct Pending<'a> {
    chunk: usize,
    hidden: &'a [f32],
    rows: usize,
    starts: &'a [usize],
    base: usize,
}

struct Knn {
    store: Rc<Store>,
    k: usize,
    lambda: f64,
    temp: f64,
    /// In-document neighbours (keys written from the scored text's own book) use this temperature and have `shift` taken
    /// off their squared distance before weighting; `temp_doc == temp` and `shift == 0` is the plain tier.
    temp_doc: f64,
    shift: f64,
    positions: usize,
    nearest: f64,
    kth: f64,
    /// Neighbours seen and how many of them were in-document keys.
    seen: usize,
    seen_doc: usize,
    dump: Option<Vec<f32>>,
    chunk: usize,
}

impl Tier for Knn {
    fn needs_hidden(&self) -> bool {
        true
    }
    fn prepare(&mut self, chunk: usize, _hidden: &[f32], _rows: usize, _starts: &[usize]) {
        self.chunk = chunk;
    }
    fn adjust(&mut self, ctx: &Ctx, probs: &mut [f64]) {
        let found = self.store.found.borrow();
        let stride = self.store.stride();
        let row = &found[&self.chunk][ctx.row * stride..(ctx.row + 1) * stride];
        // A causal memory can have fewer than k keys to offer (the start of a book): use those, or leave p alone.
        let finite = |l: &[(f32, u16)]| l[..self.k].iter().take_while(|n| n.0.is_finite()).count();
        let merged: Vec<(f32, u16)>;
        let nb = if self.store.split() {
            // The k nearest of each source, merged by distance.
            let mut m = row[..finite(&row[..KMAX])].to_vec();
            m.extend_from_slice(&row[KMAX..KMAX + finite(&row[KMAX..])]);
            m.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            merged = m;
            &merged[..]
        } else {
            &row[..finite(row)]
        };
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
        let adj = |&(dist, v): &(f32, u16)| dist as f64 - if v >= DOC { self.shift } else { 0.0 };
        let reference = nb.iter().map(adj).fold(f64::INFINITY, f64::min);
        for n in nb {
            let doc = n.1 >= DOC;
            let w = (-(adj(n) - reference) / if doc { self.temp_doc } else { self.temp }).exp();
            knn[(n.1 % DOC) as usize] += w;
            total += w;
            self.seen_doc += doc as usize;
        }
        self.seen += nb.len();
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
        format!(
            "knn ({} keys): mean squared distance to nearest {:.1}, to k-th {:.1}; {:.1}% of the neighbours are in-document keys",
            self.store.gpu.borrow().len(),
            self.nearest / n,
            self.kth / n,
            100.0 * self.seen_doc as f64 / self.seen.max(1) as f64
        )
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
fn build_tier(spec: &str, train: &[usize], store: &Option<Rc<Store>>, dump: bool, wdump: bool) -> Vec<Box<dyn Tier>> {
    // One word table for the lexicon and words parts; bigrams only if a words part asks for them.
    let bigrams = spec.split('+').any(|p| p.strip_prefix("words:").map_or(false, |a| a.starts_with('1')));
    let lists = spec.split('+').any(|p| p.starts_with("lexicon") || p.starts_with("words")).then(|| Rc::new(WordLists::new(train, bigrams)));
    spec.split('+')
        .map(|part| {
            let (kind, arg) = part.split_once(':').unwrap_or((part, ""));
            match kind {
                "wrap" => Box::new(Wrap::new(train, arg.parse().unwrap_or_else(|_| panic!("wrap:<lambda>, got {part}")))) as Box<dyn Tier>,
                "lexicon" => {
                    let (eps, flags) = arg.split_once(':').unwrap_or((arg, ""));
                    let eps = eps.parse().unwrap_or_else(|_| panic!("lexicon:<eps>[:h], got {part}"));
                    Box::new(Lexicon::new(lists.clone().unwrap(), eps, flags.contains('h'))) as Box<dyn Tier>
                }
                "words" => {
                    let a: Vec<&str> = arg.split(':').collect();
                    assert!(a.len() >= 2 && a.len() <= 3, "words:<order>:<lambda>[:<flags h f>], got {part}");
                    let flags = a.get(2).copied().unwrap_or("");
                    assert!(!(wdump && flags.contains('f')), "wdump does not support first letters");
                    let mut w = Words::new(lists.clone().unwrap(), a[0].parse().unwrap(), a[1].parse().unwrap(), flags.contains('h'), flags.contains('f'));
                    w.dump = wdump.then(Vec::new);
                    Box::new(w) as Box<dyn Tier>
                }
                "knn" => {
                    let a: Vec<f64> = arg.split(':').map(|x| x.parse().unwrap_or_else(|_| panic!("knn:<k>:<lambda>:<temp>[:<temp_doc>:<shift>], got {part}"))).collect();
                    assert!((a.len() == 3 || a.len() == 5) && a[0] >= 1.0 && a[0] as usize <= KMAX, "knn:<k>:<lambda>:<temp>[:<temp_doc>:<shift>] with 1 <= k <= {KMAX}, got {part}");
                    let store = store.clone().expect("a knn tier needs the store");
                    let (temp_doc, shift) = if a.len() == 5 { (a[3], a[4]) } else { (a[2], 0.0) };
                    Box::new(Knn { store, k: a[0] as usize, lambda: a[1], temp: a[2], temp_doc, shift, positions: 0, nearest: 0.0, kth: 0.0, seen: 0, seen_doc: 0, dump: dump.then(Vec::new), chunk: 0 }) as Box<dyn Tier>
                }
                _ => panic!("unknown tier {kind} in {spec}"),
            }
        })
        .collect()
}

fn from_is_train(v: &str) -> bool {
    v == "train"
}

/// `generate=<len>`: samples `len` bytes after a prompt (the held-out text from byte `prompt`), once from the model alone and once
/// through each `tier=` spec (lexicon and words only: they read the text so far; a knn tier would need a search per byte). The same
/// seed drives every run, so the outputs are comparable; `temp=0` is greedy. The true continuation is printed last.
fn generate(dev: &DeviceParams, cfg: &Config, train: &[usize], held_out: &[usize], specs: &[String], len: usize, prompt: usize, temp: f64, seed: u64) {
    let seq = seq_len();
    let show = |ids: &[usize]| String::from_utf8_lossy(&ids.iter().map(|&b| b as u8).collect::<Vec<_>>()).replace('\n', " ");
    println!("PROMPT: {}", show(&held_out[prompt..prompt + seq]));
    let mut costs: Vec<Vec<f64>> = vec![];
    for spec in std::iter::once(None).chain(specs.iter().map(Some)) {
        let mut tiers: Vec<Box<dyn Tier>> = spec.map_or(vec![], |s| build_tier(s, train, &None, false, false));
        let mut rng = scratchtape::nn::Rng::new(seed);
        let mut text: Vec<usize> = held_out[prompt..prompt + seq].to_vec();
        for _ in 0..len {
            let probs = next_probs(dev, cfg, &text, &mut tiers);
            let next = if temp == 0.0 {
                (0..VOCAB).max_by(|&a, &b| probs[a].total_cmp(&probs[b])).unwrap()
            } else {
                let w: Vec<f64> = probs.iter().map(|p| p.powf(1.0 / temp)).collect();
                let mut u = rng.next_f32() as f64 * w.iter().sum::<f64>();
                w.iter().position(|&x| { u -= x; u < 0.0 }).unwrap_or(VOCAB - 1)
            };
            text.push(next);
        }
        println!("{:<44}{}", spec.map_or("MODEL:".to_string(), |s| format!("{s}:")), show(&text[seq..]));
        // The same tiers on the true continuation (teacher-forced): the nats each byte costs.
        let mut tiers: Vec<Box<dyn Tier>> = spec.map_or(vec![], |s| build_tier(s, train, &None, false, false));
        costs.push((0..len).map(|i| -next_probs(dev, cfg, &held_out[prompt..prompt + seq + i], &mut tiers)[held_out[prompt + seq + i]].ln()).collect());
    }
    println!("ACTUAL: {}", show(&held_out[prompt + seq..prompt + seq + len]));
    for (k, spec) in specs.iter().enumerate() {
        let (base, c) = (&costs[0], &costs[k + 1]);
        println!("CE on ACTUAL: model {:.4}, {spec} {:.4}", base.iter().sum::<f64>() / len as f64, c.iter().sum::<f64>() / len as f64);
        let mut order: Vec<usize> = (0..len).collect();
        order.sort_by(|&a, &b| (c[a] - base[a]).total_cmp(&(c[b] - base[b])));
        for (name, picks) in [("helped most", &order[..5]), ("hurt most", &order[len - 5..])] {
            let list: Vec<String> = picks.iter().map(|&i| {
                let at = prompt + seq + i;
                format!("{:+.2} '{}'->'{}'", c[i] - base[i], show(&held_out[at.saturating_sub(12)..at]), show(&held_out[at..at + 1]))
            }).collect();
            println!("  {name}: {}", list.join(" | "));
        }
    }
}

/// Word classes for the pooled breakdowns (`prompts=`, `classes=1`): a non-letter byte, or a letter of a word that is lowercase,
/// Capitalised at a sentence start (after `. ! ?` or a line break) or Capitalised mid-sentence (mostly names), each split by how
/// many times the word occurs in the train text (0, 1-9, 10-99, 100+).
struct WordClasses {
    counts: std::collections::HashMap<Vec<u8>, u32>,
}

impl WordClasses {
    const NONLETTER: [&'static str; 7] = ["word-end space", "other space", "newline", ". ! ?", ", ; :", "straight quote", "other"];
    const COUNT: usize = 7 + 12;

    fn new(train: &[usize]) -> Self {
        let mut counts = std::collections::HashMap::<Vec<u8>, u32>::new();
        let mut i = 0;
        while i < train.len() {
            if (train[i] as u8).is_ascii_alphabetic() {
                let w = Self::word(train, i);
                i += w.len();
                *counts.entry(w).or_default() += 1;
            } else {
                i += 1;
            }
        }
        WordClasses { counts }
    }

    fn names() -> Vec<String> {
        Self::NONLETTER.iter().map(|s| s.to_string())
            .chain(["lower", "Cap-initial", "Cap-mid"].iter().flat_map(|k| ["/0", "/1-9", "/10-99", "/100+"].map(|b| format!("{k}{b}"))))
            .collect()
    }

    /// The lowercased ASCII letters around `text[i]`.
    fn word(text: &[usize], i: usize) -> Vec<u8> {
        let letter = |b: usize| (b as u8).is_ascii_alphabetic();
        let (mut a, mut b) = (i, i + 1);
        while a > 0 && letter(text[a - 1]) {
            a -= 1;
        }
        while b < text.len() && letter(text[b]) {
            b += 1;
        }
        text[a..b].iter().map(|&c| (c as u8).to_ascii_lowercase()).collect()
    }

    /// The class of the byte `text[at]`.
    fn class(&self, text: &[usize], at: usize) -> usize {
        if !(text[at] as u8).is_ascii_alphabetic() {
            let after_letter = at > 0 && (text[at - 1] as u8).is_ascii_alphabetic();
            return match text[at] as u8 {
                b' ' if after_letter => 0,
                b' ' => 1,
                b'\n' => 2,
                b'.' | b'!' | b'?' => 3,
                b',' | b';' | b':' => 4,
                b'"' | b'\'' => 5,
                _ => 6,
            };
        }
        let c = self.counts.get(&Self::word(text, at)).copied().unwrap_or(0);
        let mut a = at;
        while a > 0 && (text[a - 1] as u8).is_ascii_alphabetic() {
            a -= 1;
        }
        let sentence_start = (0..a).rev().find_map(|j| match text[j] as u8 {
            b'.' | b'!' | b'?' | b'\n' => Some(true),
            c if c.is_ascii_alphanumeric() || b",;:".contains(&c) => Some(false),
            _ => None,
        });
        let bucket = (c >= 1) as usize + (c >= 10) as usize + (c >= 100) as usize;
        let kind = if !(text[a] as u8).is_ascii_uppercase() { 0 } else if sentence_start.unwrap_or(true) { 1 } else { 2 };
        7 + 4 * kind + bucket
    }
}

/// Prints each spec's saving against the model by word class: `bytes[c]` scored bytes, `cost[0][c]` the model's nats and
/// `cost[i + 1][c]` spec i's.
fn print_word_classes(bytes: &[usize], cost: &[Vec<f64>], specs: &[String]) {
    let names = WordClasses::names();
    let total = bytes.iter().sum::<usize>() as f64;
    println!("{total} scored bytes; model {:.4} nats/byte", cost[0].iter().sum::<f64>() / total);
    for (si, spec) in specs.iter().enumerate() {
        let saved: f64 = (0..bytes.len()).map(|c| cost[0][c] - cost[si + 1][c]).sum();
        println!("{spec}: {:.4} nats/byte ({:+.4})", cost[si + 1].iter().sum::<f64>() / total, -saved / total);
        for c in 0..bytes.len() {
            let (d, nc) = (cost[0][c] - cost[si + 1][c], bytes[c].max(1) as f64);
            println!("  {:<18} {:>6} bytes  model {:.3}  change {:+.4}/byte  share of saving {:>5.1}%", names[c], bytes[c], cost[0][c] / nc, -d / nc, 100.0 * d / saved);
        }
    }
}

/// `prompts=<n> generate=<len>`: the teacher-forced cost of `len` bytes after each of `n` evenly spaced held-out prompts, per tier
/// spec against the model alone, pooled by `WordClasses`.
fn pooled(dev: &DeviceParams, cfg: &Config, train: &[usize], held_out: &[usize], specs: &[String], n: usize, len: usize) {
    let seq = seq_len();
    let wc = WordClasses::new(train);
    let starts: Vec<usize> = (0..n).map(|k| 1000 + k * (held_out.len() - seq - len - 2000) / n).collect();
    let mut cost = vec![vec![0.0f64; WordClasses::COUNT]; 1 + specs.len()];
    let mut bytes = vec![0usize; WordClasses::COUNT];
    for (si, spec) in std::iter::once(None).chain(specs.iter().map(Some)).enumerate() {
        for &p in &starts {
            let mut tiers: Vec<Box<dyn Tier>> = spec.map_or(vec![], |s| build_tier(s, train, &None, false, false));
            for i in 0..len {
                let c = wc.class(held_out, p + seq + i);
                if si == 0 {
                    bytes[c] += 1;
                }
                cost[si][c] -= next_probs(dev, cfg, &held_out[p..p + seq + i], &mut tiers)[held_out[p + seq + i]].ln();
            }
        }
    }
    println!("{n} prompts x {len} bytes");
    print_word_classes(&bytes, &cost, specs);
}

/// The next-byte distribution after `text` (its last `seq_len()` bytes are the window), through `tiers`.
fn next_probs(dev: &DeviceParams, cfg: &Config, text: &[usize], tiers: &mut [Box<dyn Tier>]) -> Vec<f64> {
    let seq = seq_len();
    let window = &text[text.len() - seq..];
    let mut data = window.to_vec();
    data.push(0); // the target slot of the last row, unused
    let mut probs = softmax(&forward(dev, cfg, &data, &[0], 0, false, true).logits[(seq - 1) * VOCAB..seq * VOCAB]);
    let context = &text[text.len().saturating_sub(CONTEXT)..];
    for tier in tiers.iter_mut() {
        tier.adjust(&Ctx { row: seq - 1, window, context, target: 0, hidden: &[] }, &mut probs);
    }
    probs
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

/// The input bytes and the bytes they predict, for the windows of `data` starting at `starts`.
fn window_ids(data: &[usize], starts: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let (mut ids, mut targets) = (vec![], vec![]);
    for &s in starts {
        ids.extend(&data[s..s + seq_len()]);
        targets.extend(&data[s + 1..s + seq_len() + 1]);
    }
    (ids, targets)
}

fn forward(dev: &DeviceParams, cfg: &Config, data: &[usize], starts: &[usize], tap: usize, want_hidden: bool, want_logits: bool) -> Forward {
    let (ids, targets) = window_ids(data, starts);
    let mut dt = DeviceTape::new(dev);
    let (outs, logits, loss) = model_forward(&mut dt, cfg, &ids, &targets, starts.len());
    // Everything the host needs comes back in one read.
    let hidden_h = match (WHITEN.get(), want_hidden) {
        (Some((mu, w)), true) => Some(whiten_device(dt.value(outs[tap]), ids.len(), mu, w, cfg.d)),
        (None, true) => Some(dt.value(outs[tap]).clone()),
        _ => None,
    };
    let mut handles = vec![];
    if want_logits {
        handles.extend([dt.value(logits).clone(), dt.row_losses(loss).clone()]);
    }
    handles.extend(hidden_h);
    let mut bytes = read_many(handles).into_iter();
    let (logits, row_loss) = if want_logits { (bytes.next().unwrap(), bytes.next().unwrap()) } else { (vec![], vec![]) };
    let hidden = bytes.next().unwrap_or_default();
    Forward { ids, targets, logits, row_loss, hidden }
}

/// Test-time optimiser for `dyn=<lr>`: plain SGD on the device (the training step, without dropout), or, with `dyn_rms=<beta>` /
/// `dyn_pull=<f>`, a host-side step (a prototype: reads the gradients and parameters back, updates, uploads) that normalises each
/// gradient by a bias-corrected running RMS and moves every parameter the fraction `f` of the way back to the checkpoint.
struct Adapter {
    lr: f32,
    rms_beta: Option<f32>,
    pull: f32,
    start: Vec<f32>,
    /// Parameters that adapt (`dyn_layers=`); None for all.
    mask: Option<Vec<bool>>,
    v: Vec<f32>,
    t: i32,
}

impl Adapter {
    fn new(lr: f32, rms_beta: Option<f32>, pull: f32, start: &[f32], mask: Option<Vec<bool>>) -> Self {
        Adapter { lr, rms_beta, pull, start: start.to_vec(), mask, v: vec![0.0; start.len()], t: 0 }
    }

    /// A new book: the optimiser state starts again (the caller re-uploads the checkpoint).
    fn reset(&mut self) {
        self.v.fill(0.0);
        self.t = 0;
    }

    fn step(&mut self, dev: &mut DeviceParams, cfg: &Config, data: &[usize], starts: &[usize]) {
        let (mut ids, mut targets) = (vec![], vec![]);
        for &s in starts {
            ids.extend(&data[s..s + seq_len()]);
            targets.extend(&data[s + 1..s + seq_len() + 1]);
        }
        dev.zero_grads();
        let mut dt = DeviceTape::new(dev);
        let (_, _, loss) = model_forward(&mut dt, cfg, &ids, &targets, starts.len());
        dt.backward(loss);
        drop(dt);
        if self.rms_beta.is_none() && self.pull == 0.0 && self.mask.is_none() {
            dev.sgd(self.lr);
            return;
        }
        let (g, mut p) = (dev.read(&dev.grads), dev.read(&dev.params));
        self.t += 1;
        let (beta, corr) = match self.rms_beta {
            Some(b) => (b, 1.0 - b.powi(self.t)),
            None => (0.0, 1.0),
        };
        for i in 0..p.len() {
            if self.mask.as_ref().is_some_and(|m| !m[i]) {
                continue;
            }
            let step = match self.rms_beta {
                Some(_) => {
                    self.v[i] = beta * self.v[i] + (1.0 - beta) * g[i] * g[i];
                    self.lr * g[i] / ((self.v[i] / corr).sqrt() + 1e-8)
                }
                None => self.lr * g[i],
            };
            p[i] = p[i] - step + self.pull * (self.start[i] - p[i]);
        }
        *dev = DeviceParams::upload(&p);
    }
}

/// Train windows embedded by the frozen model (mean hidden state over the scored rows), searched by brute force on the host.
struct NnIndex {
    starts: Vec<usize>,
    emb: Vec<f32>,
    d: usize,
}

impl NnIndex {
    /// The mean over each window's rows at window position >= `warm` of `hidden` ([windows * seq_len, d]).
    fn embed(hidden: &[f32], d: usize, warm: usize) -> Vec<f32> {
        let rows = hidden.len() / d;
        let mut out = vec![0.0f32; rows / seq_len() * d];
        for (w, o) in out.chunks_mut(d).enumerate() {
            for t in warm..seq_len() {
                for (a, b) in o.iter_mut().zip(&hidden[(w * seq_len() + t) * d..(w * seq_len() + t + 1) * d]) {
                    *a += b;
                }
            }
            o.iter_mut().for_each(|a| *a /= (seq_len() - warm) as f32);
        }
        out
    }

    /// Indices of the `k` embeddings nearest (squared L2) to `q`, nearest first.
    fn nearest(&self, q: &[f32], k: usize) -> Vec<usize> {
        let mut best: Vec<(f32, usize)> = vec![];
        for (j, e) in self.emb.chunks(self.d).enumerate() {
            let dist: f32 = e.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
            if best.len() < k || dist < best[best.len() - 1].0 {
                let at = best.partition_point(|&(x, _)| x <= dist);
                best.insert(at, (dist, j));
                best.truncate(k);
            }
        }
        best.into_iter().map(|(_, j)| j).collect()
    }
}

/// `whiten=<alpha>`: keys and queries are centred and multiplied by Sigma^(-alpha/2) (symmetric, total variance kept, so
/// distances keep their scale), Sigma estimated from a sample of train positions. Set once before the memory is built.
static WHITEN: std::sync::OnceLock<(Vec<f32>, Vec<f32>)> = std::sync::OnceLock::new();

/// The tap hidden states `x` ([rows, d]) as (x - mu) W^T on the device, in f32 so it equals the host formula to rounding; only the
/// whitened rows cross the link, with no host matmul.
fn whiten_device(x: &Handle, rows: usize, mu: &[f32], w: &[f32], d: usize) -> Handle {
    // The matrix and bias go up once: the whitener is fixed, and each upload crosses the link.
    thread_local!(static UPLOADED: std::cell::OnceCell<(Handle, Handle)> = const { std::cell::OnceCell::new() });
    let (wd, cd) = UPLOADED.with(|u| {
        u.get_or_init(|| {
            let c: Vec<f32> = w.chunks(d).map(|row| -row.iter().zip(mu).map(|(a, b)| a * b).sum::<f32>()).collect();
            (upload_f32(w), upload_f32(&c))
        })
        .clone()
    });
    let out = client().empty(rows * d * 4);
    let epi = Epilogue { bias: Some((&cd, 0)), ..Epilogue::default() };
    matmul_f32(MatRef::new(x), MatRef { trans: true, ..MatRef::new(&wd) }, MatRef::new(&out), 1, rows, d, d, epi);
    out
}

/// Cyclic Jacobi eigendecomposition of the symmetric d x d matrix `a`: (eigenvalues, V) with a = V diag(lam) V^T, V row-major.
fn jacobi_eigen(mut a: Vec<f64>, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    for _ in 0..30 {
        let off: f64 = (0..d).flat_map(|i| (0..d).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i * d + j].powi(2)).sum();
        if off < 1e-18 * (0..d).map(|i| a[i * d + i].powi(2)).sum::<f64>() {
            break;
        }
        for p in 0..d - 1 {
            for q in p + 1..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                let t = if theta == 0.0 { 1.0 } else { theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt()) };
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..d {
                    let (akp, akq) = (a[k * d + p], a[k * d + q]);
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for k in 0..d {
                    let (apk, aqk) = (a[p * d + k], a[q * d + k]);
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                for k in 0..d {
                    let (vkp, vkq) = (v[k * d + p], v[k * d + q]);
                    v[k * d + p] = c * vkp - s * vkq;
                    v[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..d).map(|i| a[i * d + i]).collect(), v)
}

/// W = c V diag(lam^(-alpha/2)) V^T for the covariance `cov` (a ridge of 1e-3 of the mean eigenvalue keeps tiny
/// eigenvalues from exploding), with c chosen so that W cov W has the trace of cov. Also returns the ridged eigenvalues.
fn whitening_matrix(cov: &[f64], d: usize, alpha: f64, rank: usize) -> (Vec<f32>, Vec<f64>) {
    let (lam, v) = jacobi_eigen(cov.to_vec(), d);
    let mean_lam = lam.iter().map(|l| l.max(0.0)).sum::<f64>() / d as f64;
    let lam: Vec<f64> = lam.iter().map(|l| l.max(0.0) + 1e-3 * mean_lam).collect();
    let scale = (lam.iter().sum::<f64>() / lam.iter().map(|l| l.powf(1.0 - alpha)).sum::<f64>()).sqrt();
    // `rank` < d keeps only the top-`rank` eigen-directions (a rank-`rank` W; the keys stay d wide).
    let mut sorted = lam.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let keep = sorted[rank - 1];
    let g: Vec<f64> = lam.iter().map(|l| if *l >= keep { scale * l.powf(-alpha / 2.0) } else { 0.0 }).collect();
    let mut w = vec![0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            w[i * d + j] = (0..d).map(|k| v[i * d + k] * g[k] * v[j * d + k]).sum::<f64>() as f32;
        }
    }
    (w, lam)
}

/// A random orthogonal d x d matrix (row-major): Gram-Schmidt on Gaussian rows, from `seed`.
fn random_rotation(d: usize, seed: u64) -> Vec<f64> {
    let mut rng = scratchtape::nn::Rng::new(seed);
    let mut r = vec![0f64; d * d];
    for i in 0..d {
        let mut v: Vec<f64> = (0..d).map(|_| rng.next_gaussian() as f64).collect();
        for _ in 0..2 {
            for j in 0..i {
                let dot: f64 = (0..d).map(|k| v[k] * r[j * d + k]).sum();
                (0..d).for_each(|k| v[k] -= dot * r[j * d + k]);
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        (0..d).for_each(|k| r[i * d + k] = v[k] / norm);
    }
    r
}

fn fit_whitener(dev: &DeviceParams, cfg: &Config, train: &[usize], tap: usize, alpha: f64, rank: usize, rotate: Option<u64>) {
    let d = cfg.d;
    let total = (train.len() - 1) / seq_len();
    let starts: Vec<usize> = (0..2000.min(total)).map(|i| i * total / 2000.min(total) * seq_len()).collect();
    // The sums and the Gram matrix of the tap's hidden states, on the device (f32 matmuls, summed over launches in f64): the
    // 260 MB of hidden states never cross the link, only a d x d matrix per launch.
    let (mut n, mut sum, mut gram) = (0usize, vec![0f64; d], vec![0f64; d * d]);
    for chunk in starts.chunks(2 * chunk_windows()) {
        let (ids, targets) = window_ids(train, chunk);
        let mut dt = DeviceTape::new(dev);
        let (outs, _, _) = model_forward(&mut dt, cfg, &ids, &targets, chunk.len());
        let (x, rows) = (dt.value(outs[tap]), ids.len());
        let (g, s, ones) = (client().empty(d * d * 4), client().empty(d * 4), upload_f32(&vec![1.0; rows]));
        matmul_f32(MatRef { trans: true, ..MatRef::new(x) }, MatRef::new(x), MatRef::new(&g), 1, d, rows, d, Epilogue::default());
        matmul_f32(MatRef::new(&ones), MatRef::new(x), MatRef::new(&s), 1, 1, rows, d, Epilogue::default());
        let r = read_many(vec![g, s]);
        n += rows;
        gram.iter_mut().zip(&r[0]).for_each(|(a, &b)| *a += b as f64);
        sum.iter_mut().zip(&r[1]).for_each(|(a, &b)| *a += b as f64);
    }
    let mu: Vec<f64> = sum.iter().map(|s| s / n as f64).collect();
    let mut cov = vec![0f64; d * d];
    for i in 0..d {
        for j in i..d {
            cov[i * d + j] = gram[i * d + j] / n as f64 - mu[i] * mu[j];
            cov[j * d + i] = cov[i * d + j];
        }
    }
    let (mut w, lam) = whitening_matrix(&cov, d, alpha, rank);
    if let Some(seed) = rotate {
        // Keys and queries are rotated alike, so distances are unchanged; the coordinates become close to Gaussian with equal variance,
        // which is what the low-bit key quantiser (`keybits`) assumes (RaBitQ, arXiv 2405.12497).
        let r = random_rotation(d, seed);
        w = (0..d * d).map(|ij| (0..d).map(|k| r[ij / d * d + k] * w[k * d + ij % d] as f64).sum::<f64>() as f32).collect();
    }
    let mut sorted = lam.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    println!(
        "whitener: alpha {alpha}, {n} sample keys, eigenvalues {:.3e} .. {:.3e} (top 5% hold {:.0}% of the variance)",
        sorted[d - 1],
        sorted[0],
        100.0 * sorted[..d / 20].iter().sum::<f64>() / sorted.iter().sum::<f64>()
    );
    WHITEN.set((mu.iter().map(|&m| m as f32).collect(), w)).ok();
}

fn main() {
    let t_main = std::time::Instant::now();
    let stamp = |what: &str| eprintln!("setup: {what} at {:.1}s", t_main.elapsed().as_secs_f64());
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: tier_eval <checkpoint> [key=value ...] (see the file header)");
    let ckpt = &args[0];
    let mut opt = std::collections::HashMap::new();
    let mut specs: Vec<String> = vec![];
    for a in &args[1..] {
        let (k, v) = a.split_once('=').unwrap_or_else(|| panic!("expected key=value, got {a}"));
        match k {
            "tier" => specs.push(v.to_string()),
            "generate" | "prompts" | "byte_gap" | "prompt" | "temp" | "seed" | "corpus" | "d_model" | "heads" | "d_ff" | "blocks" | "tap" | "expect" | "stride" | "offset" | "store" | "store_from" | "part" | "store_part" | "memory" | "warm" | "seq" | "memory_corpus"
            | "recent" | "whiten" | "whiten_rank" | "classes" | "survey" | "dyn" | "dyn_control" | "dyn_every" | "dyn_steps" | "dyn_rms" | "dyn_pull" | "dyn_keys" | "dyn_layers" | "dyn_nn" | "dyn_nn_mode" | "dyn_nn_stride" | "dyn_nn_lr" | "dump" | "wdump" | "batch" | "ivf" | "ivf_check" | "keybits" | "rotate" | "memcache" => {
                assert!(opt.insert(k, v.to_string()).is_none(), "{k} given twice")
            }
            _ => panic!("unknown key {k}"),
        }
    }
    let get = |k: &str, d: &str| opt.get(k).cloned().unwrap_or(d.to_string());
    let size = |k: &str, d: &str| -> usize { get(k, d).parse().unwrap() };
    SEQ.store(size("seq", "64"), std::sync::atomic::Ordering::Relaxed);
    let cfg = Config { vocab: VOCAB, d: size("d_model", "256"), heads: size("heads", "8"), d_ff: size("d_ff", "512"), t: seq_len(), n_blocks: size("blocks", "4"), softmax1: false };
    let tap = size("tap", &(cfg.n_blocks - 1).to_string());
    assert!(tap < cfg.n_blocks, "tap must be a block index below {}", cfg.n_blocks);
    let stride = size("stride", "1");
    assert!(stride >= 1 && (stride == 1 || !opt.contains_key("expect")), "stride >= 1, and `expect` needs stride 1");
    let warm = size("warm", "0");
    assert!(warm < seq_len() && (warm == 0 || !opt.contains_key("expect")), "warm < seq, and `expect` needs warm 0");
    let unit = seq_len() - warm;
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
    if opt.contains_key("wdump") {
        assert!(opt.contains_key("dump"), "wdump needs dump= (the kNN dump it is joined with)");
        let parts: Vec<&str> = specs.last().unwrap().split('+').collect();
        let n = parts.len();
        assert!(n >= 2 && parts[n - 2].starts_with("words:") && parts[n - 1].starts_with("knn:") && parts.iter().filter(|p| p.starts_with("words:")).count() == 1, "wdump needs the last tier= spec to have exactly one words: part, immediately before its final knn:");
    }
    let corpus = get("corpus", "novels6");

    let flat: Vec<f32> = std::fs::read_to_string(ckpt).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
    assert_eq!(flat.len(), cfg.len(), "{ckpt} has {} parameters, this config needs {}", flat.len(), cfg.len());
    let (train, held_out) = split_corpus(&corpus);
    // `memory_corpus=<name>`: the memory is filled from that corpus's training text (it must share this corpus's held-out text); the lexicon, words and whitener keep `corpus`.
    let memory_train = opt.get("memory_corpus").map(|m| split_corpus(m).0);
    let memory_train: &[usize] = memory_train.as_deref().unwrap_or(&train);
    let _lease = gpu_lease::hold(Kind::Shared, "scratchtape tier_eval", Duration::from_secs(4 * 3600));
    let mut dev = DeviceParams::upload(&flat);
    if opt.contains_key("byte_gap") {
        // Bytes whose held-out frequency far exceeds their train frequency (or that train never has): the model has seen too few.
        let (mut tr, mut ho) = ([0usize; VOCAB], [0usize; VOCAB]);
        train.iter().for_each(|&b| tr[b] += 1);
        held_out.iter().for_each(|&b| ho[b] += 1);
        let (nt, nh) = (train.len() as f64, held_out.len() as f64);
        println!("byte_gap: held-out vs train frequency per byte (shown: >= 50 held-out and 4x over-represented)");
        for b in (0..VOCAB).filter(|&b| ho[b] >= 50 && ho[b] as f64 / nh > 4.0 * tr[b] as f64 / nt) {
            println!("  byte {b:#04x}: held-out {} ({:.4}%), train {} ({:.4}%)", ho[b], 100.0 * ho[b] as f64 / nh, tr[b], 100.0 * tr[b] as f64 / nt);
        }
        return;
    }
    if let Some(n) = opt.get("prompts") {
        pooled(&dev, &cfg, &train, &held_out, &specs, n.parse().unwrap(), size("generate", "600"));
        return;
    }
    if let Some(len) = opt.get("generate") {
        generate(&dev, &cfg, &train, &held_out, &specs, len.parse().unwrap(), size("prompt", "20000"), get("temp", "0.8").parse().unwrap(), size("seed", "7") as u64);
        return;
    }

    stamp("corpus and checkpoint read");
    // Held-out words the train split has never shown: the ceiling on what a train-split lexicon can know.
    let mut train_words: std::collections::HashMap<Vec<u8>, u32> = std::collections::HashMap::new();
    for w in word_list(&train) {
        *train_words.entry(w).or_default() += 1;
    }
    let held_words = word_list(&held_out);
    let oov: Vec<&Vec<u8>> = held_words.iter().filter(|w| !train_words.contains_key(*w)).collect();
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
        if let Some(a) = opt.get("whiten").map(|a| a.parse::<f64>().unwrap_or_else(|_| panic!("whiten=<alpha>, got {a}"))).filter(|&a| a > 0.0) {
            stamp("before whitener");
            fit_whitener(&dev, &cfg, &train, tap, a, size("whiten_rank", &cfg.d.to_string()), opt.get("rotate").map(|s| s.parse().unwrap_or_else(|_| panic!("rotate=<seed>, got {s}"))));
        }
        stamp("after whitener");
        // `memcache=<dir>` (with `ivf`): the built memory is saved there once and reloaded by later runs with the same model and build options.
        let cache = opt.get("memcache").zip(opt.get("ivf")).map(|(dir, ivf)| {
            let meta = std::fs::metadata(ckpt).unwrap_or_else(|e| panic!("{ckpt}: {e}"));
            let secs = meta.modified().ok().and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs());
            let mut key = format!("v2 {ckpt} {} {secs} cells={}", meta.len(), ivf.split(':').next().unwrap_or(""));
            for k in ["corpus", "memory_corpus", "seq", "warm", "stride", "offset", "whiten", "whiten_rank", "tap", "store", "store_from", "keybits", "rotate", "memory", "d_model", "heads", "d_ff", "blocks"] {
                key += &format!(" {k}={}", opt.get(k).map_or("", |v| v.as_str()));
            }
            let hash = key.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
            std::fs::create_dir_all(dir).expect("memcache dir");
            (std::path::Path::new(dir).join(format!("{hash:016x}.mem")), key)
        });
        let t_cache = std::time::Instant::now();
        let cached = cache.as_ref().and_then(|(path, key)| KnnStore::load(path, key).unwrap_or_else(|e| panic!("memcache {}: {e}", path.display())));
        let hit = cached.is_some();
        let (mut n_train, mut n_held) = (0, 0);
        let mut gpu = match cached {
            Some((g, nt)) => {
                n_train = nt as usize;
                println!("memory loaded from {} in {:.1}s", cache.as_ref().unwrap().0.display(), t_cache.elapsed().as_secs_f64());
                g
            }
            None => {
                let mut g = KnnStore::new(cfg.d, 16384);
                if let Some(bits) = opt.get("keybits") {
                    g.set_key_bits(bits.parse().unwrap_or_else(|_| panic!("keybits=8|4|2, got {bits}")));
                }
                g
            }
        };
        let build = std::time::Instant::now();
        let mut t_fwd = std::time::Duration::ZERO;
        let mut load = |data: &[usize], starts: &[usize], doc: bool| {
            // Only the hidden states are needed (no logits or losses), so launches can be twice as large.
            for chunk in starts.chunks(2 * chunk_windows()) {
                let t0 = std::time::Instant::now();
                let f = forward(&dev, &cfg, data, chunk, tap, true, false);
                t_fwd += t0.elapsed();
                let vals: Vec<u8> = f.targets.iter().map(|&t| t as u8).collect();
                if doc { gpu.add_doc(&f.hidden, &vals) } else { gpu.add(&f.hidden, &vals) }
            }
        };
        if !hit && from != "held" && want > 0 {
            let total = (memory_train.len() - 1) / seq_len();
            let starts: Vec<usize> = (0..want.min(total)).map(|i| i * total / want.min(total) * seq_len()).collect();
            load(memory_train, &starts, false);
            n_train = starts.len();
        }
        if !hit && from != "train" {
            let windows = (held_out.len() - 1) / seq_len();
            let unscored: Vec<usize> = (0..windows).filter(|i| causal || (i % stride != offset && i * spart_n / windows == spart_i)).map(|i| i * seq_len()).collect();
            let starts: Vec<usize> = if causal { unscored } else { (0..want.min(unscored.len())).map(|i| unscored[i * unscored.len() / want.min(unscored.len())]).collect() };
            load(&held_out, &starts, true);
            n_held = starts.len();
        }
        gpu.finish();
        let nprobe = opt.get("ivf").map(|v| {
            let (cells, probes) = v.split_once(':').unwrap_or_else(|| panic!("ivf=<cells>:<probes>, got {v}"));
            assert!(online && from == "train", "ivf needs memory=online and store_from=train");
            if !hit {
                gpu.build_ivf(cells.parse().unwrap());
            }
            probes.parse::<usize>().unwrap()
        });
        if let (false, Some((path, key))) = (hit, &cache) {
            let t0 = std::time::Instant::now();
            gpu.save(path, key, n_train as u64).unwrap_or_else(|e| panic!("memcache {}: {e}", path.display()));
            println!("memory saved to {} in {:.1}s", path.display(), t0.elapsed().as_secs_f64());
        }
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
        let n_train_keys = n_train * seq_len();
        let online = online.then(|| {
            let docs = held_out_docs(&corpus);
            RefCell::new(Online { n_train: n_train_keys, doc_first_key: vec![None; docs.len()], docs })
        });
        let causal = causal.then(|| {
            let docs = held_out_docs(&corpus);
            let windows = (held_out.len() - 1) / seq_len();
            let first_window = (0..windows).map(|w| docs[docs.partition_point(|&b| b <= w * seq_len()) - 1].div_ceil(seq_len())).collect();
            Causal { n_train: n_train_keys, first_window, recent: size("recent", &usize::MAX.to_string()) }
        });
        Some(Rc::new(Store { gpu: RefCell::new(gpu), causal, online, warm, found: RefCell::new(Default::default()), nprobe, check: Cell::new(size("ivf_check", "0")) }))
    } else {
        None
    };

    client().memory_cleanup(); // a cold build's transient buffers would otherwise stay pooled and starve the search workspace
    stamp("memory ready");
    let dump_spec = opt.contains_key("dump").then(|| specs.len() - 1);
    let wdump_spec = opt.contains_key("wdump").then(|| specs.len() - 1);
    let mut tiers: Vec<Vec<Box<dyn Tier>>> = specs.iter().enumerate().map(|(i, s)| build_tier(s, &train, &store, dump_spec == Some(i), wdump_spec == Some(i))).collect();
    let need_hidden = tiers.iter().flatten().any(|t| t.needs_hidden());
    let (mut ce_model, mut ce_device) = (0.0f64, 0.0f64);
    let mut ce_tier = vec![0.0f64; tiers.len()];
    // `classes=1`: per position class, the count and the summed CE of the model and of each spec.
    let survey = opt.contains_key("survey");
    let dyn_lr: Option<f32> = opt.get("dyn").map(|v| v.parse().unwrap());
    let dyn_control = opt.contains_key("dyn_control");
    let dyn_steps = size("dyn_steps", "1");
    let layer_mask = opt.get("dyn_layers").map(|spec| {
        let mut m = vec![false; cfg.len()];
        for part in spec.split(',') {
            let (lo, hi) = match part {
                "emb" => (0, cfg.block_off(0)),
                "head" => (cfg.final_ln_off(), cfg.len()),
                b if b.starts_with('b') => {
                    let i: usize = b[1..].parse().unwrap_or_else(|_| panic!("dyn_layers part {b}"));
                    (cfg.block_off(i), cfg.block_off(i + 1))
                }
                other => panic!("dyn_layers: emb, b<i> or head, got {other}"),
            };
            m[lo..hi].fill(true);
        }
        m
    });
    let mut adapter = Adapter::new(dyn_lr.unwrap_or(0.0), opt.get("dyn_rms").map(|v| v.parse().unwrap()), get("dyn_pull", "0").parse().unwrap(), &flat, layer_mask);
    let chunk_n = size("dyn_every", &chunk_windows().to_string());
    assert!(dyn_lr.is_some() || !dyn_control, "dyn_control=1 needs dyn=<lr>");
    let frozen_keys = get("dyn_keys", "adapted") == "frozen";
    assert!(!frozen_keys || dyn_lr.is_some(), "dyn_keys=frozen needs dyn=<lr>");
    let frozen = DeviceParams::upload(&flat);
    let nn_m = size("dyn_nn", "0");
    assert!(nn_m == 0 || dyn_lr.is_some(), "dyn_nn=<m> needs dyn=<lr>");
    let nn_mode = get("dyn_nn_mode", "add");
    assert!(["add", "only", "random"].contains(&nn_mode.as_str()), "dyn_nn_mode=add|only|random, got {nn_mode}");
    let nn_lr: f32 = opt.get("dyn_nn_lr").map_or(dyn_lr.unwrap_or(0.0), |v| v.parse().unwrap());
    let nn_index = (nn_m > 0).then(|| {
        let t0 = std::time::Instant::now();
        let stride = size("dyn_nn_stride", "8");
        let total = (memory_train.len() - 1) / seq_len();
        let starts: Vec<usize> = (0..total).step_by(stride).map(|w| w * seq_len()).collect();
        let mut emb = Vec::with_capacity(starts.len() * cfg.d);
        for chunk in starts.chunks(2 * chunk_windows()) {
            emb.extend(NnIndex::embed(&forward(&frozen, &cfg, memory_train, chunk, tap, true, false).hidden, cfg.d, warm));
        }
        println!("neighbour index: {} train windows embedded in {:.1}s", starts.len(), t0.elapsed().as_secs_f64());
        NnIndex { starts, emb, d: cfg.d }
    });
    let mut nn_rng = 0x9e3779b97f4a7c15u64;
    let dyn_docs = held_out_docs(&corpus);
    let mut dyn_doc = usize::MAX;
    assert!(!survey || !specs.is_empty(), "survey=1 needs a tier spec");
    let mut survey_rows: Vec<SurveyRow> = vec![]; // (position of the input byte, model CE, last spec's CE, its three likeliest bytes)
    let mut cls_n = [0usize; 10];
    let word_classes = WordClasses::new(&train);
    let mut wc_bytes = vec![0usize; WordClasses::COUNT];
    let mut wc_cost = vec![vec![0.0f64; WordClasses::COUNT]; 1 + tiers.len()];
    let mut cls_model = [0.0f64; 10];
    let mut cls_tier = vec![[0.0f64; 10]; tiers.len()];
    // `classes=1`, line ends: [target is a newline or a space][a newline in the 64-byte window][column bucket] -> (count, model CE sum).
    let mut col = [[[(0usize, 0.0f64); COLUMN_EDGES.len() + 1]; 2]; 2];
    let n_win = (held_out.len() - 1) / seq_len();
    let starts: Vec<usize> = (offset * unit..held_out.len() - seq_len()).step_by(unit * stride).filter(|s| s / seq_len() * part_n / n_win == part_i).collect();
    // Wall time per phase: forward launch + readback, tier prepare (kNN search + readback), per-row scoring, online writes.
    let mut phase = [std::time::Duration::ZERO; 4];
    stamp("tiers built, chunk loop starts");
    let mu = client().memory_usage();
    println!("device memory: {:.2} GB in use, {:.2} GB reserved", mu.bytes_in_use as f64 / 1e9, mu.bytes_reserved as f64 / 1e9);
    let t_all = std::time::Instant::now();
    let (mut t_lap, mut paused) = (std::time::Instant::now(), std::time::Duration::ZERO);
    // Chunks are run through the model (and adapted) `batch` at a time, then their neighbour searches are done together,
    // then they are scored: adaptation and memory writes never depend on the tiers' output.
    let batch = size("batch", "1");
    let all_chunks: Vec<&[usize]> = starts.chunks(chunk_n).collect();
    let mut ready: std::collections::VecDeque<(Forward, usize)> = Default::default();
    for (ci, &chunk) in all_chunks.iter().enumerate() {
        if ready.is_empty() {
            for (cj, &chunk) in all_chunks.iter().enumerate().take(ci + batch).skip(ci) {
                // Yield to another job's exclusive lease, and log speed per 20 chunks (paused time excluded) so a slow stretch reads as contention.
                let t_p = std::time::Instant::now();
                gpu_lease::pause_while_exclusive();
                paused += t_p.elapsed();
                if cj > 0 && cj % 20 == 0 {
                    eprintln!("speed: chunks {}..{} {:.2} s/chunk ({:.0} s paused)", cj - 20, cj, (t_lap.elapsed() - paused).as_secs_f64() / 20.0, paused.as_secs_f64());
                    (t_lap, paused) = (std::time::Instant::now(), std::time::Duration::ZERO);
                    if let Some(path) = opt.get("dump") {
                        for tier in tiers.last().unwrap() {
                            tier.dump(path, false);
                        }
                    }
                    if let Some(path) = opt.get("wdump") {
                        for tier in tiers.last().unwrap() {
                            tier.wdump(path, false);
                        }
                    }
                }
                if dyn_lr.is_some() {
                    // A new book starts from the checkpoint again: nothing adapted on one book carries into the next.
                    let doc = dyn_docs.partition_point(|&b| b <= chunk[0]) - 1;
                    if doc != dyn_doc {
                        dev = DeviceParams::upload(&flat);
                        adapter.reset();
                        dyn_doc = doc;
                    }
                }
                let t0 = std::time::Instant::now();
                let mut f = forward(&dev, &cfg, &held_out, chunk, tap, need_hidden && !frozen_keys, true);
                if frozen_keys {
                    // Hidden states (memory queries and in-document keys) from the checkpoint, as the train keys are; probabilities from the adapted model.
                    f.hidden = forward(&frozen, &cfg, &held_out, chunk, tap, true, false).hidden;
                }
                phase[0] += t0.elapsed();
                let t0 = std::time::Instant::now();
                let base = store.as_ref().map_or(0, |st| st.write_chunk(&f.hidden, &f.targets, chunk, warm, cfg.d));
                phase[3] += t0.elapsed();
                if dyn_lr.is_some() {
                    // The step reads only text that has just been scored (or, for the control, text from far away).
                    let source: Vec<usize> = if dyn_control { chunk.iter().map(|&s| (s + held_out.len() / 2) % (held_out.len() - seq_len() - 1)).collect() } else { chunk.to_vec() };
                    if nn_mode != "only" || nn_index.is_none() {
                        for _ in 0..dyn_steps {
                            adapter.step(&mut dev, &cfg, &held_out, &source);
                        }
                    }
                    if let Some(idx) = &nn_index {
                        let picked: Vec<usize> = if nn_mode == "random" {
                            (0..nn_m)
                                .map(|_| {
                                    nn_rng = nn_rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                                    idx.starts[(nn_rng >> 33) as usize % idx.starts.len()]
                                })
                                .collect()
                        } else {
                            let hidden = if frozen_keys { f.hidden.clone() } else { forward(&frozen, &cfg, &held_out, chunk, tap, true, false).hidden };
                            let per = nn_m.div_ceil(chunk.len());
                            let mut v: Vec<usize> = NnIndex::embed(&hidden, cfg.d, warm).chunks(cfg.d).flat_map(|q| idx.nearest(q, per)).map(|j| idx.starts[j]).collect();
                            v.sort_unstable();
                            v.dedup();
                            v
                        };
                        let lr = std::mem::replace(&mut adapter.lr, nn_lr);
                        adapter.step(&mut dev, &cfg, memory_train, &picked);
                        adapter.lr = lr;
                    }
                }
                ready.push_back((f, base));
                debug_assert_eq!(ready.len(), cj - ci + 1);
            }
            let t0 = std::time::Instant::now();
            if let Some(st) = &store {
                let items: Vec<Pending> = ready.iter().enumerate().map(|(i, (f, base))| Pending { chunk: ci + i, hidden: &f.hidden, rows: f.ids.len(), starts: all_chunks[ci + i], base: *base }).collect();
                st.search_batch(&items);
            }
            phase[1] += t0.elapsed();
        }
        let (f, _) = ready.pop_front().unwrap();
        let rows = f.ids.len();
        let t0 = std::time::Instant::now();
        for tier in tiers.iter_mut().flatten() {
            tier.prepare(ci, &f.hidden, rows, chunk);
        }
        phase[1] += t0.elapsed();
        let t0 = std::time::Instant::now();
        for r in 0..rows {
            let t = r % seq_len();
            if t < warm {
                continue;
            }
            let window = &f.ids[r - t..=r];
            let abs = chunk[r / seq_len()] + t;
            let context = &held_out[(abs + 1).saturating_sub(CONTEXT)..=abs];
            let probs = softmax(&f.logits[r * VOCAB..(r + 1) * VOCAB]);
            ce_model -= probs[f.targets[r]].ln();
            let cls = class_of(*window.last().unwrap(), f.targets[r]);
            cls_n[cls] += 1;
            cls_model[cls] -= probs[f.targets[r]].ln();
            let wcls = word_classes.class(&held_out, abs + 1);
            wc_bytes[wcls] += 1;
            wc_cost[0][wcls] -= probs[f.targets[r]].ln();
            if let Some(kind) = [b'\n', b' '].iter().position(|&b| b as usize == f.targets[r]) {
                let column = context.iter().rposition(|&b| b == b'\n' as usize).map_or(usize::MAX, |i| context.len() - 1 - i);
                let cell = &mut col[kind][window.contains(&(b'\n' as usize)) as usize][COLUMN_EDGES.iter().filter(|&&e| column >= e).count()];
                cell.0 += 1;
                cell.1 -= probs[f.targets[r]].ln();
            }
            ce_device += f.row_loss[r] as f64;
            let hid = if need_hidden { &f.hidden[r * cfg.d..(r + 1) * cfg.d] } else { &[][..] };
            let ctx = Ctx { row: r, window, context, target: f.targets[r], hidden: hid };
            for (i, spec_tiers) in tiers.iter_mut().enumerate() {
                let mut p = probs.clone();
                for tier in spec_tiers.iter_mut() {
                    tier.adjust(&ctx, &mut p);
                }
                ce_tier[i] -= p[f.targets[r]].max(1e-300).ln();
                cls_tier[i][cls] -= p[f.targets[r]].max(1e-300).ln();
                wc_cost[i + 1][wcls] -= p[f.targets[r]].max(1e-300).ln();
                if survey && i + 1 == specs.len() {
                    let mut top = [(0u8, 0f32); 3];
                    for (b, &q) in p.iter().enumerate() {
                        let q = q as f32;
                        if q > top[2].1 {
                            top[2] = (b as u8, q);
                            top.sort_by(|a, b| b.1.total_cmp(&a.1));
                        }
                    }
                    survey_rows.push((abs, -probs[f.targets[r]].ln(), -p[f.targets[r]].max(1e-300).ln(), top));
                }
            }
        }
        phase[2] += t0.elapsed();
        if let Some(st) = &store {
            st.found.borrow_mut().remove(&ci);
        }
    }
    let total = t_all.elapsed().as_secs_f64();
    let pct = |d: std::time::Duration| format!("{:.1}s ({:.0}%)", d.as_secs_f64(), 100.0 * d.as_secs_f64() / total);
    println!(
        "timing: {total:.1}s over {} chunks: forward {}, tier prepare {}, row scoring {}, online writes {}",
        starts.chunks(chunk_n).count(),
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
    if opt.contains_key("classes") {
        // Per class: its share of the positions, the model's mean CE there and its share of the total CE, then for each
        // spec the CE gain in that class as thousandths of a nat of the overall CE (the columns sum to the overall gain).
        println!("model CE at newline and space targets by column (bytes since the last newline) and whether the 64-byte window shows a newline:");
        for (k, kind) in ["newline", "space"].iter().enumerate() {
            for (v, vis) in ["window has no newline", "window has a newline"].iter().enumerate() {
                let cells: Vec<String> = col[k][v].iter().map(|&(c, s)| if c == 0 { "      -".to_string() } else { format!("{:5.2}({:>5})", s / c as f64, c) }).collect();
                println!("  {kind:8} {vis:22} {}", cells.join(" "));
            }
        }
        println!("  columns: <20 <40 <56 <64 <68 <71 <74 >=74 (CE nats(count))");
        println!("loss by position class (gain columns: 1e-3 nats of overall CE, summing to the overall gain):");
        print!("{:30} {:>6} {:>7} {:>7}", "class", "share%", "modelCE", "of CE%");
        for i in 0..specs.len() {
            print!(" {:>7}", format!("spec{i}"));
        }
        println!();
        for c in 0..10 {
            let nc = cls_n[c].max(1) as f64;
            print!("{:30} {:6.1} {:7.3} {:7.1}", CLASS_NAMES[c], 100.0 * cls_n[c] as f64 / n, cls_model[c] / nc, 100.0 * cls_model[c] / (ce_model * n));
            for i in 0..specs.len() {
                print!(" {:7.2}", 1e3 * (cls_model[c] - cls_tier[i][c]) / n);
            }
            println!();
        }
    }
    if opt.contains_key("classes") {
        print_word_classes(&wc_bytes, &wc_cost, &specs);
    }
    if survey {
        let _ = survey_report(&held_out, &held_out_docs(&corpus), &train_words, &survey_rows, n);
    }
    if let Some(path) = opt.get("dump") {
        for tier in tiers.last().unwrap() {
            tier.dump(path, true);
        }
    }
    if let Some(path) = opt.get("wdump") {
        for tier in tiers.last().unwrap() {
            tier.wdump(path, true);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn neighbour_index_embeds_the_scored_rows_and_finds_the_nearest() {
        let d = 2;
        // Two windows: rows below warm are ignored, the rest average to (3, 0) and (0, 5).
        let (warm, n) = (seq_len() - 2, seq_len());
        let mut hidden = vec![100.0f32; 2 * n * d];
        for (w, v) in [(0usize, [3.0f32, 0.0]), (1, [0.0, 5.0])] {
            for t in warm..n {
                hidden[(w * n + t) * d..(w * n + t + 1) * d].copy_from_slice(&v);
            }
        }
        let emb = NnIndex::embed(&hidden, d, warm);
        assert_eq!(emb, vec![3.0, 0.0, 0.0, 5.0]);
        let idx = NnIndex { starts: vec![0, n], emb, d };
        assert_eq!(idx.nearest(&[0.0, 4.0], 1), vec![1]);
        assert_eq!(idx.nearest(&[2.0, 0.5], 2), vec![0, 1]);
    }

    use super::*;

    /// "aa bb aa bb\n" with only "aa" in the lexicon: the second "bb" is seen earlier in the book, the first is new.
    #[test]
    fn survey_splits_letters_by_lexicon_and_recency() {
        let text: Vec<usize> = "aa bb aa bb\n".bytes().map(|b| b as usize).collect();
        let rows: Vec<SurveyRow> = (0..text.len() - 1).map(|a| (a, 2.0, 1.0, [(0, 0.0); 3])).collect();
        let lexicon: std::collections::HashMap<Vec<u8>, u32> = [(b"aa".to_vec(), 1)].into_iter().collect();
        let t = survey_report(&text, &[0], &lexicon, &rows, rows.len() as f64);
        let count = |k: &str, p: &'static str| t.get(&(k.to_string(), p)).map_or(0, |c| c.0);
        assert_eq!(count("in lexicon", "1st letter"), 1);
        assert_eq!(count("in lexicon", "2nd-3rd letter"), 2);
        assert_eq!(count("not in lexicon, new to book", "1st letter"), 1);
        assert_eq!(count("not in lexicon, new to book", "2nd-3rd letter"), 1);
        assert_eq!(count("not in lexicon, seen earlier in book", "1st letter"), 1);
        assert_eq!(count("not in lexicon, seen earlier in book", "2nd-3rd letter"), 1);
        assert_eq!(count("space", ""), 3);
        assert_eq!(count("newline", ""), 1);
    }

    /// Lines of nine letters: a newline always follows a letter at column 9, a letter never does at column 5. The tier
    /// moves the model's newline/space split towards the train rate only where the window shows no newline.
    #[test]
    fn wrap_resplits_newline_and_space_by_column_only_when_the_window_is_blind() {
        let train: Vec<usize> = "aaaaaaaaa\n".repeat(50).bytes().map(|b| b as usize).collect();
        let mut wrap = Wrap::new(&train, 1.0);
        assert!(wrap.newline_rate(true, 9) > 0.95);
        let line = "\naaaaaaaaa".bytes().map(|b| b as usize).collect::<Vec<_>>();
        let mut probs = vec![0.0; VOCAB];
        probs[b'\n' as usize] = 0.1;
        probs[b' ' as usize] = 0.4;
        probs[b'a' as usize] = 0.5;
        // Blind window (64 bytes of 'a'), the context reaches the newline 9 bytes back.
        let window = vec![b'a' as usize; 64];
        let ctx = Ctx { row: 0, window: &window, context: &line, target: b'\n' as usize, hidden: &[] };
        wrap.adjust(&ctx, &mut probs);
        assert!(probs[b'\n' as usize] > 0.45 && probs[b' ' as usize] < 0.05 && (probs[b'a' as usize] - 0.5).abs() < 1e-12, "{probs:?}");
        // A window that shows a newline leaves the distribution alone.
        let mut probs2 = vec![0.0; VOCAB];
        probs2[b'\n' as usize] = 0.1;
        probs2[b' ' as usize] = 0.4;
        probs2[b'a' as usize] = 0.5;
        let seen = Ctx { window: &line, ..ctx };
        wrap.adjust(&seen, &mut probs2);
        assert_eq!((probs2[b'\n' as usize], probs2[b' ' as usize]), (0.1, 0.4));
    }

    /// A random rotation is orthogonal (R R^T = I), so rotating keys and queries alike keeps every distance.
    #[test]
    fn random_rotation_is_orthogonal() {
        let d = 24;
        let r = random_rotation(d, 3);
        for i in 0..d {
            for j in 0..d {
                let dot: f64 = (0..d).map(|k| r[i * d + k] * r[j * d + k]).sum();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-9, "({i},{j}) = {dot}");
            }
        }
        assert_ne!(r, random_rotation(d, 4));
    }

    /// alpha = 1 whitens: W cov W is c^2 times the identity up to the ridge (1%), and alpha = 0 leaves the metric alone (W = identity).
    #[test]
    fn whitening_matrix_whitens_at_alpha_one_and_is_identity_at_zero() {
        let d = 4;
        // A symmetric positive-definite covariance with correlated, unequal directions.
        let cov: Vec<f64> = vec![4.0, 1.0, 0.5, 0.0, 1.0, 3.0, 0.2, 0.1, 0.5, 0.2, 1.0, 0.3, 0.0, 0.1, 0.3, 0.5];
        let (w, _) = whitening_matrix(&cov, d, 1.0, d);
        let mul = |a: &[f64], b: &[f64]| -> Vec<f64> { (0..d * d).map(|ij| (0..d).map(|k| a[ij / d * d + k] * b[k * d + ij % d]).sum()).collect() };
        let w: Vec<f64> = w.iter().map(|&x| x as f64).collect();
        let m = mul(&mul(&w, &cov), &w);
        for i in 0..d {
            for j in 0..d {
                if i == j {
                    assert!((m[i * d + i] - m[0]).abs() < 1e-2 * m[0], "diagonal {i}: {} vs {}", m[i * d + i], m[0]);
                } else {
                    assert!(m[i * d + j].abs() < 1e-2 * m[0], "off-diagonal ({i},{j}) = {}", m[i * d + j]);
                }
            }
        }
        let (w0, _) = whitening_matrix(&cov, d, 0.0, d);
        for i in 0..d {
            for j in 0..d {
                assert!((w0[i * d + j] - (i == j) as u8 as f32).abs() < 1e-5);
            }
        }
    }
}
