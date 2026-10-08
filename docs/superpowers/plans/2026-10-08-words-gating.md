# Words-Tier Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a per-position words-tier weight beats the best constant words weight on held-out text, by dumping what the words tier saw and fitting gates offline.

**Architecture:** `tier_eval` gets a `wdump=<path>` option: the words tier writes eight f32 per scored position (applied flag, target probability before words, the word-list distribution's target probability, features). A new example `words_gate_fit` joins that file row by row with the existing kNN `dump=` file, checks it reproduces the logged CE, and fits a binned and a sigmoid gate on one half of the chunks and scores them on the other.

**Tech Stack:** Rust (existing examples, no new dependencies); one GPU full run for the dumps.

**Spec:** `docs/superpowers/specs/2026-10-08-words-gating-design.md`

## Global Constraints

- Machine rules (`~/.claude/WORKING_STYLE.md`, `LONG_RUNS.md`): never run `python` reading stdin; one cargo at a time, always `-j 8`; the GPU runs (Task 1 Step 6 and Task 3) are done by the controller, not by implementers, with orphan/CPU/lease/eGPU checks first and `scripts/load_log.ps1` alongside the full run. Implementers must not run `tier_eval` or touch the GPU.
- `examples/tiny_lm/tier_eval.rs` is CRLF in the working tree: use the Edit tool and check `git diff --stat` before each commit.
- Dump row formats (little-endian f32): kNN dump x6 `chunk, p_t, k_t, H, d0, dk` (existing; `p_t` is the target probability **after** words); words dump x8 `chunk, applied (0/1), p_in_t, q_t, H_in, ln(1+total), bigram (0/1), prefix_len`.
- Stack under test: `tier=lexicon:0.3+words:1:0.25+knn:256:0.4:15`, `warm=32 stride=1 memory=online store=100000`, checkpoint `runs/nov_big_k1_d0.1_8m_m0.ckpt`.
- Gate 0: mean loss at lambda_w = 0.25, mu = 0.4 equals the logged CE to 1e-4. Gate 0b: at every row `0.75 p_in + 0.25 q` equals the kNN dump's `p_t` to 1e-5.
- Verdict (one, printed by `main` after both splits): **success** if the sigmoid gain over the best constant lambda_w is > 0.002 nats per scored position on both splits; otherwise **kill** if the binned gain is < 0.002 on either split; otherwise **inconclusive** (record, no wiring). Each gain is printed with a per-chunk paired standard error.
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` (the attribution line the session's system reminder gives, for commits made by the controller or by implementers alike).

---

### Task 1: `wdump=` option in `tier_eval`

**Files:**
- Modify: `examples/tiny_lm/tier_eval.rs` (`Tier` trait; `Words` struct, `Words::new`, `Words::adjust`; `build_tier`; option parsing; dump loops in `main`; header comment)

**Interfaces:**
- Produces: option `wdump=<path>`, requires `dump=` and a last spec of the form `...words:...+knn:...` with the `words:` part before the final `knn:` part and without `first` letters; file of 32-byte rows as in Global Constraints. `p_in_t` is `probs[target]` entering words, `H_in` the natural-log entropy of that distribution, `q_t` the word-list distribution's probability of the target (for unapplied rows `q_t = p_in_t`).

- [ ] **Step 1: Read the code to adapt the steps**

Read `struct Words`, `Words::new`, `Words::adjust` (about lines 195-320), `fn build_tier` (`"words"` arm near line 497), the `Tier` trait (line 90-101), the option match (`"dump"` near line 559), the dump assert (near line 586), and the dump loops in `main` (near lines 680, 699, 772). `Words::adjust` has five early exits: the let-else `let Some(boundary) = ... else { return };` (about line 270, no semicolon) and four `return;` statements (about lines 273, 300, 306, 310; the one at 300 is inside the first-letters branch). The applied branch is the last block (about lines 308-316), `for b in 0..VOCAB { let q = ...; probs[b] = ... }`, reached only when the prefix is non-empty.

- [ ] **Step 2: Add the trait hook and the Words fields**

In the `Tier` trait, after `fn dump(...)`, add:

```rust
    /// Writes this tier's words dump (the `wdump=` option) to `path`; tiers without one do nothing.
    fn wdump(&self, _path: &str, _last: bool) {}
```

Add to `struct Words` the fields `dump: Option<Vec<f32>>` and `chunk: usize`, initialise them in `Words::new` as `dump: None, chunk: 0` (the `build_tier` arm sets `dump` below). Add to `impl Tier for Words` (if the trait's `prepare` has a different parameter list, keep its types and names, underscore-prefixing the unused ones, and only use `chunk`):

```rust
    fn prepare(&mut self, chunk: usize, _hidden: &[f32], _rows: usize, _starts: &[usize]) {
        self.chunk = chunk;
    }
```

- [ ] **Step 3: Split `Words::adjust` into a mixing body and a recording wrapper**

Rename the existing `fn adjust` body to an inherent method `fn mix(&mut self, ctx: &Ctx, probs: &mut [f64]) -> Option<(f64, f64, bool, usize)>` in `impl Words`, returning `Some((q_t, total, bigram_found, prefix.len()))` from the final applied branch and `None` from every other exit: change the let-else to `else { return None };` and each of the four `return;` to `return None;` (the first-letters branch also returns `None`). `bigram_found = found.is_some()` must be captured before `found` is consumed by `unwrap_or_else`. In the final loop capture the target's word-list probability as it is computed: declare `let mut q_t = 0.0;` before the loop and inside it, after `let q = ...;`, add `if b == ctx.target { q_t = q; }`; return `Some((q_t, total, bigram_found, prefix.len()))` after the loop. Then write the trait method:

```rust
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
```

and, in the same impl, the writer (same atomic pattern as `Knn::dump`):

```rust
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
```

If the original `Words::adjust` is `fn adjust` inside `impl Tier for Words` together with `report`, keep `report` there and move only the body.

- [ ] **Step 4: Wire the option**

`build_tier` gets a fifth parameter `wdump: bool`; in the `"words"` arm, after constructing the `Words`, set `w.dump = wdump.then(Vec::new);` and `assert!(!(wdump && first), "wdump does not support first letters");` (use the arm's actual variable names). Update the call in `main` to pass `wdump_spec == Some(i)` where `let wdump_spec = opt.contains_key("wdump").then(|| specs.len() - 1);` next to `dump_spec`. Accept `"wdump"` in the option match next to `"dump"`. Next to the existing `dump` assert add:

```rust
    if opt.contains_key("wdump") {
        assert!(opt.contains_key("dump"), "wdump needs dump= (the kNN dump it is joined with)");
        let parts: Vec<&str> = specs.last().unwrap().split('+').collect();
        let n = parts.len();
        assert!(n >= 2 && parts[n - 2].starts_with("words:") && parts[n - 1].starts_with("knn:") && parts.iter().filter(|p| p.starts_with("words:")).count() == 1, "wdump needs the last tier= spec to have exactly one words: part, immediately before its final knn:");
    }
```

In both dump loops in `main` (every-20-chunks and final), after the existing `tier.dump(...)` loop add a loop calling `tier.wdump(path, false)` / `tier.wdump(path, true)` for `opt.get("wdump")` over `tiers.last().unwrap()`. Header comment: add `[wdump=<path>]` to the usage line and a sentence: "`wdump=<path>` (with `dump=`) writes eight f32 per scored position of the last spec's words tier (chunk, applied, p_in_target, q_target, entropy_in, ln(1+count mass), bigram, prefix_len) for `words_gate_fit`."

- [ ] **Step 5: Build**

Run: `cargo build --release --example tier_eval -j 8`
Expected: `Finished`, no new warnings.

- [ ] **Step 6: Slice check (controller only; GPU, ~15 s)**

After the quiet/lease checks and `Copy-Item target\release\examples\tier_eval.exe runs\tier_eval_run.exe -Force`:

```powershell
.\runs\tier_eval_run.exe runs/nov_big_k1_d0.1_8m_m0.ckpt warm=32 stride=1 part=5/96 memory=online store=100000 tier=lexicon:0.3+words:1:0.25+knn:256:0.4:15 dump=runs/wg_slice_k.bin wdump=runs/wg_slice_w.bin > runs\wg_slice.log 2>&1
```

Expected: the log has `dump: 5248 rows` and `wdump: 5248 rows`; file sizes 125952 and 167936 bytes. (Gate 0 / 0b are checked by Task 2's tool on these files.) Note the CE the log prints for the spec.

- [ ] **Step 7: Commit**

```bash
git add examples/tiny_lm/tier_eval.rs
git commit -m "tier_eval: wdump=<path> writes the words tier's per-position inputs for words_gate_fit"
```

(plus the Co-Authored-By line)

---

### Task 2: `words_gate_fit`

**Files:**
- Create: `examples/tiny_lm/words_gate_fit.rs`
- Modify: `Cargo.toml` (append an `[[example]]` entry)

**Interfaces:**
- Consumes: the two dump formats from Global Constraints.
- Produces: `words_gate_fit <wdump> <kdump> [mu=0.4] [lw0=0.25] [logged=<CE>]`, printing Gate 0 / 0b, then for each of the even/odd and early/late splits: constant baselines, the binned table, the sigmoid gate's weights and the verdict.

- [ ] **Step 1: Create the file (reader, loss, baselines, gates, report, tests)**

`examples/tiny_lm/words_gate_fit.rs`:

```rust
// Fits and scores per-position gates for the words tier's weight from `tier_eval dump=` + `wdump=` files
// (docs/superpowers/specs/2026-10-08-words-gating-design.md).
//   words_gate_fit <wdump> <kdump> [mu=0.4] [lw0=0.25] [logged=<CE the run logged>]
// Rows are joined by index. A position's loss at words weight l is
//   -ln((1 - mu) ((1 - l) p_in + l q) + mu k),
// with q = p_in at positions where words does not apply. Gains are per scored position (all rows), baseline CE minus
// gated CE on the score half, positive is better. The small helpers below (sigmoid, quantile_edges, bin_of, the descent)
// repeat gate_fit.rs on purpose: the examples are separate crates and the loss differs.

#[derive(Clone, Copy, Debug)]
struct Row {
    chunk: u32,
    applied: bool,
    p: f64,
    q: f64,
    h: f64,
    tot: f64,
    bi: f64,
    plen: f64,
    k: f64,
    d0: f64,
    /// d0 == 0 and dk == 0: the kNN tier found no neighbours (or every distance clamped to 0); counted, see main.
    noneigh: bool,
}

fn read_f32(path: &str, width: usize) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    assert!(bytes.len() % (4 * width) == 0, "{path}: {} bytes is not a multiple of {}", bytes.len(), 4 * width);
    bytes.chunks_exact(4 * width).map(|c| c.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect()).collect()
}

/// Joins the words dump (8 f32 per row) with the kNN dump (6 f32 per row) and checks Gate 0b on every row.
fn join(w: &[Vec<f32>], k: &[Vec<f32>], lw0: f64) -> Vec<Row> {
    assert_eq!(w.len(), k.len(), "row counts differ: wdump {} vs dump {}", w.len(), k.len());
    w.iter()
        .zip(k)
        .enumerate()
        .map(|(i, (w, k))| {
            assert_eq!(w[0], k[0], "row {i}: chunk ids differ ({} vs {})", w[0], k[0]);
            let (p, q) = (w[2] as f64, w[3] as f64);
            let mixed = (1.0 - lw0) * p + lw0 * q;
            assert!((mixed - k[1] as f64).abs() < 1e-5, "gate 0b, row {i}: (1-lw0) p_in + lw0 q = {mixed}, the kNN dump has {}", k[1]);
            Row { chunk: w[0] as u32, applied: w[1] > 0.5, p, q, h: w[4] as f64, tot: w[5] as f64, bi: w[6] as f64, plen: w[7] as f64, k: k[2] as f64, d0: k[4] as f64, noneigh: k[4] == 0.0 && k[5] == 0.0 }
        })
        .collect()
}

fn loss(r: &Row, mu: f64, l: f64) -> f64 {
    -((1.0 - mu) * ((1.0 - l) * r.p + l * r.q) + mu * r.k).max(1e-300).ln()
}

fn mean_gated(rows: &[Row], mu: f64, lam: impl Fn(&Row) -> f64) -> f64 {
    rows.iter().map(|r| loss(r, mu, lam(r))).sum::<f64>() / rows.len() as f64
}

fn mean_fixed(rows: &[Row], mu: f64, l: f64) -> f64 {
    mean_gated(rows, mu, |_| l)
}

/// The best constant weight on a 0.01 grid: (weight, mean loss).
fn best_fixed(rows: &[Row], mu: f64) -> (f64, f64) {
    (0..=100).map(|i| i as f64 / 100.0).map(|l| (l, mean_fixed(rows, mu, l))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap()
}

fn split_parity(rows: &[Row]) -> (Vec<Row>, Vec<Row>) {
    rows.iter().partition(|r| r.chunk % 2 == 0)
}

/// Earlier half to fit, later half to score, cut at the first chunk change at or after the midpoint.
fn split_halves(rows: &[Row]) -> (Vec<Row>, Vec<Row>) {
    let mut cut = rows.len() / 2;
    while cut > 0 && cut < rows.len() && rows[cut].chunk == rows[cut - 1].chunk {
        cut += 1;
    }
    (rows[..cut].to_vec(), rows[cut..].to_vec())
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn quantile_edges(mut v: Vec<f64>, bins: usize) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (1..bins).map(|i| v[i * v.len() / bins]).collect()
}

fn bin_of(v: f64, edges: &[f64]) -> usize {
    edges.iter().filter(|&&e| v >= e).count()
}

/// A weight per (bigram flag, entropy quartile): the best constant weight (0.02 grid) on the rows it was fitted to.
struct Binned {
    h_edges: Vec<f64>,
    lam: Vec<f64>,
}

impl Binned {
    fn fit(applied: &[Row], mu: f64) -> Binned {
        let h_edges = quantile_edges(applied.iter().map(|r| r.h).collect(), 4);
        let mut groups: Vec<Vec<Row>> = vec![vec![]; 8];
        for r in applied {
            groups[(r.bi > 0.5) as usize * 4 + bin_of(r.h, &h_edges)].push(*r);
        }
        let lam = groups
            .iter()
            .map(|g| {
                if g.is_empty() {
                    return 0.25;
                }
                (0..=50).map(|i| i as f64 * 0.02).map(|l| (l, mean_fixed(g, mu, l))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
            })
            .collect();
        Binned { h_edges, lam }
    }

    fn lambda(&self, r: &Row) -> f64 {
        self.lam[(r.bi > 0.5) as usize * 4 + bin_of(r.h, &self.h_edges)]
    }
}

const NF: usize = 5;

fn feats(r: &Row) -> [f64; NF] {
    [r.h, r.tot, r.bi, r.plen, (1.0 + r.d0).ln()]
}

/// lambda_w = sigmoid(w . [1, z_1..z_5]) with z the standardised features (entropy in, ln(1+count mass), bigram hit,
/// prefix length, ln(1+d0)).
#[derive(Clone)]
struct Gate {
    w: [f64; NF + 1],
    mean: [f64; NF],
    std: [f64; NF],
}

impl Gate {
    fn z(&self, r: &Row) -> [f64; NF + 1] {
        let f = feats(r);
        let mut z = [1.0; NF + 1];
        for j in 0..NF {
            z[j + 1] = (f[j] - self.mean[j]) / self.std[j];
        }
        z
    }

    fn lambda(&self, r: &Row) -> f64 {
        let z = self.z(r);
        sigmoid((0..=NF).map(|j| self.w[j] * z[j]).sum())
    }
}

/// Mean loss of the gate on `rows` and its gradient in `g.w`.
fn objective(rows: &[Row], mu: f64, g: &Gate) -> (f64, [f64; NF + 1]) {
    let (mut total, mut grad) = (0.0, [0.0; NF + 1]);
    for r in rows {
        let lam = g.lambda(r);
        let m = ((1.0 - mu) * ((1.0 - lam) * r.p + lam * r.q) + mu * r.k).max(1e-300);
        total -= m.ln();
        let dl = -(1.0 - mu) * (r.q - r.p) / m * lam * (1.0 - lam);
        let z = g.z(r);
        for j in 0..=NF {
            grad[j] += dl * z[j];
        }
    }
    let n = rows.len() as f64;
    (total / n, grad.map(|x| x / n))
}

/// Full-batch gradient descent with backtracking from the constant weight `init` (strictly between 0 and 1), fitted on
/// applied rows only.
fn fit_gate(applied: &[Row], mu: f64, init: f64, iters: usize) -> Gate {
    let f: Vec<[f64; NF]> = applied.iter().map(feats).collect();
    let n = applied.len() as f64;
    let mean: [f64; NF] = std::array::from_fn(|j| f.iter().map(|x| x[j]).sum::<f64>() / n);
    let std: [f64; NF] = std::array::from_fn(|j| (f.iter().map(|x| (x[j] - mean[j]).powi(2)).sum::<f64>() / n).sqrt().max(1e-9));
    let mut w = [0.0; NF + 1];
    w[0] = (init / (1.0 - init)).ln();
    let mut g = Gate { w, mean, std };
    let mut step = 1.0;
    for _ in 0..iters {
        let (f0, grad) = objective(applied, mu, &g);
        let g2: f64 = grad.iter().map(|x| x * x).sum();
        if g2 < 1e-16 {
            break;
        }
        loop {
            let mut next = g.clone();
            for j in 0..=NF {
                next.w[j] -= step * grad[j];
            }
            if objective(applied, mu, &next).0 <= f0 - 1e-4 * step * g2 {
                g = next;
                step *= 1.5;
                break;
            }
            step *= 0.5;
            if step < 1e-12 {
                return g;
            }
        }
    }
    g
}

/// Mean per-row loss of `a` minus that of `b` on `rows` (positive: b is better) and its standard error over chunks
/// (rows are grouped by chunk, so the chunk is the unit of resampling).
fn gain_se(rows: &[Row], mu: f64, a: impl Fn(&Row) -> f64, b: impl Fn(&Row) -> f64) -> (f64, f64) {
    let mut sums: Vec<(u32, f64)> = vec![];
    for r in rows {
        let d = loss(r, mu, a(r)) - loss(r, mu, b(r));
        match sums.last_mut() {
            Some((c, s)) if *c == r.chunk => *s += d,
            _ => sums.push((r.chunk, d)),
        }
    }
    let (n, c) = (rows.len() as f64, sums.len() as f64);
    let total: f64 = sums.iter().map(|x| x.1).sum();
    let mean = total / c;
    let var = sums.iter().map(|x| (x.1 - mean).powi(2)).sum::<f64>() / (c - 1.0).max(1.0);
    (total / n, (c * var).sqrt() / n)
}

/// Prints one split's results and returns (binned gain, sigmoid gain) over the best constant, or None if skipped.
fn report(label: &str, mu: f64, lw0: f64, fit: &[Row], score: &[Row]) -> Option<(f64, f64)> {
    let applied: Vec<Row> = fit.iter().filter(|r| r.applied).copied().collect();
    if applied.len() < 100 || score.is_empty() {
        println!("[{label}] skipped: {} applied fit rows, {} score rows", applied.len(), score.len());
        return None;
    }
    let (lam, ce) = best_fixed(fit, mu);
    let at_lw0 = mean_fixed(score, mu, lw0);
    let best = mean_fixed(score, mu, lam);
    println!("[{label}] fit {} rows ({} applied) / score {} rows", fit.len(), applied.len(), score.len());
    println!("[{label}] constant {lw0}: score {at_lw0:.5}; best constant on fit = {lam:.2}: fit {ce:.5}, score {best:.5}");
    let b = Binned::fit(&applied, mu);
    let binned = mean_gated(score, mu, |r| b.lambda(r));
    let (gb, seb) = gain_se(score, mu, |_| lam, |r| b.lambda(r));
    println!("[{label}] binned (bigram x entropy quartile): score {binned:.5} (gain {:+.5} vs {lw0}, {gb:+.5} +- {seb:.5} vs best constant)", at_lw0 - binned);
    for bi in 0..2 {
        println!("[{label}]   bigram={bi}: lambda by entropy quartile low->high {}", (0..4).map(|j| format!("{:.2}", b.lam[bi * 4 + j])).collect::<Vec<_>>().join("  "));
    }
    println!("[{label}]   entropy edges {:?}", b.h_edges.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>());
    let g = fit_gate(&applied, mu, lam.clamp(0.01, 0.99), 500);
    let param = mean_gated(score, mu, |r| g.lambda(r));
    let (gp, sep) = gain_se(score, mu, |_| lam, |r| g.lambda(r));
    println!("[{label}] sigmoid gate: fit {:.5}, score {param:.5} (gain {:+.5} vs {lw0}, {gp:+.5} +- {sep:.5} vs best constant)", mean_gated(fit, mu, |r| g.lambda(r)), at_lw0 - param);
    println!("[{label}]   w = {:?} on [1, entropy_in, ln(1+count), bigram, prefix_len, ln(1+d0)] (standardised on the fit half)", g.w.map(|x| (x * 1000.0).round() / 1000.0));
    Some((gb, gp))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut files, mut mu, mut lw0, mut logged) = (vec![], 0.4, 0.25, None);
    for a in &args {
        match a.split_once('=') {
            Some(("mu", v)) => mu = v.parse().unwrap(),
            Some(("lw0", v)) => lw0 = v.parse().unwrap(),
            Some(("logged", v)) => logged = Some(v.parse::<f64>().unwrap()),
            Some(_) => panic!("unknown argument {a}"),
            None => files.push(a.clone()),
        }
    }
    assert!(files.len() == 2, "usage: words_gate_fit <wdump> <kdump> [mu=0.4] [lw0=0.25] [logged=<CE>]");
    let rows = join(&read_f32(&files[0], 8), &read_f32(&files[1], 6), lw0);
    assert!(!rows.is_empty(), "empty dumps");
    let all = mean_fixed(&rows, mu, lw0);
    let napplied = rows.iter().filter(|r| r.applied).count();
    println!("{} rows ({} applied); gate 0b passed; lambda_w {lw0}, mu {mu}: CE {all:.5}", rows.len(), napplied);
    match logged {
        Some(c) => {
            assert!((all - c).abs() < 1e-4, "gate 0: the dumps give {all:.5}, the run logged {c}");
            println!("gate 0 passed: dumps reproduce the logged {c}");
        }
        None => println!("gate 0 NOT checked (no logged=<CE>)"),
    }
    let nn = rows.iter().filter(|r| r.noneigh).count();
    if nn > 0 {
        println!("note: {nn} rows have d0 = dk = 0 (no neighbours, or every distance clamped to 0); gate 0 shows whether they matter");
    }
    let (fit, score) = split_parity(&rows);
    let a = report("even/odd", mu, lw0, &fit, &score);
    let (fit, score) = split_halves(&rows);
    let b = report("early/late", mu, lw0, &fit, &score);
    let (Some((b1, p1)), Some((b2, p2))) = (a, b) else {
        println!("verdict: inconclusive (a split was skipped)");
        return;
    };
    let verdict = if p1 > 0.002 && p2 > 0.002 {
        "SUCCESS: sigmoid gain > 0.002 on both splits"
    } else if b1 < 0.002 || b2 < 0.002 {
        "KILL: binned gain < 0.002 on a split"
    } else {
        "inconclusive: neither the success nor the kill condition holds"
    };
    println!("verdict: {verdict} (sigmoid {p1:.4} / {p2:.4}, binned {b1:.4} / {b2:.4} nats per position over the best constant, even/odd / early/late)");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic rows: at high entropy the word list is right (q = 0.6, p = 0.05), at low entropy the model is
    /// (p = 0.8, q = 0.1); k is a weak 0.2 everywhere; about a fifth of the rows are unapplied (q = p).
    fn synth() -> Vec<Row> {
        let mut s = 777u64;
        let mut u = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as f64 / (1u64 << 31) as f64
        };
        (0..4000)
            .map(|i| {
                let high = u() < 0.5;
                let applied = u() < 0.8;
                let (p, q) = if high { (0.05, 0.6) } else { (0.8, 0.1) };
                Row { chunk: i / 20, applied, p, q: if applied { q } else { p }, h: if high { 3.0 + u() } else { 0.5 * u() }, tot: 1.0 + 3.0 * u(), bi: (u() < 0.5) as u8 as f64, plen: 1.0 + 5.0 * u(), k: 0.2, d0: 5.0 + 20.0 * u(), noneigh: false }
            })
            .collect()
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let applied: Vec<Row> = synth().into_iter().filter(|r| r.applied).collect();
        let mut g = fit_gate(&applied, 0.4, 0.25, 0);
        g.w = [0.3, -0.2, 0.1, 0.4, -0.1, 0.2];
        let (_, grad) = objective(&applied, 0.4, &g);
        for j in 0..=NF {
            let (mut up, mut dn) = (g.clone(), g.clone());
            up.w[j] += 1e-5;
            dn.w[j] -= 1e-5;
            let fd = (objective(&applied, 0.4, &up).0 - objective(&applied, 0.4, &dn).0) / 2e-5;
            assert!((grad[j] - fd).abs() < 1e-6, "w[{j}]: analytic {} vs finite difference {fd}", grad[j]);
        }
    }

    #[test]
    fn sigmoid_gate_beats_the_best_constant_when_the_signal_is_real() {
        let rows = synth();
        let applied: Vec<Row> = rows.iter().filter(|r| r.applied).copied().collect();
        let (lam, _) = best_fixed(&rows, 0.4);
        let best = mean_fixed(&rows, 0.4, lam);
        let g = fit_gate(&applied, 0.4, lam.clamp(0.01, 0.99), 300);
        let gated = mean_gated(&rows, 0.4, |r| g.lambda(r));
        assert!(gated < best - 0.1, "gated {gated:.4} vs best constant {best:.4}");
        let b = Binned::fit(&applied, 0.4);
        let binned = mean_gated(&rows, 0.4, |r| b.lambda(r));
        assert!(binned < best - 0.1, "binned {binned:.4} vs best constant {best:.4}");
    }

    fn rows_f32(rows: &[[f32; 8]]) -> Vec<Vec<f32>> {
        rows.iter().map(|r| r.to_vec()).collect()
    }

    #[test]
    fn join_accepts_a_consistent_pair_and_rejects_a_bad_one() {
        // p_in 0.2, q 0.6, lw0 0.25 -> mixed 0.3; the kNN row carries p_t = 0.3, k = 0.5.
        let w = rows_f32(&[[0.0, 1.0, 0.2, 0.6, 1.5, 2.0, 1.0, 3.0]]);
        let k = vec![vec![0.0, 0.3, 0.5, 1.5, 20.0, 60.0]];
        let rows = join(&w, &k, 0.25);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].applied && (rows[0].q - 0.6).abs() < 1e-6 && (rows[0].k - 0.5).abs() < 1e-6);
        let bad = vec![vec![0.0, 0.31, 0.5, 1.5, 20.0, 60.0]];
        assert!(std::panic::catch_unwind(|| join(&w, &bad, 0.25)).is_err());
    }

    #[test]
    fn split_halves_never_splits_a_chunk() {
        let rows: Vec<Row> = (0..10).map(|i| Row { chunk: i / 4, ..synth()[0] }).collect(); // chunks 0 0 0 0 1 1 1 1 2 2
        let (a, b) = split_halves(&rows);
        assert_eq!(a.len() + b.len(), 10);
        assert!(a.last().unwrap().chunk != b.first().unwrap().chunk);
    }
}
```

Append to `Cargo.toml`:

```toml

[[example]]
name = "words_gate_fit"
path = "examples/tiny_lm/words_gate_fit.rs"
```

- [ ] **Step 2: Run the tests**

Run: `cargo test --release --example words_gate_fit -j 8`
Expected: 4 tests pass, no warnings in the build. If `gradient_matches_finite_differences` fails, re-derive: d loss / d pre-sigmoid = -(1-mu)(q-p)/m * lam(1-lam); do not loosen the tolerance.

- [ ] **Step 3: Run on the slice dumps (Gate 0 / 0b on real data)**

Uses the files from Task 1 Step 6; `<x>` is the CE the slice log printed for the spec (4 decimals):

`cargo run --release --example words_gate_fit -j 8 -q -- runs/wg_slice_w.bin runs/wg_slice_k.bin logged=<x>`

Expected: `gate 0b passed` then `gate 0 passed`, then two report blocks (numbers on 5k positions are noise). If Gate 0b panics, stop: the words dump's `q_t` (algebra in Task 1 Step 3) or the join is wrong.

- [ ] **Step 4: Commit**

```bash
git add examples/tiny_lm/words_gate_fit.rs Cargo.toml
git commit -m "words_gate_fit: join the words and kNN dumps, Gate 0/0b, constant baseline, binned and sigmoid words-weight gates"
```

(plus the Co-Authored-By line)

---

### Task 3: Full-run script, run, record

**Files:**
- Create: `scripts/tier_words_gate_dump.sh`
- Modify: `docs/tiers_design.md` (a new entry before `## Order`)

- [ ] **Step 1: Write the script**

`scripts/tier_words_gate_dump.sh`:

```sh
#!/bin/sh
# Full held-out run of the deployable stack (kNN weight 0.4) with the kNN and words per-position dumps for words_gate_fit
# (docs/superpowers/specs/2026-10-08-words-gating-design.md). One spec; no KNN_TIMING (it syncs after every tile).
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
$E $C warm=32 stride=1 memory=online store=100000 \
  tier=lexicon:0.3+words:1:0.25+knn:256:0.4:15 dump=runs/wg_dump_k.bin wdump=runs/wg_dump_w.bin > runs/wg_dump.log 2>&1
```

Commit it (`git add scripts/tier_words_gate_dump.sh`).

- [ ] **Step 2: Launch (controller; only on the user's go; contended or quiet is the user's call)**

Pre-checks per LONG_RUNS.md (orphans, CPU, leases, eGPU users, Defender), copy the exe, start `sh scripts/tier_words_gate_dump.sh` hidden, start `scripts\load_log.ps1 tier_eval_run runs\wg_dump_load.log`, record the PID, start time and logs in `docs/tiers_design.md` and commit. Expected duration 8-15 min (the previous full run took 449 s under contention).

- [ ] **Step 3: Fit when the run exits**

Take `<x>` = the CE printed for the spec in `runs/wg_dump.log` and run `cargo run --release --example words_gate_fit -j 8 -q -- runs/wg_dump_w.bin runs/wg_dump_k.bin logged=<x>`. Gate 0 and 0b must pass before reading the rest.

- [ ] **Step 4: Record**

In `docs/tiers_design.md`, replace the in-flight entry with the result: run time and machine load, Gate 0/0b, the constant baselines, binned table, sigmoid weights, the verdict against the success and kill criteria on both splits, and what it rules out or suggests next (wiring the gate needs the kNN search before words). Check `Get-Process tier_eval_run,words_gate_fit,python` for leftovers; commit.

---

## Self-review (done)

- **Spec coverage:** wdump format and validity checks (Task 1); join, Gate 0 / 0b, no-neighbour assert, both splits, constant baselines, binned and sigmoid gates, per-position gains, verdict (Task 2); the run, fit and record (Task 3). Out-of-scope items are not planned.
- **Placeholders:** none; `<x>` is a value read from the run's log.
- **Types:** `Row`, `join(&[Vec<f32>], &[Vec<f32>], f64)`, `loss(&Row, mu, l)`, `best_fixed(&[Row], mu)`, `Binned::{fit, lambda}`, `Gate::{z, lambda}`, `objective(&[Row], mu, &Gate)`, `fit_gate(&[Row], mu, init, iters)`, `report(label, mu, lw0, fit, score)` agree across the code and tests; the words dump row has 8 f32 as in Global Constraints and the reader's `read_f32(path, 8)`.
