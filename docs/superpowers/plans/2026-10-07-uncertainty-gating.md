# Uncertainty Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a per-position kNN-memory weight (set from model entropy and nearest-key distances) lowers held-out CE below the fixed lambda = 0.5 stack (1.1313).

**Architecture:** `tier_eval` dumps six f32 per scored position (chunk, p_target, k_target, entropy, d0, dk) from the last, `knn`-terminated spec. A new offline example `gate_fit` reads the dump and fits/scores a binned and a 4-parameter sigmoid gate on even chunks, scoring on odd chunks.

**Tech Stack:** Rust (existing examples; no new dependencies), the existing `tier_eval` GPU run for the dump.

**Spec:** `docs/superpowers/specs/2026-10-07-uncertainty-gating-design.md`

## Global Constraints

- Machine rules (`~/.claude/WORKING_STYLE.md`, `LONG_RUNS.md`): never run `python` reading stdin (write a script file and run `python -P f.py < /dev/null`, or avoid python); the GPU run needs an orphan check (`tasklist | grep -i tier_eval_run`), a quiet-machine check (CPU, eGPU users, leases), `scripts/load_log.ps1` alongside, and a check for leftovers afterwards. Do not touch other sessions' processes. Subagents that may edit files must be given the python rule in their prompt.
- Files in this repo may have CRLF line endings; edit with the Edit tool or a CRLF-preserving script, and check `git diff --stat` before each commit.
- Gate 0: the dump's CE at lambda = 0.5 over all rows must equal the run's logged CE for that spec to 1e-4.
- Success: odd-half CE of the parametric gate beats odd-half CE at fixed lambda 0.5 by more than 0.002 nats. Kill: the binned gate gains less than 0.002 out of sample.
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

---

### Task 1: `dump=` option in `tier_eval`

**Files:**
- Modify: `examples/tiny_lm/tier_eval.rs` (header comment; `Tier` trait; `Knn` struct, `prepare`, `adjust`; `build_tier`; option parsing; the `build_tier` call site; end of `main`)

**Interfaces:**
- Produces: option `dump=<path>`; a file of 24-byte rows, little-endian f32 x 6: `chunk, p_t, k_t, H, d0, dk`, one row per scored position of the last spec's `knn` tier, in scoring order. `p_t` = target probability entering the knn tier, `k_t` = target probability under the kNN distribution, `H` = natural-log entropy of the distribution entering the knn tier, `d0` / `dk` = squared distance to the nearest / k-th neighbour. A position with no neighbours writes `[chunk, p_t, p_t, H, 0, 0]` (so any lambda gives `p_t`).

- [ ] **Step 1: Add the trait hook**

In the `Tier` trait, after `fn report`, add:

```rust
    /// Writes this tier's per-position dump (the `dump=` option) to `path`; tiers without one do nothing.
    fn dump(&self, _path: &str) {}
```

- [ ] **Step 2: Extend `Knn`**

Add fields `dump: Option<Vec<f32>>` and `chunk: usize` to `struct Knn`. In `Knn::prepare`, make the first line `self.chunk = chunk;`. Replace `Knn::adjust` with:

```rust
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
    fn dump(&self, path: &str) {
        if let Some(d) = &self.dump {
            let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(path, bytes).unwrap_or_else(|e| panic!("writing {path}: {e}"));
            println!("dump: {} rows to {path}", d.len() / 6);
        }
    }
```

The entropy is computed before the mixing loop mutates `probs`, so it is the distribution entering the memory tier. (`dump` goes inside `impl Tier for Knn` next to `report`.)

- [ ] **Step 3: `build_tier` and the call site**

Change the signature to `fn build_tier(spec: &str, train: &[usize], store: &Option<Rc<Store>>, dump: bool) -> Vec<Box<dyn Tier>>`. In the `"knn"` arm construct `Knn { store, k: a[0] as usize, lambda: a[1], temp: a[2], positions: 0, nearest: 0.0, kth: 0.0, dump: dump.then(Vec::new), chunk: 0 }`. Replace the call `specs.iter().map(|s| build_tier(s, &train, &store)).collect()` with:

```rust
    let dump_spec = opt.contains_key("dump").then(|| specs.len() - 1);
    let mut tiers: Vec<Vec<Box<dyn Tier>>> = specs.iter().enumerate().map(|(i, s)| build_tier(s, &train, &store, dump_spec == Some(i))).collect();
```

(the existing line is `let mut tiers: Vec<Vec<Box<dyn Tier>>> = specs.iter().map(|s| build_tier(s, &train, &store)).collect();`; `opt` and `specs` are both defined above it.)

- [ ] **Step 4: Option parsing and writing**

Add `"dump"` to the accepted keys in the `match k` arm (next to `"recent"`). After the line `let (spart_i, spart_n) = part("store_part");` add:

```rust
    if opt.contains_key("dump") {
        assert!(specs.last().is_some_and(|s| s.rsplit('+').next().is_some_and(|p| p.starts_with("knn:"))), "dump needs the last tier= spec to end in knn");
    }
```

After the `for (i, spec) in specs.iter().enumerate()` report loop at the end of `main`, add:

```rust
    if let Some(path) = opt.get("dump") {
        for tier in tiers.last().unwrap() {
            tier.dump(path);
        }
    }
```

In the file header comment, add `[dump=<path>]` to the usage line and one sentence: "`dump=<path>` writes six f32 per scored position of the last spec's knn tier (chunk, p_target, k_target, entropy, d0, dk) for `gate_fit`."

- [ ] **Step 5: Build**

Run: `cargo build --release --example tier_eval -j 8`
Expected: `Finished`, no new warnings.

- [ ] **Step 6: Slice check (needs the GPU, ~15 s; do the quiet check from Task 4 Step 2 first)**

```powershell
cd "C:\Users\Winston Li\Documents\GitHub\scratchtape"
Copy-Item target\release\examples\tier_eval.exe runs\tier_eval_run.exe -Force
.\runs\tier_eval_run.exe runs\nov_big_k1_d0.1_8m_m0.ckpt warm=32 stride=1 part=5/96 memory=online store=100000 tier=lexicon:0.3+words:1:0.25+knn:256:0.5:15 dump=runs/gate_slice.bin > runs\gate_slice.log 2>&1
```

Expected in `runs/gate_slice.log`: a line `tier lexicon:0.3+words:1:0.25+knn:256:0.5:15: CE <x>` and `dump: <n> rows to runs/gate_slice.bin`, with `<n>` equal to the position count in the `checkpoint ...` line, and `(Get-Item runs\gate_slice.bin).Length` equal to `<n> * 24`. Note `<x>` for Task 2.

- [ ] **Step 7: Commit**

```bash
git add examples/tiny_lm/tier_eval.rs
git commit -m "tier_eval: dump=<path> writes (chunk, p_t, k_t, entropy, d0, dk) per scored position of the last knn spec"
```

(plus the Co-Authored-By line)

---

### Task 2: `gate_fit` reader, fixed-lambda baselines and Gate 0

**Files:**
- Create: `examples/tiny_lm/gate_fit.rs`
- Modify: `Cargo.toml` (append an `[[example]]` entry)

**Interfaces:**
- Consumes: the dump format from Task 1.
- Produces (used by Task 3): `struct Row { chunk: u32, p: f64, k: f64, h: f64, d0: f64, dk: f64 }` (derive `Clone, Copy, Debug`); `fn read_rows(path: &str) -> Vec<Row>`; `fn loss(r: &Row, lam: f64) -> f64`; `fn mean_fixed(rows: &[Row], lam: f64) -> f64`; `fn best_fixed(rows: &[Row]) -> (f64, f64)` (weight, mean loss; grid 0.00..=1.00 step 0.01); `fn split(rows: &[Row]) -> (Vec<Row>, Vec<Row>)` (even chunks, odd chunks).

- [ ] **Step 1: Create the file with the baselines and their tests**

`examples/tiny_lm/gate_fit.rs`:

```rust
// Fits and scores per-position gates for the kNN memory's mixing weight from a `tier_eval dump=` file
// (docs/superpowers/specs/2026-10-07-uncertainty-gating-design.md).
//   gate_fit <dump> [logged=<CE the run logged for the dumped spec>]
// Fits on even chunks, scores on odd chunks. Rows are (chunk, p_target, k_target, entropy, d0, dk); a position's loss at
// weight l is -ln((1 - l) p + l k).

#[derive(Clone, Copy, Debug)]
struct Row {
    chunk: u32,
    p: f64,
    k: f64,
    h: f64,
    d0: f64,
    dk: f64,
}

fn read_rows(path: &str) -> Vec<Row> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    assert!(bytes.len() % 24 == 0, "{path}: {} bytes is not a multiple of 24", bytes.len());
    bytes
        .chunks_exact(24)
        .map(|c| {
            let f = |i: usize| f32::from_le_bytes(c[i * 4..i * 4 + 4].try_into().unwrap()) as f64;
            Row { chunk: f(0) as u32, p: f(1), k: f(2), h: f(3), d0: f(4), dk: f(5) }
        })
        .collect()
}

fn loss(r: &Row, lam: f64) -> f64 {
    -((1.0 - lam) * r.p + lam * r.k).max(1e-300).ln()
}

fn mean_fixed(rows: &[Row], lam: f64) -> f64 {
    rows.iter().map(|r| loss(r, lam)).sum::<f64>() / rows.len() as f64
}

/// The best constant weight on a 0.01 grid: (weight, mean loss).
fn best_fixed(rows: &[Row]) -> (f64, f64) {
    (0..=100).map(|i| i as f64 / 100.0).map(|l| (l, mean_fixed(rows, l))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap()
}

/// (even-chunk rows, odd-chunk rows).
fn split(rows: &[Row]) -> (Vec<Row>, Vec<Row>) {
    rows.iter().partition(|r| r.chunk % 2 == 0)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: gate_fit <dump> [logged=<CE>]");
    let rows = read_rows(&args[0]);
    assert!(!rows.is_empty(), "empty dump");
    let all = mean_fixed(&rows, 0.5);
    println!("{} rows; fixed lambda 0.5: CE {all:.5}", rows.len());
    for a in &args[1..] {
        let logged: f64 = a.strip_prefix("logged=").unwrap_or_else(|| panic!("unknown argument {a}")).parse().unwrap();
        assert!((all - logged).abs() < 1e-4, "gate 0: dump gives {all:.5}, the run logged {logged}");
        println!("gate 0 passed: dump reproduces the logged {logged}");
    }
    let (even, odd) = split(&rows);
    let (lam, ce) = best_fixed(&even);
    println!("even {} rows / odd {} rows", even.len(), odd.len());
    println!("fixed 0.5: even {:.5}, odd {:.5}", mean_fixed(&even, 0.5), mean_fixed(&odd, 0.5));
    println!("best fixed on even = {lam:.2}: even {ce:.5}, odd {:.5}", mean_fixed(&odd, lam));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(chunk: u32, p: f64, k: f64) -> Row {
        Row { chunk, p, k, h: 0.0, d0: 1.0, dk: 2.0 }
    }

    #[test]
    fn loss_at_the_ends_is_the_two_distributions() {
        let r = row(0, 0.2, 0.6);
        assert!((loss(&r, 0.0) + 0.2f64.ln()).abs() < 1e-12);
        assert!((loss(&r, 1.0) + 0.6f64.ln()).abs() < 1e-12);
        assert!((loss(&r, 0.5) + 0.4f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn best_fixed_picks_the_better_end_when_one_dominates() {
        let rows: Vec<Row> = (0..10).map(|i| row(i, 0.1, 0.9)).collect();
        assert_eq!(best_fixed(&rows).0, 1.0);
        let rows: Vec<Row> = (0..10).map(|i| row(i, 0.9, 0.1)).collect();
        assert_eq!(best_fixed(&rows).0, 0.0);
    }

    #[test]
    fn split_is_by_chunk_parity() {
        let rows: Vec<Row> = (0..6).map(|i| row(i, 0.5, 0.5)).collect();
        let (even, odd) = split(&rows);
        assert_eq!(even.iter().map(|r| r.chunk).collect::<Vec<_>>(), vec![0, 2, 4]);
        assert_eq!(odd.iter().map(|r| r.chunk).collect::<Vec<_>>(), vec![1, 3, 5]);
    }
}
```

Append to `Cargo.toml`:

```toml

[[example]]
name = "gate_fit"
path = "examples/tiny_lm/gate_fit.rs"
```

- [ ] **Step 2: Run the tests**

Run: `cargo test --release --example gate_fit -j 8`
Expected: 3 tests pass.

- [ ] **Step 3: Run on the slice dump (Gate 0 against the real log)**

Use the CE printed in `runs/gate_slice.log` (4 decimals) for `<x>`:

`cargo run --release --example gate_fit -- runs/gate_slice.bin logged=<x>`

Expected: `gate 0 passed`, then the even/odd baselines. If Gate 0 fails, stop: the dump or `gate_fit`'s CE is wrong (check the no-neighbour row and that entropy is taken before the mixing loop in Task 1).

- [ ] **Step 4: Commit**

```bash
git add examples/tiny_lm/gate_fit.rs Cargo.toml
git commit -m "gate_fit: read the dump, fixed-lambda baselines, Gate 0 against the logged CE"
```

(plus the Co-Authored-By line)

---

### Task 3: Binned and parametric gates in `gate_fit`

**Files:**
- Modify: `examples/tiny_lm/gate_fit.rs`

**Interfaces:**
- Consumes: `Row`, `loss`, `mean_fixed`, `best_fixed`, `split` from Task 2.
- Produces: `struct Binned` with `fn fit(rows: &[Row], nh: usize, nd: usize) -> Binned` and `fn lambda(&self, r: &Row) -> f64`; `#[derive(Clone)] struct Gate { w: [f64; 4], mean: [f64; 3], std: [f64; 3] }` with `fn lambda(&self, r: &Row) -> f64`; `fn fit_gate(rows: &[Row], init_lambda: f64, iters: usize) -> Gate`; `fn objective(rows: &[Row], g: &Gate) -> (f64, [f64; 4])` (mean loss and its gradient in `g.w`); `fn mean_gated(rows: &[Row], lam: impl Fn(&Row) -> f64) -> f64`.

- [ ] **Step 1: Write the failing tests**

Add inside `mod tests`:

```rust
    /// Deterministic rows: high entropy -> the kNN distribution is right (k = 0.5, p = 0.05); low entropy -> the model is
    /// (p = 0.8, k = 0.05). d0 and dk vary a little so the other features are not constant.
    fn synth() -> Vec<Row> {
        let mut s = 12345u64;
        let mut u = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as f64 / (1u64 << 31) as f64
        };
        (0..4000)
            .map(|i| {
                let high = u() < 0.5;
                let d0 = 5.0 + 20.0 * u();
                Row { chunk: i / 20, p: if high { 0.05 } else { 0.8 }, k: if high { 0.5 } else { 0.05 }, h: if high { 3.0 + u() } else { 0.5 * u() }, d0, dk: d0 + 50.0 * u() }
            })
            .collect()
    }

    #[test]
    fn binned_gate_learns_a_weight_per_entropy_bin() {
        let rows = synth();
        let b = Binned::fit(&rows, 2, 1);
        let (lo, hi) = (Row { h: 0.1, ..rows[0] }, Row { h: 3.5, ..rows[0] });
        assert!(b.lambda(&lo) < 0.2 && b.lambda(&hi) > 0.8, "{} {}", b.lambda(&lo), b.lambda(&hi));
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let rows = synth();
        let mut g = fit_gate(&rows, 0.5, 0); // zero iterations: the standardised starting gate
        g.w = [0.3, -0.2, 0.1, 0.4];
        let (_, grad) = objective(&rows, &g);
        for j in 0..4 {
            let (mut up, mut dn) = (g.clone(), g.clone());
            up.w[j] += 1e-5;
            dn.w[j] -= 1e-5;
            let fd = (objective(&rows, &up).0 - objective(&rows, &dn).0) / 2e-5;
            assert!((grad[j] - fd).abs() < 1e-6, "w[{j}]: analytic {} vs finite difference {fd}", grad[j]);
        }
    }

    #[test]
    fn parametric_gate_beats_the_best_fixed_weight_when_the_signal_is_real() {
        let rows = synth();
        let (_, fixed) = best_fixed(&rows);
        let g = fit_gate(&rows, 0.5, 300);
        let gated = mean_gated(&rows, |r| g.lambda(r));
        assert!(gated < fixed - 0.3, "gated {gated:.4} vs best fixed {fixed:.4}");
    }
```

- [ ] **Step 2: Run to see them fail**

Run: `cargo test --release --example gate_fit -j 8`
Expected: compile errors (`Binned`, `fit_gate`, `objective`, `mean_gated` not defined).

- [ ] **Step 3: Implement**

Add above `fn main`:

```rust
fn mean_gated(rows: &[Row], lam: impl Fn(&Row) -> f64) -> f64 {
    rows.iter().map(|r| loss(r, lam(r))).sum::<f64>() / rows.len() as f64
}

/// Quantile edges: `bins - 1` values splitting `v` into `bins` equal-count groups.
fn quantile_edges(mut v: Vec<f64>, bins: usize) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (1..bins).map(|i| v[i * v.len() / bins]).collect()
}

fn bin_of(v: f64, edges: &[f64]) -> usize {
    edges.iter().filter(|&&e| v >= e).count()
}

/// A weight per (entropy bin, nearest-distance bin): the best constant weight (0.02 grid) on the rows it was fitted to.
struct Binned {
    h_edges: Vec<f64>,
    d_edges: Vec<f64>,
    lam: Vec<f64>,
}

impl Binned {
    fn fit(rows: &[Row], nh: usize, nd: usize) -> Binned {
        let h_edges = quantile_edges(rows.iter().map(|r| r.h).collect(), nh);
        let d_edges = quantile_edges(rows.iter().map(|r| r.d0).collect(), nd);
        let mut groups: Vec<Vec<Row>> = vec![vec![]; nh * nd];
        for r in rows {
            groups[bin_of(r.h, &h_edges) * nd + bin_of(r.d0, &d_edges)].push(*r);
        }
        let lam = groups
            .iter()
            .map(|g| {
                if g.is_empty() {
                    return 0.5;
                }
                (0..=50).map(|i| i as f64 * 0.02).map(|l| (l, mean_fixed(g, l))).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
            })
            .collect();
        Binned { h_edges, d_edges, lam }
    }

    fn lambda(&self, r: &Row) -> f64 {
        self.lam[bin_of(r.h, &self.h_edges) * (self.d_edges.len() + 1) + bin_of(r.d0, &self.d_edges)]
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// lambda = sigmoid(w . [1, z_h, z_d, z_s]) where z are standardised entropy, ln(1 + d0) and (dk - d0) / (1 + d0).
#[derive(Clone)]
struct Gate {
    w: [f64; 4],
    mean: [f64; 3],
    std: [f64; 3],
}

fn feats(r: &Row) -> [f64; 3] {
    [r.h, (1.0 + r.d0).ln(), (r.dk - r.d0) / (1.0 + r.d0)]
}

impl Gate {
    fn z(&self, r: &Row) -> [f64; 4] {
        let f = feats(r);
        [1.0, (f[0] - self.mean[0]) / self.std[0], (f[1] - self.mean[1]) / self.std[1], (f[2] - self.mean[2]) / self.std[2]]
    }

    fn lambda(&self, r: &Row) -> f64 {
        let z = self.z(r);
        sigmoid((0..4).map(|j| self.w[j] * z[j]).sum())
    }
}

/// Mean loss of the gate on `rows` and its gradient in `g.w`.
fn objective(rows: &[Row], g: &Gate) -> (f64, [f64; 4]) {
    let (mut total, mut grad) = (0.0, [0.0; 4]);
    for r in rows {
        let lam = g.lambda(r);
        let m = ((1.0 - lam) * r.p + lam * r.k).max(1e-300);
        total -= m.ln();
        let dl = -(r.k - r.p) / m * lam * (1.0 - lam);
        let z = g.z(r);
        for j in 0..4 {
            grad[j] += dl * z[j];
        }
    }
    let n = rows.len() as f64;
    (total / n, grad.map(|x| x / n))
}

/// Full-batch gradient descent with backtracking, starting at the constant weight `init_lambda` (so it never ends worse
/// than that constant on the rows it is fitted to). `init_lambda` must be strictly between 0 and 1.
fn fit_gate(rows: &[Row], init_lambda: f64, iters: usize) -> Gate {
    let f: Vec<[f64; 3]> = rows.iter().map(feats).collect();
    let n = rows.len() as f64;
    let mean: [f64; 3] = std::array::from_fn(|j| f.iter().map(|x| x[j]).sum::<f64>() / n);
    let std: [f64; 3] = std::array::from_fn(|j| (f.iter().map(|x| (x[j] - mean[j]).powi(2)).sum::<f64>() / n).sqrt().max(1e-9));
    let mut g = Gate { w: [(init_lambda / (1.0 - init_lambda)).ln(), 0.0, 0.0, 0.0], mean, std };
    let mut step = 1.0;
    for _ in 0..iters {
        let (f0, grad) = objective(rows, &g);
        let g2: f64 = grad.iter().map(|x| x * x).sum();
        if g2 < 1e-16 {
            break;
        }
        loop {
            let mut next = g.clone();
            for j in 0..4 {
                next.w[j] -= step * grad[j];
            }
            if objective(rows, &next).0 <= f0 - 1e-4 * step * g2 {
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
```

- [ ] **Step 4: Run the tests**

Run: `cargo test --release --example gate_fit -j 8`
Expected: all 6 tests pass. If `gradient_matches_finite_differences` fails, re-derive `dl` (d loss / d pre-sigmoid = -(k - p) / m * lam * (1 - lam)) before touching the tolerance.

- [ ] **Step 5: Report in `main`**

Replace the last line of `main` (the `best fixed on even` println) and everything after it in `main` with:

```rust
    println!("best fixed on even = {lam:.2}: even {ce:.5}, odd {:.5}", mean_fixed(&odd, lam));
    let init = lam.clamp(0.01, 0.99);
    let base_odd = mean_fixed(&odd, 0.5);

    let b = Binned::fit(&even, 4, 4);
    let binned_odd = mean_gated(&odd, |r| b.lambda(r));
    println!("binned 4x4 (entropy x nearest distance, quantile edges from even): odd {binned_odd:.5} ({:+.5} vs fixed 0.5)", binned_odd - base_odd);
    println!("  lambda by bin (rows = entropy bins low->high, columns = nearest-distance bins low->high):");
    for i in 0..4 {
        println!("    {}", (0..4).map(|j| format!("{:.2}", b.lam[i * 4 + j])).collect::<Vec<_>>().join("  "));
    }
    println!("  entropy edges {:?}, d0 edges {:?}", b.h_edges.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>(), b.d_edges.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());

    let g = fit_gate(&even, init, 500);
    let param_odd = mean_gated(&odd, |r| g.lambda(r));
    println!("parametric gate: even {:.5}, odd {param_odd:.5} ({:+.5} vs fixed 0.5); w = {:?}", mean_gated(&even, |r| g.lambda(r)), param_odd - base_odd, g.w.map(|x| (x * 1000.0).round() / 1000.0));
    println!("  features: entropy, ln(1 + d0), (dk - d0) / (1 + d0), standardised on the even half: mean {:?}, std {:?}", g.mean, g.std);

    println!("verdict: binned gain {:.4} nats, parametric gain {:.4} nats (success: parametric > 0.002; kill: binned < 0.002)", base_odd - binned_odd, base_odd - param_odd);
}
```

(`main`'s closing brace is part of the replacement; keep `mod tests` after it unchanged.)

- [ ] **Step 6: Run on the slice dump and eyeball**

Run: `cargo run --release --example gate_fit -- runs/gate_slice.bin`
Expected: a lambda table, a parametric line and a verdict, all finite. The parametric even-half CE must not exceed the best-fixed even-half CE (it starts there and only accepts improving steps).

- [ ] **Step 7: Commit**

```bash
git add examples/tiny_lm/gate_fit.rs
git commit -m "gate_fit: binned and 4-parameter sigmoid gates, fitted on even chunks, scored on odd"
```

(plus the Co-Authored-By line)

---

### Task 4: Full-run dump, fit, and record

**Files:**
- Create: `scripts/tier_gate_dump.sh`
- Modify: `docs/tiers_design.md` (new entry immediately before `## Order`... place it with the other 2026-10-07 entries, i.e. just before the `## Fused matmul + threshold filter` heading, after the last dated bullet); `README.md` only if the gate succeeds (one sentence in "Direction: tiered AI")

**Interfaces:**
- Consumes: `tier_eval` `dump=` (Task 1) and `gate_fit` (Tasks 2-3).

- [ ] **Step 1: Write the script**

`scripts/tier_gate_dump.sh`:

```sh
#!/bin/sh
# Full held-out run of the deployable stack with a per-position dump for gate_fit (docs/superpowers/specs/2026-10-07-uncertainty-gating-design.md).
# Only the full-stack spec is scored (the knn search is shared, so this costs about the same as tier_full9).
export KNN_TIMING=1
E=runs/tier_eval_run.exe; C=runs/nov_big_k1_d0.1_8m_m0.ckpt
$E $C warm=32 stride=1 memory=online store=100000 \
  tier=lexicon:0.3+words:1:0.25+knn:256:0.5:15 dump=runs/tier_gate_dump.bin > runs/tier_gate_dump.log 2>&1
```

- [ ] **Step 2: Pre-launch checks (all must pass; do not launch otherwise)**

```powershell
Get-Process tier_eval_run,knn_scale_check,python -EA 0                      # nothing of ours left over
(Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 2 -MaxSamples 3).CounterSamples.CookedValue   # report; ideally under 20
Get-ChildItem "$env:LOCALAPPDATA\gpu-leases" -EA 0                           # no exclusive lease held by others
```

Also list eGPU users with the LONG_RUNS snippet. If the machine is busy, report the load with the result (tier_full9 ran at 36% CPU on average). Copy the fresh exe: `Copy-Item target\release\examples\tier_eval.exe runs\tier_eval_run.exe -Force`.

- [ ] **Step 3: Launch with the load logger**

```powershell
cd "C:\Users\Winston Li\Documents\GitHub\scratchtape"
$r = Start-Process "C:\Program Files\Git\bin\sh.exe" -ArgumentList 'scripts/tier_gate_dump.sh' -WindowStyle Hidden -PassThru
Start-Sleep 5; (Get-Process tier_eval_run).Id
Start-Process powershell -ArgumentList '-NoProfile','-File','scripts\load_log.ps1','tier_eval_run','runs\tier_gate_dump_load.log' -WindowStyle Hidden
```

Record the PID, start time, log paths and expected duration (~14 min) in `docs/tiers_design.md` as an "in flight" entry and commit it, so a new session can reattach. Expect `speed:` lines in `runs/tier_gate_dump.log`.

- [ ] **Step 4: Fit when the run exits**

Wait for `tier_eval_run` to exit (a Monitor with an until-loop on `tasklist`). Take `<x>` = the logged CE for the spec in `runs/tier_gate_dump.log` (expected 1.1313) and run:

`cargo run --release --example gate_fit -- runs/tier_gate_dump.bin logged=<x>`

Expected: `gate 0 passed`, then the baselines, the 4x4 lambda table, the parametric line and the verdict. If Gate 0 fails, stop and debug Task 1; do not interpret the rest.

- [ ] **Step 5: Record**

In `docs/tiers_design.md`, replace the in-flight entry with the result: run time, machine load, Gate 0, fixed-0.5 / best-fixed / binned / parametric odd-half CE, the lambda table, the fitted `w`, and the verdict against the success and kill criteria. If the parametric gate succeeds, add one sentence with the gain to README "Direction: tiered AI" and say what to try next (gating the CPU tiers); if the kill criterion triggers, say so plainly and what that rules out.

- [ ] **Step 6: Check for leftovers and commit**

`Get-Process tier_eval_run,gate_fit,python -EA 0` must show nothing of ours.

```bash
git add scripts/tier_gate_dump.sh docs/tiers_design.md README.md
git commit -m "Gate the kNN memory by uncertainty: dump, fit, result"
```

(plus the Co-Authored-By line; drop README.md from `git add` if unchanged)

---

## Self-review (done)

- **Spec coverage:** dump (Task 1); fit and score with the even/odd split, the fixed and best-fixed references, the binned table and the parametric gate (Tasks 2-3); Gate 0 and the gradient test (Tasks 2-3); success/kill verdict printed (Task 3 Step 5); record in docs (Task 4). Out-of-scope items are not planned.
- **Placeholders:** none; `<x>` and `<n>` in commands are values read from the run's log at execution time.
- **Types:** `Row`, `loss`, `mean_fixed`, `best_fixed`, `split`, `Binned::{fit, lambda}`, `Gate::{lambda, z}`, `fit_gate`, `objective`, `mean_gated` have the same signatures in the tests, `main` and the interface blocks.
