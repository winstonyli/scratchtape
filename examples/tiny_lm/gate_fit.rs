// Fits and scores per-position gates for the kNN memory's mixing weight from a `tier_eval dump=` file
// (docs/superpowers/specs/2026-10-07-uncertainty-gating-design.md).
//   gate_fit <dump>... [logged=<CE the run logged for the dumped spec>]
// Several dump files are pooled (file i's chunk ids are shifted by i * 100000); `logged=` (Gate 0) needs exactly one file.
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

/// Chunk-id offset between dump files: even, so chunk parity (the even/odd split) is preserved, and larger than any run's chunk count.
const FILE_CHUNK_OFFSET: u32 = 100_000;

/// All rows of several dump files, in file order; file i's chunk ids are shifted by i * FILE_CHUNK_OFFSET so files never share a chunk.
fn read_all(paths: &[String]) -> Vec<Row> {
    let mut all = vec![];
    for (i, p) in paths.iter().enumerate() {
        let off = i as u32 * FILE_CHUNK_OFFSET;
        all.extend(read_rows(p).into_iter().map(|r| Row { chunk: r.chunk + off, ..r }));
    }
    all
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

/// (earlier rows, later rows) in appearance order, cut at the first chunk-id change at or after the midpoint so no chunk is split.
fn split_halves(rows: &[Row]) -> (Vec<Row>, Vec<Row>) {
    let cut = (rows.len() / 2).max(1);
    let cut = (cut..rows.len()).find(|&i| rows[i].chunk != rows[i - 1].chunk).unwrap_or(rows.len());
    (rows[..cut].to_vec(), rows[cut..].to_vec())
}

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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (opts, files): (Vec<&String>, Vec<&String>) = args.iter().partition(|a| a.contains('='));
    assert!(!files.is_empty(), "usage: gate_fit <dump>... [logged=<CE>]");
    let files: Vec<String> = files.into_iter().cloned().collect();
    let rows = read_all(&files);
    assert!(!rows.is_empty(), "empty dump");
    let all = mean_fixed(&rows, 0.5);
    println!("{} rows from {} file(s); fixed lambda 0.5: CE {all:.5}", rows.len(), files.len());
    let mut checked = false;
    for a in opts {
        let logged: f64 = a.strip_prefix("logged=").unwrap_or_else(|| panic!("unknown argument {a}")).parse().unwrap();
        assert!(files.len() == 1, "logged= (Gate 0) checks one run's log against one dump; got {} dump files", files.len());
        assert!((all - logged).abs() < 1e-4, "gate 0: dump gives {all:.5}, the run logged {logged}");
        println!("gate 0 passed: dump reproduces the logged {logged}");
        checked = true;
    }
    if !checked {
        println!("gate 0 NOT checked (no logged=<CE>)");
    }
    let (even, odd) = split(&rows);
    report("even/odd", &even, &odd);
    let (early, late) = split_halves(&rows);
    report("early/late", &early, &late);
}

/// Fits the gates on `fit` and scores them on `score`; every line is prefixed with `label`.
fn report(label: &str, fit: &[Row], score: &[Row]) {
    if fit.is_empty() || score.is_empty() {
        println!("[{label}] skipped: fit half has {} rows, score half {}", fit.len(), score.len());
        return;
    }
    let (lam, ce) = best_fixed(fit);
    println!("[{label}] fit {} rows / score {} rows", fit.len(), score.len());
    println!("[{label}] fixed 0.5: fit {:.5}, score {:.5}", mean_fixed(fit, 0.5), mean_fixed(score, 0.5));
    println!("[{label}] best fixed on fit = {lam:.2}: fit {ce:.5}, score {:.5}", mean_fixed(score, lam));
    let init = lam.clamp(0.01, 0.99);
    let base_score = mean_fixed(score, 0.5);
    let base_best = mean_fixed(score, lam);

    let b = Binned::fit(fit, 4, 4);
    let binned_score = mean_gated(score, |r| b.lambda(r));
    println!("[{label}] binned 4x4 (entropy x nearest distance, quantile edges from fit): score {binned_score:.5} (gain {:+.5} vs fixed 0.5, {:+.5} vs best fixed {lam:.2})", base_score - binned_score, base_best - binned_score);
    println!("[{label}]   lambda by bin (rows = entropy bins low->high, columns = nearest-distance bins low->high):");
    for i in 0..4 {
        println!("[{label}]     {}", (0..4).map(|j| format!("{:.2}", b.lam[i * 4 + j])).collect::<Vec<_>>().join("  "));
    }
    println!("[{label}]   entropy edges {:?}, d0 edges {:?}", b.h_edges.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>(), b.d_edges.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());

    let g = fit_gate(fit, init, 500);
    let param_score = mean_gated(score, |r| g.lambda(r));
    println!("[{label}] parametric gate: fit {:.5}, score {param_score:.5} (gain {:+.5} vs fixed 0.5, {:+.5} vs best fixed {lam:.2}); w = {:?}", mean_gated(fit, |r| g.lambda(r)), base_score - param_score, base_best - param_score, g.w.map(|x| (x * 1000.0).round() / 1000.0));
    println!("[{label}]   features: entropy, ln(1 + d0), (dk - d0) / (1 + d0), standardised on the fit half: mean {:?}, std {:?}", g.mean, g.std);

    println!(
        "[{label}] verdict: binned gain {:.4} vs fixed 0.5, {:.4} vs best fixed; parametric gain {:.4} vs fixed 0.5, {:.4} vs best fixed (nats). Success (parametric > 0.002) and kill (binned < 0.002) apply to the gain vs best fixed, the baseline that excludes mere retuning of a constant; the vs-0.5 numbers are for reference.",
        base_score - binned_score,
        base_best - binned_score,
        base_score - param_score,
        base_best - param_score
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(chunk: u32, p: f64, k: f64) -> Row {
        Row { chunk, p, k, h: 0.0, d0: 1.0, dk: 2.0 }
    }

    fn write_dump(name: &str, rows: &[[f32; 6]]) -> String {
        let path = std::env::temp_dir().join(format!("gate_fit_test_{}_{name}.bin", std::process::id()));
        let bytes: Vec<u8> = rows.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        path.to_str().unwrap().to_string()
    }

    #[test]
    fn read_rows_round_trips_the_file_format() {
        let path = write_dump("roundtrip", &[[0.0, 0.5, 0.25, 1.5, 8.0, 12.0], [7.0, 0.125, 0.75, 0.0, 2.5, 3.0], [100.0, 1.0, 0.0625, 4.0, 0.5, 1.0]]);
        let rows = read_rows(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(rows.len(), 3);
        let want = [(0u32, 0.5, 0.25, 1.5, 8.0, 12.0), (7, 0.125, 0.75, 0.0, 2.5, 3.0), (100, 1.0, 0.0625, 4.0, 0.5, 1.0)];
        for (r, w) in rows.iter().zip(want) {
            assert_eq!((r.chunk, r.p, r.k, r.h, r.d0, r.dk), w);
        }
    }

    #[test]
    fn read_all_offsets_chunk_ids_per_file_and_keeps_parity() {
        let a = write_dump("all_a", &[[0.0, 0.5, 0.5, 0.0, 1.0, 2.0], [3.0, 0.5, 0.5, 0.0, 1.0, 2.0]]);
        let b = write_dump("all_b", &[[0.0, 0.5, 0.5, 0.0, 1.0, 2.0], [3.0, 0.5, 0.5, 0.0, 1.0, 2.0]]);
        let rows = read_all(&[a.clone(), b.clone()]);
        std::fs::remove_file(&a).unwrap();
        std::fs::remove_file(&b).unwrap();
        let ids: Vec<u32> = rows.iter().map(|r| r.chunk).collect();
        assert_eq!(ids, vec![0, 3, 100_000, 100_003]);
        assert!(rows.iter().zip([0u32, 3, 0, 3]).all(|(r, orig)| r.chunk % 2 == orig % 2));
    }

    #[test]
    fn split_halves_never_splits_a_chunk_and_keeps_order() {
        // 3 rows per chunk, 5 chunks: the midpoint (row 7) is inside chunk 2, so the cut moves to the start of chunk 3.
        let rows: Vec<Row> = (0..15).map(|i| row(i / 3, i as f64 / 100.0, 0.5)).collect();
        let (early, late) = split_halves(&rows);
        assert_eq!(early.len() + late.len(), rows.len());
        assert_eq!(early.len(), 9);
        let joined: Vec<f64> = early.iter().chain(&late).map(|r| r.p).collect();
        assert_eq!(joined, rows.iter().map(|r| r.p).collect::<Vec<_>>());
        assert!(early.last().unwrap().chunk < late.first().unwrap().chunk);
        // One chunk only: nothing to cut, the late half is empty.
        let one: Vec<Row> = (0..4).map(|_| row(7, 0.5, 0.5)).collect();
        assert_eq!(split_halves(&one).1.len(), 0);
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
}
