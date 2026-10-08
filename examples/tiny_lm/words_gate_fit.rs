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
