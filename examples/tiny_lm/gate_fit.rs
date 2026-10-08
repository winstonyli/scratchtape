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
