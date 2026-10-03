//! Brute-force k-nearest-neighbour search on the device, for the memory tier
//! (docs/tiers_design.md, Experiment B): the keys live on the GPU in tiles,
//! and a query block's distances to one tile come from one matmul, then a
//! kernel keeps each query's `KMAX` nearest in a running top list.
//! dist(q, key) = |key|² − 2 q·key (+ |q|², added on the host at the end).
//! The matmul writes the tile's distances transposed ([keys, queries]) so the
//! top-k kernel, one thread per query, reads them coalesced.
use super::matmul::{Epilogue, MatRef, matmul};
use super::{EW_DIM, buf, client, cubes, read, upload_f32};
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Neighbours kept per query.
pub const KMAX: usize = 256;
const KM: u32 = KMAX as u32;
/// Queries per block (bounds the [tile, block] distance buffer).
const QBLOCK: usize = 2048;
/// Threads per query: each scans its own contiguous part of a slice into its own top list, and the lists are merged on
/// the host. One thread per query left the GPU idle (the scan is serial over millions of keys; docs/tiers_design.md).
const PARTS_DEFAULT: usize = 8;
/// Keys per top-k launch: one launch must stay short (a long one trips the OS's GPU watchdog; with KMAX = 256 a
/// whole 16384-key tile lost the device when one thread scanned all of it). Each thread scans SLICE / PARTS keys.
const SLICE: usize = 4096;

struct Tile {
    keys: Handle,
    norms: Handle,
    vals: Handle,
    len: usize,
}

/// Keys (vectors of width `d`) with a byte value each, in device tiles of `tile` keys.
pub struct KnnStore {
    d: usize,
    tile: usize,
    tiles: Vec<Tile>,
    pending_keys: Vec<f32>,
    pending_vals: Vec<u8>,
}

impl KnnStore {
    pub fn new(d: usize, tile: usize) -> Self {
        KnnStore { d, tile, tiles: vec![], pending_keys: vec![], pending_vals: vec![] }
    }

    /// Appends keys (row-major [n, d]) and their values; full tiles are uploaded as they fill.
    pub fn add(&mut self, keys: &[f32], vals: &[u8]) {
        assert_eq!(keys.len(), vals.len() * self.d);
        self.pending_keys.extend_from_slice(keys);
        self.pending_vals.extend_from_slice(vals);
        while self.pending_vals.len() >= self.tile {
            self.flush(self.tile);
        }
    }

    /// Uploads whatever is pending as a final, smaller tile. Call once after the last `add`.
    pub fn finish(&mut self) {
        let n = self.pending_vals.len();
        if n > 0 {
            self.flush(n);
        }
    }

    fn flush(&mut self, n: usize) {
        let keys: Vec<f32> = self.pending_keys.drain(..n * self.d).collect();
        let vals: Vec<u8> = self.pending_vals.drain(..n).collect();
        let norms: Vec<f32> = keys.chunks_exact(self.d).map(|k| k.iter().map(|x| x * x).sum()).collect();
        let vals_f: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
        self.tiles.push(Tile { keys: upload_f32(&keys), norms: upload_f32(&norms), vals: upload_f32(&vals_f), len: n });
    }

    pub fn len(&self) -> usize {
        self.tiles.iter().map(|t| t.len).sum()
    }

    /// For each of `rows` queries (row-major [rows, d]) the `KMAX` nearest keys, ascending: (squared L2 distance, value).
    pub fn search(&self, queries: &[f32], rows: usize) -> Vec<(f32, u8)> {
        self.search_masked(queries, rows, None)
    }

    /// `search`, but query r may only match keys j (in the order they were added) with j < `always` or
    /// lo_r <= j < hi_r, for `mask = Some((ranges, always))`, ranges[r] = (lo_r, hi_r). With fewer than `KMAX` allowed
    /// keys the list ends in (infinity, 0) entries. This is how a memory is made causal: a query sees only keys older than it.
    pub fn search_masked(&self, queries: &[f32], rows: usize, mask: Option<(&[(u32, u32)], u32)>) -> Vec<(f32, u8)> {
        assert!(self.pending_vals.is_empty(), "call finish() before search");
        assert!(self.len() >= KMAX, "need at least {KMAX} keys");
        assert_eq!(queries.len(), rows * self.d);
        let (ranges, always) = match mask {
            Some((ranges, always)) => (ranges.to_vec(), always),
            None => (vec![(0, u32::MAX); rows], 0),
        };
        assert_eq!(ranges.len(), rows);
        let parts: usize = std::env::var("KNN_PARTS").ok().and_then(|v| v.parse().ok()).unwrap_or(PARTS_DEFAULT);
        let mut out = Vec::with_capacity(rows * KMAX);
        for q0 in (0..rows).step_by(QBLOCK) {
            let q = QBLOCK.min(rows - q0);
            let block = &queries[q0 * self.d..(q0 + q) * self.d];
            let qh = upload_f32(block);
            let best_d = upload_f32(&vec![f32::INFINITY; parts * q * KMAX]);
            let best_v = upload_f32(&vec![0.0f32; parts * q * KMAX]);
            let dist = client().empty(self.tile * q * 4);
            let lo = client().create_from_slice(u32::as_bytes(&ranges[q0..q0 + q].iter().map(|r| r.0).collect::<Vec<_>>()));
            let hi = client().create_from_slice(u32::as_bytes(&ranges[q0..q0 + q].iter().map(|r| r.1).collect::<Vec<_>>()));
            let mut base = 0usize;
            // KNN_TIMING=1: sync between stages (a read of a small buffer waits for the queue) and report their wall time.
            let timing = std::env::var_os("KNN_TIMING").is_some();
            let (mut t_mm, mut t_topk) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
            let t_start = std::time::Instant::now();
            for t in &self.tiles {
                let t0 = std::time::Instant::now();
                let (a, b, o) = (MatRef::new(&t.keys), MatRef { trans: true, ..MatRef::new(&qh) }, MatRef::new(&dist));
                matmul(a, b, o, 1, t.len, self.d, q, Epilogue::default());
                if timing {
                    read(&hi);
                    t_mm += t0.elapsed();
                }
                let t0 = std::time::Instant::now();
                for j0 in (0..t.len).step_by(SLICE) {
                    super::count_launch();
                    let n = SLICE.min(t.len - j0);
                    #[rustfmt::skip]
                    k_topk::launch(client(), cubes(parts * q), CubeDim::new_1d(EW_DIM), buf(&dist, self.tile * q), buf(&t.norms, t.len), buf(&t.vals, t.len), buf(&best_d, parts * q * KMAX), buf(&best_v, parts * q * KMAX), buf(&lo, q), buf(&hi, q), always, q as u32, parts as u32, (base + j0) as u32, j0 as u32, n as u32);
                }
                if timing {
                    read(&hi);
                    t_topk += t0.elapsed();
                }
                base += t.len;
            }
            let t0 = std::time::Instant::now();
            let (bd, bv) = (read(&best_d), read(&best_v));
            let t_read = t0.elapsed();
            if timing {
                eprintln!(
                    "knn timing: {q} queries x {} keys: matmul {:.0} ms, top-k {:.0} ms, readback {:.0} ms, total {:.0} ms",
                    self.len(),
                    t_mm.as_secs_f64() * 1e3,
                    t_topk.as_secs_f64() * 1e3,
                    t_read.as_secs_f64() * 1e3,
                    t_start.elapsed().as_secs_f64() * 1e3
                );
            }
            for r in 0..q {
                let qn: f32 = block[r * self.d..(r + 1) * self.d].iter().map(|x| x * x).sum();
                let mut c: Vec<(f32, f32)> = (0..parts).flat_map(|p| (0..KMAX).map(move |j| (p * q + r) * KMAX + j)).map(|i| (bd[i], bv[i])).collect();
                c.select_nth_unstable_by(KMAX - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
                c.truncate(KMAX);
                c.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                out.extend(c.iter().map(|&(dd, v)| ((dd + qn).max(0.0), v as u8)));
            }
        }
        out
    }
}

/// Merges one tile into each query's running top list (`best_*`, ascending, `KM` per query). `d` is the
/// tile's transposed matmul output: d[j * q + qi] = q·key_j; this launch covers the tile's keys j0..j0 + len, whose
/// store-wide indices start at gj0 (keys outside a query's allowed ranges count as infinitely far).
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_topk(d: &[f32], norms: &[f32], vals: &[f32], best_d: &mut [f32], best_v: &mut [f32], lo: &[u32], hi: &[u32], always: u32, q: u32, parts: u32, gj0: u32, j0: u32, len: u32) {
    // Thread t scans part t / q of the slice for query t % q (neighbouring threads: neighbouring queries, coalesced).
    let t = ABSOLUTE_POS as u32;
    if t < q * parts {
        let qi = t % q;
        let chunk = (len + parts - 1) / parts;
        let start = (t / q) * chunk;
        let base = t * KM;
        let mut worst = best_d[(base + KM - 1) as usize];
        let (lo_q, hi_q) = (lo[qi as usize], hi[qi as usize]);
        for jl in 0..chunk {
            let jj = start + jl;
            if jj < len {
                let j = j0 + jj;
                let g = gj0 + jj;
                let mut dist = norms[j as usize] - 2.0 * d[(j * q + qi) as usize];
                if g >= always {
                    if g < lo_q {
                        dist = f32::INFINITY;
                    }
                    if g >= hi_q {
                        dist = f32::INFINITY;
                    }
                }
                if dist < worst {
                    // Insert at the rank of `dist` (the number of kept entries not above it), shifting the tail down.
                    let mut p = 0u32;
                    for i in 0..KM {
                        if best_d[(base + i) as usize] <= dist {
                            p += 1;
                        }
                    }
                    for s in 0..KM - 1 - p {
                        let i = KM - 1 - s;
                        best_d[(base + i) as usize] = best_d[(base + i - 1) as usize];
                        best_v[(base + i) as usize] = best_v[(base + i - 1) as usize];
                    }
                    best_d[(base + p) as usize] = dist;
                    best_v[(base + p) as usize] = vals[j as usize];
                    worst = best_d[(base + KM - 1) as usize];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Rng;

    #[test]
    fn matches_cpu_brute_force() {
        // 1000 keys in tiles of 300 (a partial last tile), 70 queries, width 24: not multiples of the matmul tile.
        let (d, n, rows) = (24, 1000, 70);
        let mut rng = Rng::new(7);
        let keys: Vec<f32> = (0..n * d).map(|_| rng.next_gaussian()).collect();
        let vals: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let queries: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        let mut store = KnnStore::new(d, 300);
        store.add(&keys[..400 * d], &vals[..400]);
        store.add(&keys[400 * d..], &vals[400..]);
        store.finish();
        let got = store.search(&queries, rows);
        for r in 0..rows {
            let q = &queries[r * d..(r + 1) * d];
            let mut all: Vec<(f32, u8)> = (0..n).map(|i| (keys[i * d..(i + 1) * d].iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum(), vals[i])).collect();
            all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for j in 0..KMAX {
                let g = got[r * KMAX + j];
                assert!((g.0 - all[j].0).abs() < 1e-3 * (1.0 + all[j].0), "query {r} rank {j}: gpu {} vs cpu {}", g.0, all[j].0);
                // Values can differ only between near-tied distances (float rounding reorders those).
                let gap = |a: usize, b: usize| (all[a].0 - all[b].0).abs() > 1e-3 * (1.0 + all[a].0);
                if (j == 0 || gap(j, j - 1)) && (j + 1 == n || gap(j, j + 1)) {
                    assert_eq!(g.1, all[j].1, "query {r} rank {j}");
                }
            }
        }
    }

    #[test]
    fn masked_search_respects_ranges() {
        let (d, n, rows) = (24, 1000, 40);
        let mut rng = Rng::new(11);
        let keys: Vec<f32> = (0..n * d).map(|_| rng.next_gaussian()).collect();
        let vals: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let queries: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        let mut store = KnnStore::new(d, 300);
        store.add(&keys, &vals);
        store.finish();
        // Keys below 100 are always allowed; query r also sees [lo_r, hi_r), some with fewer than KMAX allowed keys.
        let ranges: Vec<(u32, u32)> = (0..rows).map(|r| (200 + (r as u32 * 17) % 400, 300 + (r as u32 * 29) % 700)).collect();
        let got = store.search_masked(&queries, rows, Some((&ranges, 100)));
        for r in 0..rows {
            let q = &queries[r * d..(r + 1) * d];
            let mut all: Vec<(f32, u8)> = (0..n)
                .filter(|&i| (i as u32) < 100 || (i as u32 >= ranges[r].0 && (i as u32) < ranges[r].1))
                .map(|i| (keys[i * d..(i + 1) * d].iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum(), vals[i]))
                .collect();
            all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for j in 0..KMAX {
                let g = got[r * KMAX + j];
                match all.get(j) {
                    Some(c) => assert!((g.0 - c.0).abs() < 1e-3 * (1.0 + c.0), "query {r} rank {j}: gpu {} vs cpu {}", g.0, c.0),
                    None => assert!(g.0.is_infinite(), "query {r} rank {j}: expected no neighbour, got {}", g.0),
                }
            }
        }
    }
}
