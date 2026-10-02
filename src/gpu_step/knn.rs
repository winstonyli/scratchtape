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
pub const KMAX: usize = 128;
const KM: u32 = KMAX as u32;
/// Queries per block (bounds the [tile, block] distance buffer).
const QBLOCK: usize = 2048;

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
        assert!(self.pending_vals.is_empty(), "call finish() before search");
        assert!(self.len() >= KMAX, "need at least {KMAX} keys");
        assert_eq!(queries.len(), rows * self.d);
        let mut out = Vec::with_capacity(rows * KMAX);
        for q0 in (0..rows).step_by(QBLOCK) {
            let q = QBLOCK.min(rows - q0);
            let block = &queries[q0 * self.d..(q0 + q) * self.d];
            let qh = upload_f32(block);
            let best_d = upload_f32(&vec![f32::INFINITY; q * KMAX]);
            let best_v = upload_f32(&vec![0.0f32; q * KMAX]);
            let dist = client().empty(self.tile * q * 4);
            for t in &self.tiles {
                let (a, b, o) = (MatRef::new(&t.keys), MatRef { trans: true, ..MatRef::new(&qh) }, MatRef::new(&dist));
                matmul(a, b, o, 1, t.len, self.d, q, Epilogue::default());
                super::count_launch();
                k_topk::launch(
                    client(),
                    cubes(q),
                    CubeDim::new_1d(EW_DIM),
                    buf(&dist, self.tile * q),
                    buf(&t.norms, t.len),
                    buf(&t.vals, t.len),
                    buf(&best_d, q * KMAX),
                    buf(&best_v, q * KMAX),
                    q as u32,
                    t.len as u32,
                );
            }
            let (bd, bv) = (read(&best_d), read(&best_v));
            for r in 0..q {
                let qn: f32 = block[r * self.d..(r + 1) * self.d].iter().map(|x| x * x).sum();
                for j in 0..KMAX {
                    out.push(((bd[r * KMAX + j] + qn).max(0.0), bv[r * KMAX + j] as u8));
                }
            }
        }
        out
    }
}

/// Merges one tile into each query's running top list (`best_*`, ascending, `KM` per query). `d` is the
/// tile's transposed matmul output: d[j * q + qi] = q·key_j.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_topk(d: &[f32], norms: &[f32], vals: &[f32], best_d: &mut [f32], best_v: &mut [f32], q: u32, t: u32) {
    let qi = ABSOLUTE_POS as u32;
    if qi < q {
        let base = qi * KM;
        let mut worst = best_d[(base + KM - 1) as usize];
        for j in 0..t {
            let dist = norms[j as usize] - 2.0 * d[(j * q + qi) as usize];
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
                assert_eq!(g.1, all[j].1, "query {r} rank {j}");
            }
        }
    }
}
