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
use half::f16;
use std::cell::RefCell;

/// Neighbours kept per query.
pub const KMAX: usize = 256;
const KM: u32 = KMAX as u32;
/// Queries per block (bounds the [tile, block] distance buffer).
const QBLOCK: usize = 2048;
/// Threads per query: each scans its own contiguous part of a slice into its own top list, and the lists are merged on
/// the host. One thread per query left the GPU idle (the scan is serial over millions of keys; docs/tiers_design.md).
/// Threads in flight per top-k launch to aim for: a block of q queries uses `THREADS / q` parts (clamped to 8..=32; more parts
/// cost host merge time). Measured on 1024-query blocks: 8 parts 3.7 s, 16: 2.9 s, 32: 2.3 s of top-k (contended).
const THREADS: usize = 32768;
/// Keys per top-k launch: one launch must stay short (a long one trips the OS's GPU watchdog; with KMAX = 256 a
/// whole 16384-key tile lost the device when one thread scanned all of it). Each thread scans SLICE / PARTS keys.
const SLICE: usize = 16384;

/// Every `SAMPLE_STRIDE`-th key (by index; coprime to the 64-byte window length, so the sample covers all window positions) is also kept on the host. A query's distances to the sample give its threshold:
/// the `THRESH_RANK`-th nearest sample key is about the `THRESH_RANK * SAMPLE_STRIDE`-th nearest key overall, so only keys
/// closer than that can enter the top `KMAX` (~4x fewer list insertions than starting from an empty list; if fewer than
/// `KMAX` keys qualify, the block is searched again without thresholds, so results stay exact).
const SAMPLE_STRIDE: usize = 61;
const SS: u32 = SAMPLE_STRIDE as u32;
const THRESH_RANK: usize = 32;
const TR: u32 = THRESH_RANK as u32;

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
    sample_keys: Vec<f32>,
    /// Keys and queries as f16 on the matrix cores (the default; `KNN_F32=1` selects the exact f32 path): about 2x faster matmul and half the key memory, but
    /// distances carry f16 rounding (~1e-3 relative); norms are taken from the rounded vectors so a distance is exactly
    /// the squared distance between the rounded vectors.
    f16: bool,
    /// Wall time spent in `flush` so far: [host copy + norms + upload, unused] (diagnostic).
    pub flush_time: [std::time::Duration; 2],
    /// Device copy of `sample_keys` as (keys, norms, len) chunks, with the sample count it was built from.
    sample_dev: RefCell<(usize, Vec<(Handle, Handle, usize)>)>,
}

impl KnnStore {
    pub fn new(d: usize, tile: usize) -> Self {
        // f16 matrix cores unless `KNN_F32` is set, or the shapes or the device do not allow them.
        let f16 = std::env::var_os("KNN_F32").is_none() && d % 16 == 0 && tile % 16 == 0 && client().features().matmul.cmma.contains(&f16_config());
        Self::with_precision(d, tile, f16)
    }

    pub fn with_precision(d: usize, tile: usize, f16: bool) -> Self {
        if f16 {
            assert!(d % 16 == 0 && tile % 16 == 0, "f16 matrix cores need d and tile to be multiples of 16");
            assert!(client().features().matmul.cmma.contains(&f16_config()), "this device reports no f16 x f16 -> f32 matrix-core configuration");
        }
        KnnStore {
            d,
            tile,
            tiles: vec![],
            pending_keys: vec![],
            pending_vals: vec![],
            sample_keys: vec![],
            f16,
            flush_time: [std::time::Duration::ZERO; 2],
            sample_dev: RefCell::new((0, vec![])),
        }
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

    /// Uploads whatever is pending as a final, smaller tile. Optional before a search (which handles pending keys), but
    /// call it once after the last `add` of a store that is built once and searched many times.
    pub fn finish(&mut self) {
        let n = self.pending_vals.len();
        if n > 0 {
            self.flush(n);
        }
    }

    fn flush(&mut self, n: usize) {
        let t0 = std::time::Instant::now();
        let keys: Vec<f32> = self.pending_keys.drain(..n * self.d).collect();
        let first = self.uploaded();
        for i in (first.next_multiple_of(SAMPLE_STRIDE) - first..n).step_by(SAMPLE_STRIDE) {
            self.sample_keys.extend_from_slice(&keys[i * self.d..(i + 1) * self.d]);
        }
        let vals: Vec<u8> = self.pending_vals.drain(..n).collect();
        let tile = self.tile_of(&keys, &vals);
        self.flush_time[0] += t0.elapsed();
        self.tiles.push(tile);
    }

    /// Device tile for `keys` (row-major [vals.len(), d]) and their values.
    fn tile_of(&self, keys: &[f32], vals: &[u8]) -> Tile {
        let vals_f: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
        let (key_h, norms) = self.upload_keys(keys);
        Tile { keys: key_h, norms: upload_f32(&norms), vals: upload_f32(&vals_f), len: vals.len() }
    }

    /// Device copy of `keys` (row-major [n, d]) in the store's precision, and the norms of what was stored. The f16 copy
    /// is padded with zero rows to a multiple of 16 (the matrix-core tile).
    fn upload_keys(&self, keys: &[f32]) -> (Handle, Vec<f32>) {
        if self.f16 {
            let mut k16: Vec<f16> = keys.iter().map(|&x| f16::from_f32(x)).collect();
            let norms = k16.chunks_exact(self.d).map(|k| k.iter().map(|x| x.to_f32() * x.to_f32()).sum()).collect();
            k16.resize(keys.len().div_ceil(self.d).next_multiple_of(16) * self.d, f16::ZERO);
            (client().create_from_slice(f16::as_bytes(&k16)), norms)
        } else {
            (upload_f32(keys), keys.chunks_exact(self.d).map(|k| k.iter().map(|x| x * x).sum()).collect())
        }
    }

    /// Whether keys and queries are f16 on the matrix cores.
    pub fn is_f16(&self) -> bool {
        self.f16
    }

    /// Keys added so far, uploaded or still pending.
    pub fn len(&self) -> usize {
        self.uploaded() + self.pending_vals.len()
    }

    fn uploaded(&self) -> usize {
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
        assert!(self.len() >= KMAX, "need at least {KMAX} keys");
        assert_eq!(queries.len(), rows * self.d);
        let (ranges, always) = match mask {
            Some((ranges, always)) => (ranges.to_vec(), always),
            None => (vec![(0, u32::MAX); rows], 0),
        };
        assert_eq!(ranges.len(), rows);
        // Keys not yet in a full tile are searched as a transient last tile (an online memory adds a few keys at a time;
        // uploading each addition as its own tile made every search launch for thousands of tiny tiles).
        let tail = (!self.pending_vals.is_empty()).then(|| self.tile_of(&self.pending_keys, &self.pending_vals));
        let forced: Option<usize> = std::env::var("KNN_PARTS").ok().and_then(|v| v.parse().ok());
        let mut out = Vec::with_capacity(rows * KMAX);
        for q0 in (0..rows).step_by(QBLOCK) {
            let q = QBLOCK.min(rows - q0);
            let block = &queries[q0 * self.d..(q0 + q) * self.d];
            let rg = &ranges[q0..q0 + q];
            let parts = forced.unwrap_or((THREADS / q).clamp(8, 32));
            let found = self.search_block(block, q, rg, always, parts, true, tail.as_ref()).unwrap_or_else(|| self.search_block(block, q, rg, always, parts, false, tail.as_ref()).unwrap());
            out.extend(found);
        }
        out
    }

    /// dist[j, i] = key_j . query_i for the `len` keys in `keys`, as [rows, qp] (qp = q unless f16-padded).
    fn dots(&self, keys: &Handle, len: usize, qh: &Handle, dist: &Handle, q: usize, qp: usize) {
        if self.f16 {
            let rows = len.next_multiple_of(16);
            let plane = client().properties().hardware.plane_size_max;
            #[rustfmt::skip]
            k_dots_cmma::launch(client(), CubeCount::Static((qp / 16) as u32, (rows / 16) as u32, 1), CubeDim::new_1d(plane), buf(keys, rows * self.d), buf(qh, self.d * qp), buf(dist, self.tile * qp), self.d as u32, qp as u32);
        } else {
            let (a, b, o) = (MatRef::new(keys), MatRef { trans: true, ..MatRef::new(qh) }, MatRef::new(dist));
            matmul(a, b, o, 1, len, self.d, q, Epilogue::default());
        }
    }

    /// Uploads the sample keys if more arrived since the last call: full chunks stay on the device (the sample only
    /// grows at the end), so an online memory re-uploads just the last partial chunk.
    fn sample_chunks(&self) -> Vec<(Handle, Handle, usize)> {
        let n = self.sample_keys.len() / self.d;
        let mut dev = self.sample_dev.borrow_mut();
        if dev.0 != n {
            let keep = dev.0 / self.tile;
            dev.1.truncate(keep);
            for c in self.sample_keys[keep * self.tile * self.d..].chunks(self.tile * self.d) {
                let (keys, norms) = self.upload_keys(c);
                dev.1.push((keys, upload_f32(&norms), norms.len()));
            }
            dev.0 = n;
        }
        dev.1.clone()
    }

    /// One block of `q` queries. With `thresholds`, returns None when some query's threshold left fewer than `KMAX` keys.
    #[allow(clippy::too_many_arguments)]
    fn search_block(&self, block: &[f32], q: usize, ranges: &[(u32, u32)], always: u32, parts: usize, thresholds: bool, tail: Option<&Tile>) -> Option<Vec<(f32, u8)>> {
        // Queries as the matmul wants them: f32 [q, d], or (f16) rounded, transposed to [d, qp] and padded to a multiple of 16.
        let qp = if self.f16 { q.next_multiple_of(16) } else { q };
        let rounded: Vec<f32>;
        let (qh, block) = if self.f16 {
            let mut t = vec![f16::ZERO; self.d * qp];
            for r in 0..q {
                for c in 0..self.d {
                    t[c * qp + r] = f16::from_f32(block[r * self.d + c]);
                }
            }
            rounded = (0..q * self.d).map(|i| t[(i % self.d) * qp + i / self.d].to_f32()).collect();
            (client().create_from_slice(f16::as_bytes(&t)), &rounded[..])
        } else {
            (upload_f32(block), block)
        };
        let best_d = upload_f32(&vec![f32::INFINITY; parts * q * KMAX]);
        let best_v = upload_f32(&vec![0.0f32; parts * q * KMAX]);
        let dist = client().empty(self.tile * qp * 4);
        let lo = client().create_from_slice(u32::as_bytes(&ranges.iter().map(|r| r.0).collect::<Vec<_>>()));
        let hi = client().create_from_slice(u32::as_bytes(&ranges.iter().map(|r| r.1).collect::<Vec<_>>()));
        // KNN_TIMING=1: sync between stages (a read of a small buffer waits for the queue) and report their wall time.
        let timing = std::env::var_os("KNN_TIMING").is_some();
        let (mut t_mm, mut t_topk) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        if timing {
            read(&hi);
        }
        let t_start = std::time::Instant::now();
        let mut stage = [std::time::Duration::ZERO; 3];
        // Pass 1: each query's threshold from the sample.
        let thr_d = upload_f32(&vec![f32::INFINITY; parts * q * THRESH_RANK]);
        let sample = if thresholds { self.sample_chunks() } else { vec![] };
        if timing {
            read(&hi);
            stage[0] = t_start.elapsed();
        }
        let mut gbase = 0usize;
        for (sk, sn, len) in &sample {
            let t0 = std::time::Instant::now();
            self.dots(sk, *len, &qh, &dist, q, qp);
            if timing {
                read(&hi);
                stage[1] += t0.elapsed();
            }
            super::count_launch();
            #[rustfmt::skip]
            k_thresh::launch(client(), cubes(parts * q), CubeDim::new_1d(EW_DIM), buf(&dist, self.tile * qp), buf(sn, *len), buf(&thr_d, parts * q * THRESH_RANK), buf(&lo, q), buf(&hi, q), always, q as u32, qp as u32, parts as u32, gbase as u32, *len as u32);
            gbase += len;
        }
        let t0 = std::time::Instant::now();
        let thr = if thresholds { read(&thr_d) } else { vec![] };
        // The kernel's starting threshold per query (infinity: none).
        let thr_q: Vec<f32> = (0..q).map(|r| if thresholds { widen(rank_of_parts(&thr, r, q, parts)) } else { f32::INFINITY }).collect();
        let thr_buf = upload_f32(&thr_q);
        stage[2] = t0.elapsed();
        let t_pass1 = t_start.elapsed();
        if timing {
            eprintln!(
                "knn pass 1: sample upload {:.0} ms, sample matmuls {:.0} ms, k_thresh + readback {:.0} ms",
                stage[0].as_secs_f64() * 1e3,
                stage[1].as_secs_f64() * 1e3,
                stage[2].as_secs_f64() * 1e3
            );
        }
        if timing {
            t_mm += t_pass1;
        }
        // `KNN_SLICE` overrides the keys per top-k launch (experiments).
        let slice: usize = std::env::var("KNN_SLICE").ok().and_then(|v| v.parse().ok()).unwrap_or(SLICE);
        let mut base = 0usize;
        for t in self.tiles.iter().chain(tail) {
            let t0 = std::time::Instant::now();
            self.dots(&t.keys, t.len, &qh, &dist, q, qp);
            if timing {
                read(&hi);
                t_mm += t0.elapsed();
            }
            let t0 = std::time::Instant::now();
            for j0 in (0..t.len).step_by(slice) {
                super::count_launch();
                let n = slice.min(t.len - j0);
                #[rustfmt::skip]
                k_topk::launch(client(), cubes(parts * q), CubeDim::new_1d(EW_DIM), buf(&dist, self.tile * qp), buf(&t.norms, t.len), buf(&t.vals, t.len), buf(&best_d, parts * q * KMAX), buf(&best_v, parts * q * KMAX), buf(&lo, q), buf(&hi, q), buf(&thr_buf, q), always, q as u32, qp as u32, parts as u32, (base + j0) as u32, j0 as u32, n as u32);
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
                "knn timing: {q} queries x {} keys{}: matmul+thresholds {:.0} ms (thresholds alone {:.0}), top-k {:.0} ms, readback {:.0} ms, total {:.0} ms",
                self.len(),
                if thresholds { "" } else { " (no thresholds)" },
                t_mm.as_secs_f64() * 1e3,
                t_pass1.as_secs_f64() * 1e3,
                t_topk.as_secs_f64() * 1e3,
                t_read.as_secs_f64() * 1e3,
                t_start.elapsed().as_secs_f64() * 1e3
            );
        }
        let mut out = Vec::with_capacity(q * KMAX);
        for r in 0..q {
            let qn: f32 = block[r * self.d..(r + 1) * self.d].iter().map(|x| x * x).sum();
            let mut c: Vec<(f32, f32)> = (0..parts).flat_map(|p| (0..KMAX).map(move |j| (p * q + r) * KMAX + j)).map(|i| (bd[i], bv[i])).collect();
            c.select_nth_unstable_by(KMAX - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            c.truncate(KMAX);
            c.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            if thr_q[r].is_finite() && c[KMAX - 1].0.is_infinite() {
                if timing {
                    eprintln!("knn: query {r} had fewer than {KMAX} keys under its threshold {}", thr_q[r]);
                }
                return None;
            }
            out.extend(c.iter().map(|&(dd, v)| ((dd + qn).max(0.0), v as u8)));
        }
        Some(out)
    }
}

/// A threshold just above `t`, so keys at exactly that distance (duplicate keys are common: a window's first position
/// sees only its own byte) still qualify under the kernel's strict `<`.
fn widen(t: f32) -> f32 {
    t + 1e-4 * (t.abs() + 1.0)
}

fn f16_config() -> cubecl::features::MmaConfig {
    cubecl::features::MmaConfig {
        a_type: cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16),
        b_type: cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16),
        cd_type: cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        m: 16,
        k: 16,
        n: 16,
    }
}

/// out[M, N] (f32) = a[M, K] @ b[K, N] (f16, row-major) on matrix cores: one plane per cube computes one 16x16 tile
/// (the spike's `k_matmul_cmma`). M, N, K are multiples of 16.
#[cube(launch)]
fn k_dots_cmma(a: &[f16], b: &[f16], out: &mut [f32], #[comptime] k: u32, n: u32) {
    let row0 = CUBE_POS_Y * 16;
    let col0 = CUBE_POS_X * 16;
    let c = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    #[unroll]
    for kk in 0..k / 16 {
        let a_off = (row0 * k + kk * 16) as usize;
        let b_off = (kk * 16 * n + col0) as usize;
        let at = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &a[a_off..a.len()], k);
        let bt = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &b[b_off..b.len()], n);
        cmma::execute(&at, &bt, &c, &c);
    }
    let o_off = (row0 * n + col0) as usize;
    let len = out.len();
    cmma::store(&mut out[o_off..len], &c, n, cmma::MatrixLayout::RowMajor);
}

/// The `THRESH_RANK`-th smallest of query r's per-part lists in `thr` (`parts` lists of `THRESH_RANK`, thread order).
fn rank_of_parts(thr: &[f32], r: usize, q: usize, parts: usize) -> f32 {
    let mut c: Vec<f32> = (0..parts).flat_map(|p| thr[(p * q + r) * THRESH_RANK..][..THRESH_RANK].iter().copied()).collect();
    *c.select_nth_unstable_by(THRESH_RANK - 1, |a, b| a.partial_cmp(b).unwrap()).1
}

/// Pass 1: the `THRESH_RANK` nearest sample keys of each (query, part) thread (ascending, in `best`, `TR` per thread;
/// thread t scans part t / q of the chunk for query t % q, as in `k_topk`; the host merges the parts). The sample chunk's
/// key i has store-wide index (gs0 + i) * SAMPLE_STRIDE; keys outside the query's allowed ranges are skipped.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_thresh(d: &[f32], norms: &[f32], best: &mut [f32], lo: &[u32], hi: &[u32], always: u32, q: u32, ld: u32, parts: u32, gs0: u32, len: u32) {
    let t = ABSOLUTE_POS as u32;
    if t < q * parts {
        let qi = t % q;
        let chunk = (len + parts - 1) / parts;
        let start = (t / q) * chunk;
        let base = t * TR;
        let (lo_q, hi_q) = (lo[qi as usize], hi[qi as usize]);
        for jl in 0..chunk {
            let j = start + jl;
            if j < len {
                let g = (gs0 + j) * SS;
                let mut dist = norms[j as usize] - 2.0 * d[(j * ld + qi) as usize];
                if g >= always {
                    if g < lo_q {
                        dist = f32::INFINITY;
                    }
                    if g >= hi_q {
                        dist = f32::INFINITY;
                    }
                }
                if dist < best[(base + TR - 1) as usize] {
                    let mut p = 0u32;
                    for i in 0..TR {
                        if best[(base + i) as usize] <= dist {
                            p += 1;
                        }
                    }
                    for s in 0..TR - 1 - p {
                        let i = TR - 1 - s;
                        best[(base + i) as usize] = best[(base + i - 1) as usize];
                    }
                    best[(base + p) as usize] = dist;
                }
            }
        }
    }
}

/// Merges one tile into each query's running top list (`best_*`, ascending, `KM` per query). `d` is the
/// tile's transposed matmul output: d[j * q + qi] = q·key_j; this launch covers the tile's keys j0..j0 + len, whose
/// store-wide indices start at gj0 (keys outside a query's allowed ranges count as infinitely far).
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_topk(
    d: &[f32],
    norms: &[f32],
    vals: &[f32],
    best_d: &mut [f32],
    best_v: &mut [f32],
    lo: &[u32],
    hi: &[u32],
    thr: &[f32],
    always: u32,
    q: u32,
    ld: u32,
    parts: u32,
    gj0: u32,
    j0: u32,
    len: u32,
) {
    // Thread t scans part t / q of the slice for query t % q (neighbouring threads: neighbouring queries, coalesced).
    let t = ABSOLUTE_POS as u32;
    if t < q * parts {
        let qi = t % q;
        let chunk = (len + parts - 1) / parts;
        let start = (t / q) * chunk;
        let base = t * KM;
        let thr_q = thr[qi as usize];
        let mut worst = best_d[(base + KM - 1) as usize];
        if thr_q < worst {
            worst = thr_q;
        }
        let (lo_q, hi_q) = (lo[qi as usize], hi[qi as usize]);
        // Live entries in the list (the rest is INF padding, which an insertion need not move): the first INF, by binary
        // search, since the list is sorted and earlier slices or tiles may have filled part of it.
        let mut c_lo = 0u32;
        let mut c_hi = c_lo + KM;
        for _it in 0..9 {
            if c_lo < c_hi {
                let mid = (c_lo + c_hi) / 2;
                if best_d[(base + mid) as usize] < f32::INFINITY {
                    c_lo = mid + 1;
                } else {
                    c_hi = mid;
                }
            }
        }
        let mut cnt = c_lo;
        for jl in 0..chunk {
            let jj = start + jl;
            if jj < len {
                let j = j0 + jj;
                let g = gj0 + jj;
                let mut dist = norms[j as usize] - 2.0 * d[(j * ld + qi) as usize];
                if g >= always {
                    if g < lo_q {
                        dist = f32::INFINITY;
                    }
                    if g >= hi_q {
                        dist = f32::INFINITY;
                    }
                }
                if dist < worst {
                    // Insert at the rank of `dist` (the number of kept entries not above it, by binary search), shifting the
                    // tail down.
                    let mut lo_i = 0u32;
                    let mut hi_i = lo_i + KM;
                    for _it in 0..9 {
                        if lo_i < hi_i {
                            let mid = (lo_i + hi_i) / 2;
                            if best_d[(base + mid) as usize] <= dist {
                                lo_i = mid + 1;
                            } else {
                                hi_i = mid;
                            }
                        }
                    }
                    let p = lo_i;
                    // Entries p..top move down one place (top = the last live entry; with a full list it falls off).
                    let mut top = cnt;
                    if top > KM - 1 {
                        top = KM - 1;
                    }
                    if cnt < KM {
                        cnt += 1;
                    }
                    for s in 0..top - p {
                        let i = top - s;
                        best_d[(base + i) as usize] = best_d[(base + i - 1) as usize];
                        best_v[(base + i) as usize] = best_v[(base + i - 1) as usize];
                    }
                    best_d[(base + p) as usize] = dist;
                    best_v[(base + p) as usize] = vals[j as usize];
                    worst = best_d[(base + KM - 1) as usize];
                    if thr_q < worst {
                        worst = thr_q;
                    }
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
        let mut store = KnnStore::with_precision(d, 300, false);
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
        let mut store = KnnStore::with_precision(d, 300, false);
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

    /// Enough keys for the threshold pass to engage (sample of n / 64 keys), with masks, against a CPU brute force.
    #[test]
    fn pending_keys_are_searched_without_one_tile_per_add() {
        // An online memory adds 32 keys at a time and searches after each: no finish(), so adds must not become tiles,
        // `len()` counts pending keys, and the pending tail is found (also under a mask that points into it).
        let (d, rows, step) = (16, 12, 32);
        let mut rng = Rng::new(5);
        let keys: Vec<f32> = (0..3200 * d).map(|_| rng.next_gaussian()).collect();
        let vals: Vec<u8> = (0..3200).map(|i| (i % 251) as u8).collect();
        let queries: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        let mut store = KnnStore::with_precision(d, 1024, false);
        for n in (step..=3200).step_by(step) {
            store.add(&keys[(n - step) * d..n * d], &vals[n - step..n]);
            assert_eq!(store.len(), n);
            if n < KMAX || n % 320 != 0 {
                continue;
            }
            let lo = (n - 100) as u32; // allowed: the newest 100 keys, which sit in the tail unless a tile just filled
            let ranges = vec![(lo, n as u32); rows];
            for (mask, label) in [(None, "plain"), (Some((&ranges[..], 0u32)), "masked")] {
                let got = store.search_masked(&queries, rows, mask);
                for r in 0..rows {
                    let q = &queries[r * d..(r + 1) * d];
                    let mut all: Vec<f32> = (if mask.is_some() { lo as usize } else { 0 }..n)
                        .map(|i| keys[i * d..(i + 1) * d].iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum())
                        .collect();
                    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    for j in 0..KMAX.min(all.len()) {
                        let g = got[r * KMAX + j].0;
                        assert!((g - all[j]).abs() < 1e-3 * (1.0 + all[j]), "{label} n {n} query {r} rank {j}: gpu {g} vs cpu {}", all[j]);
                    }
                }
            }
        }
        assert_eq!(store.tiles.len(), 3, "3200 keys in tiles of 1024 are 3 full tiles, the rest pending");
    }

    #[test]
    fn search_stays_exact_while_the_store_grows() {
        // Tiles of 64 make sample chunks of 64 sample keys (3904 keys): growth crosses chunk boundaries between searches.
        let (d, rows) = (16, 20);
        let mut rng = Rng::new(11);
        let keys: Vec<f32> = (0..30000 * d).map(|_| rng.next_gaussian()).collect();
        let vals: Vec<u8> = (0..30000).map(|i| (i % 251) as u8).collect();
        let queries: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        let mut store = KnnStore::with_precision(d, 64, false);
        let mut n = 0;
        for step in [3000, 4000, 1000, 9000, 13000] {
            store.add(&keys[n * d..(n + step) * d], &vals[n..n + step]);
            n += step;
            store.finish();
            let got = store.search(&queries, rows);
            for r in 0..rows {
                let q = &queries[r * d..(r + 1) * d];
                let mut all: Vec<f32> = (0..n).map(|i| keys[i * d..(i + 1) * d].iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum()).collect();
                all.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for j in 0..KMAX {
                    let g = got[r * KMAX + j].0;
                    assert!((g - all[j]).abs() < 1e-3 * (1.0 + all[j]), "n {n} query {r} rank {j}: gpu {g} vs cpu {}", all[j]);
                }
            }
        }
    }

    #[test]
    fn thresholded_search_matches_cpu_brute_force() {
        let (d, n, rows) = (16, 60000, 48);
        let mut rng = Rng::new(5);
        let keys: Vec<f32> = (0..n * d).map(|_| rng.next_gaussian()).collect();
        let vals: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let queries: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        let mut store = KnnStore::with_precision(d, 16384, false);
        store.add(&keys, &vals);
        store.finish();
        assert!(store.sample_keys.len() / d >= n / SAMPLE_STRIDE);
        // Always-visible prefix 20000, then a per-query range; some ranges are narrow (fewer than 16 sample hits).
        let ranges: Vec<(u32, u32)> = (0..rows).map(|r| (30000 + (r as u32 * 331) % 9000, 30000 + (r as u32 * 331) % 9000 + [100, 700, 5000, 25000][r % 4])).collect();
        for (mask, always) in [(None, 0u32), (Some(&ranges[..]), 20000u32)] {
            let got = store.search_masked(&queries, rows, mask.map(|m| (m, always)));
            for r in 0..rows {
                let q = &queries[r * d..(r + 1) * d];
                let mut all: Vec<f32> = (0..n)
                    .filter(|&i| mask.is_none() || (i as u32) < always || (i as u32 >= ranges[r].0 && (i as u32) < ranges[r].1))
                    .map(|i| keys[i * d..(i + 1) * d].iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum())
                    .collect();
                all.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for j in 0..KMAX {
                    let g = got[r * KMAX + j].0;
                    assert!((g - all[j]).abs() < 1e-3 * (1.0 + all[j]), "query {r} rank {j}: gpu {g} vs cpu {}", all[j]);
                }
            }
        }
    }

    /// The f16 matrix-core path equals a CPU brute force over the f16-rounded keys and queries (padding: 48 queries is not
    /// a multiple of 16 tiles of keys either, the last tile is short).
    #[test]
    fn f16_search_matches_cpu_brute_force_on_rounded_vectors() {
        if !client().features().matmul.cmma.contains(&f16_config()) {
            return;
        }
        let (d, n, rows) = (32, 50000, 50);
        let mut rng = Rng::new(9);
        let keys: Vec<f32> = (0..n * d).map(|_| rng.next_gaussian()).collect();
        let vals: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let queries: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        let round = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| f16::from_f32(x).to_f32()).collect() };
        let (kr, qr) = (round(&keys), round(&queries));
        let mut store = KnnStore::with_precision(d, 16384, true);
        store.add(&keys, &vals);
        store.finish();
        let got = store.search(&queries, rows);
        for r in 0..rows {
            let q = &qr[r * d..(r + 1) * d];
            let mut all: Vec<f32> = (0..n).map(|i| kr[i * d..(i + 1) * d].iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum()).collect();
            all.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for j in 0..KMAX {
                let g = got[r * KMAX + j].0;
                assert!((g - all[j]).abs() < 2e-3 * (1.0 + all[j]), "query {r} rank {j}: gpu {g} vs cpu {}", all[j]);
            }
        }
    }

    /// Spike for a fused matmul + threshold filter: a 16x16 accumulator tile stored to shared memory, read back per lane.
    /// out[0..256] = the tile as seen through shared memory; out[256] = number of entries below `thr`.
    #[cube(launch)]
    fn k_tile_to_shared(a: &[f16], b: &[f16], out: &mut [f32], thr: f32) {
        let c = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
        let at = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &a[0..a.len()], 16u32);
        let bt = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &b[0..b.len()], 16u32);
        cmma::execute(&at, &bt, &c, &c);
        let mut tile = Shared::<[f32]>::new_slice(256usize);
        cmma::store(&mut tile, &c, 16u32, cmma::MatrixLayout::RowMajor);
        sync_cube();
        let mut below = 0.0f32;
        for i in 0..256u32 {
            if i % CUBE_DIM == UNIT_POS {
                let v = tile[i as usize];
                out[i as usize] = v;
                if v < thr {
                    below += 1.0;
                }
            }
        }
        let total = plane_sum(below);
        if UNIT_POS == 0 {
            out[256usize] = total;
        }
    }

    #[test]
    fn matrix_core_tile_goes_through_shared_memory() {
        if !client().features().matmul.cmma.contains(&f16_config()) {
            return;
        }
        let mut rng = Rng::new(3);
        let a: Vec<f16> = (0..256).map(|_| f16::from_f32(rng.next_gaussian())).collect();
        let b: Vec<f16> = (0..256).map(|_| f16::from_f32(rng.next_gaussian())).collect();
        let (ah, bh) = (client().create_from_slice(f16::as_bytes(&a)), client().create_from_slice(f16::as_bytes(&b)));
        let out = upload_f32(&vec![-1.0f32; 257]);
        let plane = client().properties().hardware.plane_size_max;
        let thr = 0.0f32;
        k_tile_to_shared::launch(client(), CubeCount::Static(1, 1, 1), CubeDim::new_1d(plane), buf(&ah, 256), buf(&bh, 256), buf(&out, 257), thr);
        let got = read(&out);
        let mut below = 0;
        for i in 0..16 {
            for j in 0..16 {
                let want: f32 = (0..16).map(|k| a[i * 16 + k].to_f32() * b[k * 16 + j].to_f32()).sum();
                assert!((got[i * 16 + j] - want).abs() < 1e-3, "tile[{i},{j}]: {} vs {want}", got[i * 16 + j]);
                below += (want < thr) as usize;
            }
        }
        assert_eq!(got[256] as usize, below);
    }
}
