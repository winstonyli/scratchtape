//! Approximate search over the first keys of a `KnnStore`: an inverted-file (IVF) index. `build_ivf` clusters those keys
//! (k-means on the store's key sample, then one nearest-centroid pass over every key) and stores them cell by cell;
//! `search_ivf` sends each query only to the `nprobe` cells with the nearest centroids. Queries are handled in groups: all
//! the group's distances to the cells it probes come from one matmul per cell (the queries probing that cell, against its
//! keys) into a shared workspace, then one kernel launch gives every (query, part) thread its own top list over the query's
//! cells, nearest cell first (docs/tiers_design.md, approximate search).
use super::*;

/// Threads per query in the scan; each keeps its own top list (lists are merged on the host at the end).
const PARTS: usize = 8;
/// Most queries in one group.
const GROUP_MAX: usize = 8192;
/// Floats in the distance workspace shared by a group's matmuls.
const WS_FLOATS: usize = 256 << 20;
/// Every `SAMP`-th row of a query's cells is sampled to bound its `KMAX`-th neighbour: the `THRESH_RANK`-th nearest sampled
/// row is about the `THRESH_RANK * SAMP`-th nearest row, 4x the 256 needed, and keys beyond it are never inserted.
const SAMP: u32 = 32;
/// Lloyd iterations on the key sample.
const KMEANS_ITERS: usize = 10;
const ARGMIN_DIM: u32 = 256;
/// Cubes per row of a grouped matmul launch (a launch dimension is limited to 65535).
const GROUP_GX: u32 = 8192;

pub(super) struct Ivf {
    /// Centroids as a device key tile (f16, rounded as the keys are) and their squared norms.
    pub(super) keys: Handle,
    pub(super) norm: Vec<f32>,
    pub(super) nlist: usize,
    /// Clustered rows (the first `n` of the store), and their squared norms and values for the scan kernel.
    pub(super) n: usize,
    pub(super) norms: Handle,
    pub(super) vals: Handle,
    /// Each cell as pieces of device tiles.
    pub(super) cells: Vec<Vec<Piece>>,
}

/// Rows a..b of device tile `tile`, which are store rows g0..g0 + (b - a).
pub(super) struct Piece {
    pub(super) tile: usize,
    pub(super) a: usize,
    pub(super) b: usize,
    pub(super) g0: usize,
}

impl Piece {
    /// First row of the matrix-core computation (16-row tiles) and its row count.
    fn rows(&self) -> (usize, usize) {
        let base = self.a & !15;
        (base, self.b.next_multiple_of(16) - base)
    }
}

/// Centroids in the f16 matmul's second-operand layout ([d, nlist padded to 16]), with their squared norms.
struct Cent {
    qh: Handle,
    norm: Handle,
    norm_host: Vec<f32>,
    nlist: usize,
    qp: usize,
}

impl Cent {
    fn new(centroids: &[f32], d: usize) -> Cent {
        let r16: Vec<f16> = centroids.iter().map(|&x| f16::from_f32(x)).collect();
        let norm_host: Vec<f32> = r16.chunks_exact(d).map(|c| c.iter().map(|x| x.to_f32() * x.to_f32()).sum()).collect();
        let (qh, qp) = transposed_f16(&r16, d);
        Cent { qh, norm: upload_f32(&norm_host), norm_host, nlist: r16.len() / d, qp }
    }
}

/// out[j] = the index (as f32) of the centroid i minimising cnorm[i] - 2 dist[j, i], for j < len. One cube of `ARGMIN_DIM` threads per
/// row j: the threads read the row's columns together (coalesced; a thread per row strides `ld` floats between neighbours, 14x slower)
/// and reduce through shared memory; ties go to the lowest index.
#[cube(launch)]
fn k_argmin(dist: &[f32], cnorm: &[f32], out: &mut [f32], ld: u32, nc: u32) {
    let j = CUBE_POS_X;
    let t = UNIT_POS;
    let mut best = 1.0e30f32;
    let mut bi = 0u32;
    for s in 0..(nc + ARGMIN_DIM - 1) / ARGMIN_DIM {
        let i = s * ARGMIN_DIM + t;
        if i < nc {
            let sc = cnorm[i as usize] - 2.0 * dist[(j * ld + i) as usize];
            if sc < best {
                best = sc;
                bi = i;
            }
        }
    }
    let mut score = Shared::<[f32]>::new_slice(ARGMIN_DIM as usize);
    let mut index = Shared::<[u32]>::new_slice(ARGMIN_DIM as usize);
    score[t as usize] = best;
    index[t as usize] = bi;
    sync_cube();
    #[unroll]
    for r in 0..8u32 {
        let half = ARGMIN_DIM >> (r + 1);
        if t < half {
            let (o, m) = (score[(t + half) as usize], score[t as usize]);
            if o < m || (o == m && index[(t + half) as usize] < index[t as usize]) {
                score[t as usize] = o;
                index[t as usize] = index[(t + half) as usize];
            }
        }
        sync_cube();
    }
    if t == 0 {
        out[j as usize] = f32::cast_from(index[0usize]);
    }
}

/// Gathers query columns into the matmuls' second operands: for pair i, column pdst[i] (+ c * pstride[i] for component c) of `dst`
/// is query pq[i] of `src` (f16 [., d]).
#[cube(launch)]
fn k_gather_q(src: &[f16], dst: &mut [f16], pq: &[u32], pdst: &[u32], pstride: &[u32], n: u32, d: u32) {
    // One thread per (pair, component): neighbouring threads read neighbouring components of one query.
    let e = ABSOLUTE_POS as u32;
    if e < n * d {
        let i = e / d;
        let c = e % d;
        dst[(pdst[i as usize] + c * pstride[i as usize]) as usize] = src[(pq[i as usize] * d + c) as usize];
    }
}

/// Most cells one query probes.
const PROBE_MAX: u32 = 16;

/// Query i's `nprobe` nearest cells, nearest first, as f32 ids at out[i * nprobe..]: dist is the [nl16, qp] centroid dot products
/// (cell c, query i at c * qp + i), cnorm the squared centroid norms.
#[cube(launch)]
fn k_probe(dist: &[f32], cnorm: &[f32], out: &mut [f32], q: u32, qp: u32, nl: u32, nprobe: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < q {
        let mut bd = Array::<f32>::new(PROBE_MAX as usize);
        let mut bi = Array::<u32>::new(PROBE_MAX as usize);
        for k in 0..PROBE_MAX {
            bd[k as usize] = 3.0e38f32;
            bi[k as usize] = 0u32;
        }
        for c in 0..nl {
            let dd = cnorm[c as usize] - 2.0 * dist[(c * qp + i) as usize];
            if dd < bd[(nprobe - 1) as usize] {
                let mut p = nprobe - 1;
                for _s in 0..PROBE_MAX {
                    if p > 0 {
                        if bd[(p - 1) as usize] > dd {
                            bd[p as usize] = bd[(p - 1) as usize];
                            bi[p as usize] = bi[(p - 1) as usize];
                            p -= 1;
                        }
                    }
                }
                bd[p as usize] = dd;
                bi[p as usize] = c;
            }
        }
        for k in 0..nprobe {
            out[(i * nprobe + k) as usize] = f32::cast_from(bi[k as usize]);
        }
    }
}

/// Pass 1 of the scan: thread t = (part t / nq, query t % nq) keeps the `TR` nearest of every `SAMP`-th row of its part of
/// query gi's row ranges (laid out as in `k_scan`), ascending, at thr_d[t * TR].
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_sample(ws: &[f32], norms: &[f32], thr_d: &mut [f32], q_off: &[u32], v_base: &[u32], v_stride: &[u32], v_g0: &[u32], v_len: &[u32], nq: u32, parts: u32) {
    let t = ABSOLUTE_POS as u32;
    if t < nq * parts {
        let gi = t % nq;
        let p = t / nq;
        let base = t * TR;
        for v in q_off[gi as usize]..q_off[(gi + 1) as usize] {
            let len = v_len[v as usize];
            let chunk = (len + parts - 1) / parts;
            let start = p * chunk;
            let dbase = v_base[v as usize];
            let stride = v_stride[v as usize];
            let g0 = v_g0[v as usize];
            let ns = (chunk + SAMP - 1) / SAMP;
            for sj in 0..ns {
                let jj = start + sj * SAMP;
                if jj < len {
                    let dist = norms[(g0 + jj) as usize] - 2.0 * ws[(dbase + jj * stride) as usize];
                    if dist < thr_d[(base + TR - 1) as usize] {
                        let mut pos = 0u32;
                        for i in 0..TR {
                            if thr_d[(base + i) as usize] <= dist {
                                pos += 1;
                            }
                        }
                        for s in 0..TR - 1 - pos {
                            let i = TR - 1 - s;
                            thr_d[(base + i) as usize] = thr_d[(base + i - 1) as usize];
                        }
                        thr_d[(base + pos) as usize] = dist;
                    }
                }
            }
        }
    }
}

/// Thread t = (part t / nq, query t % nq) keeps its own top list (`KM` entries at t * KM) over its part of every row range of
/// query gi: the ranges q_off[gi]..q_off[gi + 1] of the v_* arrays, in order. Range v is v_len[v] store rows from v_g0[v], whose
/// dot products with the query are ws[v_base[v] + j * v_stride[v]]. Lists start empty (infinity); keys at or beyond a query's
/// threshold thr[gi] are skipped.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_scan(
    ws: &[f32],
    norms: &[f32],
    vals: &[f32],
    best_d: &mut [f32],
    best_v: &mut [f32],
    thr: &[f32],
    q_off: &[u32],
    v_base: &[u32],
    v_stride: &[u32],
    v_g0: &[u32],
    v_len: &[u32],
    nq: u32,
    parts: u32,
) {
    let t = ABSOLUTE_POS as u32;
    if t < nq * parts {
        let gi = t % nq;
        let p = t / nq;
        let base = t * KM;
        let thr_q = thr[gi as usize];
        let mut worst = best_d[(base + KM - 1) as usize];
        if thr_q < worst {
            worst = thr_q;
        }
        let mut cnt = 0u32;
        for v in q_off[gi as usize]..q_off[(gi + 1) as usize] {
            let len = v_len[v as usize];
            let chunk = (len + parts - 1) / parts;
            let start = p * chunk;
            let dbase = v_base[v as usize];
            let stride = v_stride[v as usize];
            let g0 = v_g0[v as usize];
            for jl in 0..chunk {
                let jj = start + jl;
                if jj < len {
                    let dist = norms[(g0 + jj) as usize] - 2.0 * ws[(dbase + jj * stride) as usize];
                    if dist < worst {
                        // As in `k_topk`: insert at the rank of `dist`, shifting the tail down.
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
                        let pos = lo_i;
                        let mut top = cnt;
                        if top > KM - 1 {
                            top = KM - 1;
                        }
                        if cnt < KM {
                            cnt += 1;
                        }
                        for s in 0..top - pos {
                            let i = top - s;
                            best_d[(base + i) as usize] = best_d[(base + i - 1) as usize];
                            best_v[(base + i) as usize] = best_v[(base + i - 1) as usize];
                        }
                        best_d[(base + pos) as usize] = dist;
                        best_v[(base + pos) as usize] = vals[(g0 + jj) as usize];
                        worst = best_d[(base + KM - 1) as usize];
                        if thr_q < worst {
                            worst = thr_q;
                        }
                    }
                }
            }
        }
    }
}

/// Grouped matmul over the cell pieces of one tile: piece p of `p0..p1` (descriptors `desc[4p..]`: first tile row of its keys, its
/// padded query count `qp`, element offset of its queries in `b`, element offset of its output in `out`) owns cubes
/// `cube_start[p]..cube_start[p + 1]`, one 16x16 tile of its [rows, qp] output each, as `k_dots_cmma`. The launch has `count` cubes
/// starting at `cube_start[p0]`. With `quant` the tile's keys are `bits`-bit integers in `q` (one scale per row in `scale`) and each
/// cube dequantises its 16x16 key block into shared memory, so the dequantised rows never go through memory; otherwise they are f16
/// in `keys`.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_dots_group(keys: &[f16], q: &[u32], scale: &[f32], b: &[f16], out: &mut [f32], desc: &[u32], cube_start: &[u32], #[comptime] k: u32, #[comptime] quant: bool, bits: u32, p0: u32, p1: u32, count: u32) {
    let mut sh = Shared::<[f16]>::new_slice(256usize);
    let lin = CUBE_POS_Y * GROUP_GX + CUBE_POS_X;
    if lin < count {
        let id = cube_start[p0 as usize] + lin;
        let mut lo = p0;
        let mut hi = p1;
        for _ in 0..24u32 {
            if hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if cube_start[mid as usize] <= id {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        }
        let a_row = desc[(lo * 4) as usize];
        let n = desc[(lo * 4 + 1) as usize];
        let b_off0 = desc[(lo * 4 + 2) as usize];
        let out_off = desc[(lo * 4 + 3) as usize];
        let loc = id - cube_start[lo as usize];
        let nct = n / 16;
        let rows16 = (cube_start[(lo + 1) as usize] - cube_start[lo as usize]) / nct * 16;
        let row0 = (loc / nct) * 16;
        let col0 = (loc % nct) * 16;
        let c = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
        let wpr = bits / 2;
        let per = 32 / bits;
        let dw = k * bits / 32;
        let mask = (1u32 << bits) - 1;
        let half = 1u32 << (bits - 1);
        for kk in 0..k / 16 {
            let b_off = (b_off0 + kk * 16 * n + col0) as usize;
            let qt = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &b[b_off..b.len()], n);
            if quant {
                for j in 0..4u32 {
                    let t = UNIT_POS_X + j * CUBE_DIM_X;
                    if t < 16 * wpr {
                        let r = t / wpr;
                        let wi = t % wpr;
                        let trow = a_row + row0 + r;
                        let w = q[(trow * dw + kk * wpr + wi) as usize];
                        let s = scale[trow as usize];
                        for e in 0..per {
                            let mut v = f32::cast_from((w >> (bits * e)) & mask);
                            if v >= f32::cast_from(half) {
                                v -= f32::cast_from(1u32 << bits);
                            }
                            sh[(r * 16 + wi * per + e) as usize] = f16::cast_from(v * s);
                        }
                    }
                }
                sync_cube();
                let kt = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &sh[0..256], 16);
                cmma::execute(&qt, &kt, &c, &c);
                sync_cube();
            } else {
                let a_off = ((a_row + row0) * k + kk * 16) as usize;
                let kt = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &keys[a_off..keys.len()], k);
                cmma::execute(&qt, &kt, &c, &c);
            }
        }
        let o_off = (out_off + col0 * rows16 + row0) as usize;
        let len = out.len();
        cmma::store(&mut out[o_off..len], &c, rows16, cmma::MatrixLayout::RowMajor);
    }
}

impl KnnStore {
    /// Nearest centroid of each of the `len` rows of the device key tile `keys` (`dist` is a [tile, c.qp] scratch buffer).
    fn nearest_centroid(&self, keys: &Handle, a_row0: usize, len: usize, c: &Cent, dist: &Handle) -> Vec<u32> {
        self.dots(keys, a_row0, len, &c.qh, 0, dist, self.tile * c.qp, 0, c.nlist, c.qp);
        let out = client().empty(len * 4);
        crate::gpu_step::count_launch();
        k_argmin::launch(client(), CubeCount::Static(len as u32, 1, 1), CubeDim::new_1d(ARGMIN_DIM), buf(dist, self.tile * c.qp), buf(&c.norm, c.nlist), buf(&out, len), c.qp as u32, c.nlist as u32);
        read(&out).iter().map(|&x| x as u32).collect()
    }

    /// Clusters all keys uploaded so far into `nlist` cells and reorders them cell by cell, so that `search_ivf` can scan whole
    /// cells. Call after the last train key is added and `finish`ed, before any in-document key: those are appended after the
    /// clustered rows and are searched exactly (`search_tail`). Exact searches stay valid (the order of the first keys
    /// changes, the set does not).
    pub fn build_ivf(&mut self, nlist: usize) {
        assert!(self.f16 && self.ivf.is_none() && self.pending_vals.is_empty(), "build_ivf: f16 store, once, after finish");
        let (d, n) = (self.d, self.uploaded());
        let ns = self.sample_keys.len() / d;
        assert!(nlist <= self.tile && ns >= 8 * nlist, "build_ivf: {nlist} cells from {ns} sample keys");
        let t0 = std::time::Instant::now();
        let dist = client().empty(self.tile * nlist.next_multiple_of(16) * 4);
        // Lloyd's algorithm on the sample, starting from evenly spaced sample keys.
        let mut cent: Vec<f32> = (0..nlist).flat_map(|c| self.sample_keys[(c * ns / nlist) * d..][..d].to_vec()).collect();
        for it in 0..KMEANS_ITERS {
            let c = Cent::new(&cent, d);
            let (mut sum, mut count) = (vec![0f32; nlist * d], vec![0usize; nlist]);
            for block in self.sample_keys.chunks(self.tile * d) {
                let (kh, _) = self.upload_keys(block);
                for (j, &a) in self.nearest_centroid(&kh, 0, block.len() / d, &c, &dist).iter().enumerate() {
                    count[a as usize] += 1;
                    sum[a as usize * d..][..d].iter_mut().zip(&block[j * d..(j + 1) * d]).for_each(|(s, x)| *s += x);
                }
            }
            for a in 0..nlist {
                if count[a] > 0 {
                    cent[a * d..(a + 1) * d].iter_mut().zip(&sum[a * d..(a + 1) * d]).for_each(|(c, s)| *c = s / count[a] as f32);
                } else {
                    // An empty cell restarts at another sample key.
                    let j = (a * 7919 + it * 104729) % ns;
                    cent[a * d..(a + 1) * d].copy_from_slice(&self.sample_keys[j * d..(j + 1) * d]);
                }
            }
        }
        // Number the cells along a nearest-neighbour chain, so that nearby ids are nearby cells and queries sorted by cell are
        // spatially coherent (a group of them probes few distinct cells).
        let mut chain = vec![0usize];
        let mut left: Vec<usize> = (1..nlist).collect();
        while !left.is_empty() {
            let last = &cent[chain[chain.len() - 1] * d..][..d];
            let dist2 = |a: usize| cent[a * d..][..d].iter().zip(last).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
            let at = (0..left.len()).min_by(|&i, &j| dist2(left[i]).partial_cmp(&dist2(left[j])).unwrap()).unwrap();
            chain.push(left.swap_remove(at));
        }
        let cent: Vec<f32> = chain.iter().flat_map(|&a| cent[a * d..][..d].to_vec()).collect();
        // Every key's cell, from the device tiles.
        let c = Cent::new(&cent, d);
        let mut cell: Vec<u32> = Vec::with_capacity(n);
        for t in &self.tiles {
            let (kh, r0) = self.rows_f16(t, 0, t.len.next_multiple_of(16));
            cell.extend(self.nearest_centroid(&kh, r0, t.len, &c, &dist));
        }
        let t_assign = t0.elapsed();
        let mut start = vec![0usize; nlist + 1];
        for &a in &cell {
            start[a as usize + 1] += 1;
        }
        for a in 0..nlist {
            start[a + 1] += start[a];
        }
        let mut cursor = start[..nlist].to_vec();
        let dest: Vec<usize> = cell
            .iter()
            .map(|&a| {
                cursor[a as usize] += 1;
                cursor[a as usize] - 1
            })
            .collect();
        // Reorder through the host, then replace the tiles.
        // (Quantised keys move as packed bytes with their scales; f16 keys as halves, re-uploaded as f32.)
        let kb = if self.kbits > 0 { self.key_bytes() } else { 2 * d };
        let (mut keys, mut scales, mut norms, mut vals) = (vec![0u8; n * kb], vec![0f32; if self.kbits > 0 { n } else { 0 }], vec![0f32; if self.kbits > 0 { n } else { 0 }], vec![0u16; n]);
        let mut i = 0;
        for t in &self.tiles {
            let bytes = client().read_one(t.keys.clone()).unwrap();
            let v = read(&t.vals);
            let sc = t.scales.as_ref().map(read);
            let nm = if sc.is_some() { read(&t.norms) } else { vec![] };
            for j in 0..t.len {
                let to = dest[i + j];
                keys[to * kb..(to + 1) * kb].copy_from_slice(&bytes[j * kb..(j + 1) * kb]);
                if let Some(sc) = &sc {
                    scales[to] = sc[j];
                    norms[to] = nm[j];
                }
                vals[to] = v[j] as u16;
            }
            i += t.len;
        }
        self.tiles.clear();
        for (ci, (kb, vb)) in keys.chunks(self.tile * kb).zip(vals.chunks(self.tile)).enumerate() {
            let tile = if self.kbits > 0 {
                self.tile_of_quant(kb, &scales[ci * self.tile..][..vb.len()], &norms[ci * self.tile..][..vb.len()], vb)
            } else {
                let kf: Vec<f32> = f16::from_bytes(kb).iter().map(|x| x.to_f32()).collect();
                self.tile_of(&kf, vb)
            };
            self.tiles.push(tile);
        }
        let cent_rounded: Vec<f32> = cent.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        let (cent_keys, _) = self.upload_keys(&cent_rounded);
        let sizes = (0..nlist).map(|a| start[a + 1] - start[a]);
        eprintln!(
            "ivf: {nlist} cells over {n} keys, sizes min {} / mean {} / max {}; k-means + assignment {:.1}s, reorder {:.1}s",
            sizes.clone().min().unwrap(),
            n / nlist,
            sizes.max().unwrap(),
            t_assign.as_secs_f64(),
            (t0.elapsed() - t_assign).as_secs_f64()
        );
        let norms: Vec<f32> = self.tiles.iter().flat_map(|t| read(&t.norms)).collect();
        let cells = (0..nlist).map(|a| self.pieces(start[a], start[a + 1])).collect();
        let vals_f: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
        self.ivf = Some(Ivf { keys: cent_keys, norm: c.norm_host, nlist, n, norms: upload_f32(&norms), vals: upload_f32(&vals_f), cells });
    }

    /// Store rows a..b as pieces of device tiles.
    fn pieces(&self, a: usize, b: usize) -> Vec<Piece> {
        let (mut off, mut out) = (0, vec![]);
        for (tile, t) in self.tiles.iter().enumerate() {
            let (lo, hi) = (a.max(off), b.min(off + t.len));
            if lo < hi {
                out.push(Piece { tile, a: lo - off, b: hi - off, g0: lo });
            }
            off += t.len;
        }
        out
    }

    /// `search`, approximately and over the clustered keys only: for each query, the `KMAX` nearest among the keys of its
    /// `nprobe` nearest cells. Same output as `search_tagged`. Groups of queries share a workspace of `WS_FLOATS` distances, so a
    /// call wants thousands of queries: a cell probed by few queries wastes a matmul launch.
    pub fn search_ivf(&self, queries: &[f32], rows: usize, nprobe: usize) -> Vec<(f32, u16)> {
        let ivf = self.ivf.as_ref().expect("search_ivf needs build_ivf");
        let (d, nl) = (self.d, ivf.nlist);
        assert!(queries.len() == rows * d && nprobe >= 1 && nprobe <= nl && nprobe <= PROBE_MAX as usize);
        let rq: Vec<f16> = queries.iter().map(|&x| f16::from_f32(x)).collect();
        let rq_dev = client().create_from_slice(f16::as_bytes(&rq));
        // Each query's nprobe nearest cells, nearest first.
        let t0 = std::time::Instant::now();
        let nl16 = nl.next_multiple_of(16);
        let mut probes = vec![0u32; rows * nprobe];
        let cnorm = upload_f32(&ivf.norm);
        for q0 in (0..rows).step_by(QBLOCK) {
            let q = QBLOCK.min(rows - q0);
            let (qh, qp) = transposed_f16(&rq[q0 * d..(q0 + q) * d], d);
            let dist = self.scratch(S_CENT, nl16 * qp * 4);
            self.dots(&ivf.keys, 0, nl, &qh, 0, &dist, nl16 * qp, 0, q, qp);
            let out = client().empty(q * nprobe * 4);
            crate::gpu_step::count_launch();
            #[rustfmt::skip]
            k_probe::launch(client(), cubes(q), CubeDim::new_1d(EW_DIM), buf(&dist, nl16 * qp), buf(&cnorm, nl), buf(&out, q * nprobe), q as u32, qp as u32, nl as u32, nprobe as u32);
            for (o, &c) in probes[q0 * nprobe..(q0 + q) * nprobe].iter_mut().zip(&read(&out)) {
                *o = c as u32;
            }
        }
        let t_probe = t0.elapsed();
        // Queries per group so that a group's matmul outputs fit the workspace (cells average n / nl rows).
        let per_query = nprobe * (ivf.n / nl + 16) * 5 / 4;
        let group = (WS_FLOATS / per_query).clamp(16, GROUP_MAX);
        let ws = self.scratch(S_WS, WS_FLOATS * 4);
        let mut stage = [std::time::Duration::ZERO; 3];
        // Queries sorted by their nearest cell, so that a group probes few distinct cells (each matmul is then wide).
        let mut order: Vec<u32> = (0..rows as u32).collect();
        order.sort_by_key(|&q| probes[q as usize * nprobe]);
        let sorted_probes: Vec<u32> = order.iter().flat_map(|&q| probes[q as usize * nprobe..][..nprobe].iter().copied()).collect();
        let mut sorted = Vec::with_capacity(rows * KMAX);
        for g0 in (0..rows).step_by(group) {
            let g1 = (g0 + group).min(rows);
            self.scan_group(ivf, &rq, &rq_dev, &order[g0..g1], &sorted_probes[g0 * nprobe..g1 * nprobe], nprobe, &ws, true, &mut stage, &mut sorted);
        }
        let mut out = vec![(f32::INFINITY, 0u16); rows * KMAX];
        for (pos, &q) in order.iter().enumerate() {
            out[q as usize * KMAX..(q as usize + 1) * KMAX].copy_from_slice(&sorted[pos * KMAX..(pos + 1) * KMAX]);
        }
        if std::env::var_os("KNN_WALL").is_some() {
            eprintln!("WALL ivf {rows} queries {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        if std::env::var_os("KNN_TIMING").is_some() {
            eprintln!(
                "ivf: {rows} queries x {nprobe} of {nl} cells in groups of {group}: {:.0} ms (probes {:.0}, matmuls {:.0}, scan launch + readback {:.0}, host merge {:.0})",
                t0.elapsed().as_secs_f64() * 1e3,
                t_probe.as_secs_f64() * 1e3,
                stage[0].as_secs_f64() * 1e3,
                stage[1].as_secs_f64() * 1e3,
                stage[2].as_secs_f64() * 1e3
            );
        }
        out
    }

    /// One group, queries `qs` (indices into `rq`) with their cells `probes` (nprobe each, nearest first); appends their lists
    /// to `out`, in the order of `qs`.
    #[allow(clippy::too_many_arguments)]
    fn scan_group(&self, ivf: &Ivf, rq: &[f16], rq_dev: &Handle, qs: &[u32], probes: &[u32], nprobe: usize, ws: &Handle, thresholds: bool, stage: &mut [std::time::Duration; 3], out: &mut Vec<(f32, u16)>) {
        let (d, nq) = (self.d, qs.len());
        // The queries probing each cell, in order, and each (query, probe)'s column among them.
        let mut cell_q: Vec<Vec<u32>> = vec![vec![]; ivf.nlist];
        let mut col = vec![0usize; nq * nprobe];
        for gi in 0..nq {
            for k in 0..nprobe {
                let c = probes[gi * nprobe + k] as usize;
                col[gi * nprobe + k] = cell_q[c].len();
                cell_q[c].push(gi as u32);
            }
        }
        // Workspace regions: one per (cell, piece), [rows, qp] floats.
        let mut region: Vec<Vec<usize>> = vec![vec![]; ivf.nlist];
        let mut used = 0usize;
        for (c, cq) in cell_q.iter().enumerate().filter(|(_, cq)| !cq.is_empty()) {
            let qp = cq.len().next_multiple_of(16);
            for piece in &ivf.cells[c] {
                region[c].push(used);
                used += piece.rows().1 * qp;
            }
        }
        if used > WS_FLOATS {
            assert!(nq > 1, "one query's cells need {used} workspace floats");
            let mid = nq / 2;
            self.scan_group(ivf, rq, rq_dev, &qs[..mid], &probes[..mid * nprobe], nprobe, ws, thresholds, stage, out);
            self.scan_group(ivf, rq, rq_dev, &qs[mid..], &probes[mid * nprobe..], nprobe, ws, thresholds, stage, out);
            return;
        }
        let timing = std::env::var_os("KNN_TIMING").is_some();
        let t0 = std::time::Instant::now();
        // The queries of each active cell as one [d, qp] operand each, gathered on the device into one buffer.
        let (mut q_at, mut cell_at, mut total) = (vec![0usize; ivf.nlist], vec![], 0usize);
        let (mut pq, mut pdst, mut pstride) = (vec![], vec![], vec![]);
        for (c, cq) in cell_q.iter().enumerate().filter(|(_, cq)| !cq.is_empty()) {
            let qp = cq.len().next_multiple_of(16);
            q_at[c] = total;
            cell_at.push(c);
            for (j, &gi) in cq.iter().enumerate() {
                pq.push(qs[gi as usize]);
                pdst.push((total + j) as u32);
                pstride.push(qp as u32);
            }
            total += d * qp;
        }
        let t_a = t0.elapsed();
        let qall = self.filled(S_QALL, total / 2, 0.0); // zero f16 pairs (total is even: d * qp)
        let up = |v: &[u32]| client().create_from_slice(u32::as_bytes(v));
        let np = pq.len();
        let (pq, pdst, pstride) = (up(&pq), up(&pdst), up(&pstride));
        assert!(np * d <= 65535 * EW_DIM as usize, "gather of {np} pairs exceeds one dispatch");
        crate::gpu_step::count_launch();
        #[rustfmt::skip]
        k_gather_q::launch(client(), cubes(np * d), CubeDim::new_1d(EW_DIM), buf(rq_dev, rq.len()), buf(&qall, total), buf(&pq, np), buf(&pdst, np), buf(&pstride, np), np as u32, d as u32);
        let t_b = t0.elapsed();
        if timing {
            read(&pstride);
        }
        let t_c = t0.elapsed();
        // Every piece's matmul, one grouped launch per tile (pieces are sorted by tile; a piece never spans two).
        let mut items: Vec<(usize, usize, usize, usize, usize, usize)> = vec![];
        for &c in &cell_at {
            let qp = cell_q[c].len().next_multiple_of(16);
            for (piece, &off) in ivf.cells[c].iter().zip(&region[c]) {
                let (base, rows16) = piece.rows();
                items.push((piece.tile, base, rows16, q_at[c], off, qp));
            }
        }
        items.sort_unstable_by_key(|it| (it.0, it.1));
        let (mut desc, mut cube_start) = (vec![], vec![0u32]);
        for it in &items {
            desc.extend([it.1 as u32, it.5 as u32, it.3 as u32, it.4 as u32]);
            cube_start.push(cube_start.last().unwrap() + (it.2 / 16 * (it.5 / 16)) as u32);
        }
        let (desc_dev, cube_start_dev) = (up(&desc), up(&cube_start));
        let t_d = t0.elapsed();
        let dummy = client().create_from_slice(&[0u8; 4]);
        let (plane, quant) = (client().properties().hardware.plane_size_max, self.kbits > 0);
        let mut p0 = 0;
        while p0 < items.len() {
            let tile = items[p0].0;
            let p1 = p0 + items[p0..].iter().take_while(|it| it.0 == tile).count();
            let t = &self.tiles[tile];
            let count = (cube_start[p1] - cube_start[p0]) as usize;
            let padded = t.len.next_multiple_of(16);
            let (keys, klen) = if quant { (&dummy, 1) } else { (&t.keys, padded * d) };
            let (qk, qlen) = if quant { (&t.keys, padded * self.key_bytes() / 4) } else { (&dummy, 1) };
            let (scales, slen) = match &t.scales {
                Some(s) => (s, padded),
                None => (&dummy, 1),
            };
            crate::gpu_step::count_launch();
            #[rustfmt::skip]
            k_dots_group::launch(client(), CubeCount::Static(GROUP_GX.min(count as u32), (count as u32).div_ceil(GROUP_GX), 1), CubeDim::new_1d(plane), buf(keys, klen), buf(qk, qlen), buf(scales, slen), buf(&qall, total), buf(ws, WS_FLOATS), buf(&desc_dev, items.len() * 4), buf(&cube_start_dev, items.len() + 1), d as u32, quant, self.kbits.max(4), p0 as u32, p1 as u32, count as u32);
            p0 = p1;
        }
        if timing {
            read(&cube_start_dev);
            eprintln!("  mm: gather queries {:.0} ms, qall+uploads {:.0}, gather kernel {:.0}, items {:.0}, launches+kernels {:.0}", t_a.as_secs_f64() * 1e3, (t_b - t_a).as_secs_f64() * 1e3, (t_c - t_b).as_secs_f64() * 1e3, (t_d - t_c).as_secs_f64() * 1e3, (t0.elapsed() - t_d).as_secs_f64() * 1e3);
            stage[0] += t0.elapsed();
        }
        // Per query, its pieces in probe order: workspace position of its column, row stride, first store row, rows.
        let t0 = std::time::Instant::now();
        let (mut q_off, mut v_base, mut v_stride, mut v_g0, mut v_len) = (vec![0u32], vec![], vec![], vec![], vec![]);
        for gi in 0..nq {
            for k in 0..nprobe {
                let c = probes[gi * nprobe + k] as usize;
                for (piece, &off) in ivf.cells[c].iter().zip(&region[c]) {
                    v_base.push((off + col[gi * nprobe + k] * piece.rows().1 + (piece.a - piece.rows().0)) as u32);
                    v_stride.push(1);
                    v_g0.push(piece.g0 as u32);
                    v_len.push((piece.b - piece.a) as u32);
                }
            }
            q_off.push(v_len.len() as u32);
        }
        let nv = v_len.len();
        let (q_off, v_base, v_stride, v_g0, v_len) = (up(&q_off), up(&v_base), up(&v_stride), up(&v_g0), up(&v_len));
        // Pass 1: each query's threshold from a sample of its rows.
        let thr_q: Vec<f32> = if thresholds {
            let thr_d = self.filled(S_THR, PARTS * nq * THRESH_RANK, f32::INFINITY);
            crate::gpu_step::count_launch();
            #[rustfmt::skip]
            k_sample::launch(client(), cubes(PARTS * nq), CubeDim::new_1d(EW_DIM), buf(ws, WS_FLOATS), buf(&ivf.norms, ivf.n), buf(&thr_d, PARTS * nq * THRESH_RANK), buf(&q_off, nq + 1), buf(&v_base, nv), buf(&v_stride, nv), buf(&v_g0, nv), buf(&v_len, nv), nq as u32, PARTS as u32);
            let thr = read(&thr_d);
            (0..nq).map(|gi| widen(rank_of_parts(&thr, gi, nq, PARTS, THRESH_RANK))).collect()
        } else {
            vec![f32::INFINITY; nq]
        };
        let t_s = t0.elapsed();
        let thr_buf = upload_f32(&thr_q);
        let (best_d, best_v) = (self.filled(S_BEST_D, PARTS * nq * KMAX, f32::INFINITY), self.filled(S_BEST_V, PARTS * nq * KMAX, 0.0));
        crate::gpu_step::count_launch();
        #[rustfmt::skip]
        k_scan::launch(client(), cubes(PARTS * nq), CubeDim::new_1d(EW_DIM), buf(ws, WS_FLOATS), buf(&ivf.norms, ivf.n), buf(&ivf.vals, ivf.n), buf(&best_d, PARTS * nq * KMAX), buf(&best_v, PARTS * nq * KMAX), buf(&thr_buf, nq), buf(&q_off, nq + 1), buf(&v_base, nv), buf(&v_stride, nv), buf(&v_g0, nv), buf(&v_len, nv), nq as u32, PARTS as u32);
        let t_l = t0.elapsed();
        read(&thr_buf);
        let t_k = t0.elapsed();
        let (bd, bv) = merged_lists(&best_d, &best_v, nq, PARTS);
        if timing {
            eprintln!("  group: sample pass {:.0} ms, list upload + scan launch {:.0}, scan kernel {:.0}, readback {:.0}", t_s.as_secs_f64() * 1e3, (t_l - t_s).as_secs_f64() * 1e3, (t_k - t_l).as_secs_f64() * 1e3, (t0.elapsed() - t_k).as_secs_f64() * 1e3);
        }
        stage[1] += t0.elapsed();
        let t0 = std::time::Instant::now();
        let first = out.len();
        for gi in 0..nq {
            let r = qs[gi] as usize;
            let qn: f32 = rq[r * d..(r + 1) * d].iter().map(|x| x.to_f32() * x.to_f32()).sum();
            let c: Vec<(f32, f32)> = (0..KMAX).map(|j| (bd[gi * KMAX + j], bv[gi * KMAX + j])).collect();
            if thr_q[gi].is_finite() && c[KMAX - 1].0.is_infinite() {
                // A threshold left fewer than KMAX keys: this group again without thresholds.
                out.truncate(first);
                stage[2] += t0.elapsed();
                return self.scan_group(ivf, rq, rq_dev, qs, probes, nprobe, ws, false, stage, out);
            }
            out.extend(c.iter().map(|&(dd, v)| ((dd + qn).max(0.0), v as u16)));
        }
        stage[2] += t0.elapsed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Rng;

    /// Throwaway timing: `KNN_TIMING=1 cargo test --release ivf_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn ivf_timing() {
        let (d, n, rows, nlist) = (256, 2_000_000, 16384, 316);
        let mut rng = Rng::new(5);
        let mut store = KnnStore::new(d, 16384);
        for _ in 0..n / 100_000 {
            let keys: Vec<f32> = (0..100_000 * d).map(|_| rng.next_gaussian()).collect();
            store.add(&keys, &vec![1u8; 100_000]);
        }
        store.finish();
        store.build_ivf(nlist);
        let queries: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        for nprobe in [4, 16] {
            let t = std::time::Instant::now();
            store.search_ivf(&queries, rows, nprobe);
            eprintln!("BENCH nprobe {nprobe}: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        }
    }

    /// Clustered keys: probing every cell must equal the exact search, and fewer cells must still find near neighbours.
    #[test]
    fn probing_all_cells_is_exact_and_fewer_cells_mostly_find_the_neighbours() {
        let (d, n, rows, nlist) = (32, 24_000, 300, 8);
        let mut rng = Rng::new(11);
        let centers: Vec<f32> = (0..nlist * d).map(|_| 4.0 * rng.next_gaussian()).collect();
        let keys: Vec<f32> = (0..n * d).map(|i| centers[((i / d) % nlist) * d + i % d] + rng.next_gaussian()).collect();
        let vals: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let queries: Vec<f32> = (0..rows * d).map(|i| centers[((i / d) % nlist) * d + i % d] + rng.next_gaussian()).collect();
        let mut store = KnnStore::with_precision(d, 4096, true);
        store.add(&keys, &vals);
        store.finish();
        // The key sample (every 61st key) is the k-means data: 393 keys for 8 cells.
        store.build_ivf(nlist);
        let exact = store.search(&queries, rows);
        let all = store.search_ivf(&queries, rows, nlist);
        for r in 0..rows {
            for j in 0..KMAX {
                let (a, b) = (exact[r * KMAX + j].0, all[r * KMAX + j].0);
                assert!((a - b).abs() < 2e-3 * (1.0 + a), "query {r} rank {j}: exact {a} vs all-cell ivf {b}");
            }
        }
        let some = store.search_ivf(&queries, rows, 2);
        let hits: usize = (0..rows).map(|r| (0..KMAX).filter(|&j| some[r * KMAX + j].0 <= exact[r * KMAX + KMAX - 1].0 * (1.0 + 2e-3)).count()).sum();
        let recall = hits as f64 / (rows * KMAX) as f64;
        assert!(recall > 0.8, "recall {recall} with 2 of {nlist} cells");
    }

    /// Quantised keys (8 and 4 bits): the same search to within the rounding, and probing every cell equals the exact quantised search.
    #[test]
    fn quantised_keys_search_like_f16_keys() {
        for (bits, tol) in [(8u32, 0.05f32), (4, 0.4)] {
            quantised_case(bits, tol);
        }
    }

    fn quantised_case(bits: u32, tol: f32) {
        let (d, n, rows, nlist) = (32, 12_000, 100, 4);
        let mut rng = Rng::new(5);
        let keys: Vec<f32> = (0..n * d).map(|_| rng.next_gaussian()).collect();
        let vals: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let queries: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        let mut f = KnnStore::with_precision(d, 4096, true);
        f.add(&keys, &vals);
        f.finish();
        let mut q = KnnStore::with_precision(d, 4096, true);
        q.set_key_bits(bits);
        q.add(&keys, &vals);
        q.finish();
        let (a, b) = (f.search(&queries, rows), q.search(&queries, rows));
        for r in 0..rows {
            for j in [0, 10, KMAX - 1] {
                let (x, y) = (a[r * KMAX + j].0, b[r * KMAX + j].0);
                assert!((x - y).abs() < tol * (1.0 + x), "query {r} rank {j}: f16 {x} vs {bits}-bit {y}");
            }
        }
        q.build_ivf(nlist);
        let all = q.search_ivf(&queries, rows, nlist);
        let c = q.search(&queries, rows);
        for r in 0..rows {
            for j in 0..KMAX {
                let (x, y) = (c[r * KMAX + j].0, all[r * KMAX + j].0);
                assert!((x - y).abs() < 2e-3 * (1.0 + x), "query {r} rank {j}: {bits}-bit exact {x} vs all-cell ivf {y}");
            }
        }
    }

    /// A saved and reloaded store (4-bit keys, IVF built) answers exactly as the original, and a different key string misses.
    #[test]
    fn saved_store_reloads_identically() {
        let (d, n, rows, nlist) = (32, 20_000, 64, 4);
        let mut rng = Rng::new(9);
        let keys: Vec<f32> = (0..n * d).map(|_| rng.next_gaussian()).collect();
        let vals: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let queries: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        let mut a = KnnStore::with_precision(d, 4096, true);
        a.set_key_bits(4);
        a.add(&keys, &vals);
        a.finish();
        a.build_ivf(nlist);
        let path = std::env::temp_dir().join(format!("knn_cache_test_{}.mem", std::process::id()));
        a.save(&path, "key one", 7).unwrap();
        assert!(KnnStore::load(&path, "key two").unwrap().is_none());
        let (b, extra) = KnnStore::load(&path, "key one").unwrap().unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(extra, 7);
        assert_eq!(a.search(&queries, rows), b.search(&queries, rows));
        assert_eq!(a.search_ivf(&queries, rows, 2), b.search_ivf(&queries, rows, 2));
    }

    /// The device's nearest centroid of each key equals a CPU argmin over the same f16-rounded values (cell counts that are not a
    /// multiple of the reduction width, and duplicate centroids, whose tie must go to the lowest index).
    #[test]
    fn nearest_centroid_matches_cpu_argmin() {
        let (d, rows, nlist) = (32usize, 300usize, 700usize);
        let mut rng = Rng::new(9);
        let keys: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian()).collect();
        let mut cent: Vec<f32> = (0..nlist * d).map(|_| rng.next_gaussian()).collect();
        let dup = cent[3 * d..4 * d].to_vec();
        cent[650 * d..651 * d].copy_from_slice(&dup);
        let store = KnnStore::new(d, 1024);
        let (kh, _) = store.upload_keys(&keys);
        let c = Cent::new(&cent, d);
        let dist = client().empty(1024 * c.qp * 4);
        let got = store.nearest_centroid(&kh, 0, rows, &c, &dist);
        let r = |x: f32| f16::from_f32(x).to_f32();
        for (j, &g) in got.iter().enumerate() {
            let score = |i: usize| c.norm_host[i] - 2.0 * (0..d).map(|k| r(keys[j * d + k]) * r(cent[i * d + k])).sum::<f32>();
            let best = (0..nlist).map(score).fold(f32::MAX, f32::min);
            assert!(score(g as usize) <= best + 1e-3 * (1.0 + best.abs()), "row {j}: picked {g} with {} vs best {best}", score(g as usize));
        }
    }
}
