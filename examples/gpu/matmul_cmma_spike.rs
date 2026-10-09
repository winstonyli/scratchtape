// SPIKE (throwaway measurement): can an f16 matrix-core (cmma) matmul beat the f32 training matmul
// (src/gpu_step/matmul.rs) on the shapes that dominate the training step?
//
// The cmma kernel reads f32 operands, converts to f16 on the way into shared memory (K staged 32 at a time), and each
// plane holds FM x FN 16x16 f32 accumulator fragments. NN / NT / TN by comptime flags (a transposed operand is staged
// in its stored order and loaded with ColMajor fragments). Weight-gradient shapes (accumulate) use split-k plus a sum
// launch, as the f32 path does. No epilogue (bias / ReLU / residual) on either side: raw products only.
// Timing: ROUNDS rounds of REPS back-to-back launches ended by one 4-byte readback; GPU-queue ms per launch.
// Shared GPU lease. Run: cargo run --release --example matmul_cmma_spike
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::matmul::{Epilogue, MatRef, matmul};
use scratchtape::gpu_step::{client, read, upload_f32};
use scratchtape::nn::Rng;
use std::time::{Duration, Instant};

const ROUNDS: usize = 21;
const REPS: usize = 20;
const BK: u32 = 32;

#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn k_cmma(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: u32,
    n: u32,
    lda: u32,
    ldb: u32,
    slice_stages: u32,
    stages: u32,
    #[comptime] ta: bool,
    #[comptime] tb: bool,
    #[comptime] bm: u32,
    #[comptime] bn: u32,
    #[comptime] pm: u32,
    #[comptime] pn: u32,
    #[comptime] pad: u32,
    #[comptime] threads: u32,
) {
    let fm = comptime![bm / (pm * 16)];
    let fnn = comptime![bn / (pn * 16)];
    let a_ld = comptime![if ta { bm + pad } else { 32 + pad }];
    let b_ld = comptime![if tb { 32 + pad } else { bn + pad }];
    let a_len = comptime![(if ta { 32 * (bm + pad) } else { bm * (32 + pad) }) as usize];
    let b_len = comptime![(if tb { bn * (32 + pad) } else { 32 * (bn + pad) }) as usize];
    let a_iters = comptime![bm * 32 / threads];
    let b_iters = comptime![bn * 32 / threads];
    let mut a_s = Shared::<[f16]>::new_slice(a_len);
    let mut b_s = Shared::<[f16]>::new_slice(b_len);

    let tid = UNIT_POS;
    let pid = UNIT_POS_Y;
    let wm0 = (pid / pn) * fm * 16;
    let wn0 = (pid % pn) * fnn * 16;
    let row0 = CUBE_POS_Y * bm;
    let col0 = CUBE_POS_X * bn;
    let z = CUBE_POS_Z;
    let s0 = z * slice_stages;
    let s1 = u32::min(s0 + slice_stages, stages);

    let fm_n = comptime![(bm / (pm * 16)) as usize];
    let fn_n = comptime![(bn / (pn * 16)) as usize];
    let la = comptime![if ta { cmma::MatrixLayout::ColMajor } else { cmma::MatrixLayout::RowMajor }];
    let lb = comptime![if tb { cmma::MatrixLayout::ColMajor } else { cmma::MatrixLayout::RowMajor }];
    let mut acc = Sequence::<cmma::Matrix<f32>>::new();
    #[unroll]
    for _i in 0..fm_n * fn_n {
        acc.push(cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0));
    }

    for s in s0..s1 {
        let k0 = s * 32;
        #[unroll]
        for i in 0..a_iters {
            let e = tid + threads * i;
            if ta {
                let c = e / bm;
                let r = e % bm;
                a_s[(c * a_ld + r) as usize] = f16::cast_from(a[((k0 + c) * lda + row0 + r) as usize]);
            } else {
                let r = e / 32;
                let c = e % 32;
                a_s[(r * a_ld + c) as usize] = f16::cast_from(a[((row0 + r) * lda + k0 + c) as usize]);
            }
        }
        #[unroll]
        for i in 0..b_iters {
            let e = tid + threads * i;
            if tb {
                let c = e / 32;
                let r = e % 32;
                b_s[(c * b_ld + r) as usize] = f16::cast_from(b[((col0 + c) * ldb + k0 + r) as usize]);
            } else {
                let r = e / bn;
                let c = e % bn;
                b_s[(r * b_ld + c) as usize] = f16::cast_from(b[((k0 + r) * ldb + col0 + c) as usize]);
            }
        }
        sync_cube();
        #[unroll]
        for kf in 0..2u32 {
            let mut bf = Sequence::<cmma::Matrix<f16>>::new();
            #[unroll]
            for j in 0..fn_n {
                let mut off = (kf * 16 * b_ld + wn0 + j as u32 * 16) as usize;
                if tb {
                    off = ((wn0 + j as u32 * 16) * b_ld + kf * 16) as usize;
                }
                bf.push(cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, lb, &b_s[off..b_len], b_ld));
            }
            #[unroll]
            for i in 0..fm_n {
                let mut off = ((wm0 + i as u32 * 16) * a_ld + kf * 16) as usize;
                if ta {
                    off = (kf * 16 * a_ld + wm0 + i as u32 * 16) as usize;
                }
                let af = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, la, &a_s[off..a_len], a_ld);
                #[unroll]
                for j in 0..fn_n {
                    cmma::execute(&af, bf.index(j), acc.index(i * fn_n + j), acc.index(i * fn_n + j));
                }
            }
        }
        sync_cube();
    }

    let len = out.len();
    #[unroll]
    for i in 0..fm_n {
        #[unroll]
        for j in 0..fn_n {
            let o = (z * m * n + (row0 + wm0 + i as u32 * 16) * n + col0 + wn0 + j as u32 * 16) as usize;
            cmma::store(&mut out[o..len], acc.index(i * fn_n + j), n, cmma::MatrixLayout::RowMajor);
        }
    }
}

/// out[i] (+)= sum over s of partial[s * len + i], s ascending.
#[cube(launch)]
fn k_sum(partial: &[f32], out: &mut [f32], len: u32, splits: u32, #[comptime] accumulate: bool) {
    let e = ABSOLUTE_POS as u32;
    if e < len {
        let mut v = 0.0f32;
        for s in 0..splits {
            v += partial[(s * len + e) as usize];
        }
        if accumulate {
            v += out[e as usize];
        }
        out[e as usize] = v;
    }
}

fn arg(h: &Handle, len: usize) -> BufferArg {
    // Safety: every handle here holds at least `len` f32s.
    unsafe { BufferArg::from_raw_parts(h.clone(), len) }
}

#[derive(Clone, Copy)]
struct Cfg {
    bm: u32,
    bn: u32,
    pm: u32,
    pn: u32,
    pad: u32,
}

impl Cfg {
    fn name(&self, splits: usize) -> String {
        format!("{}x{} p{}x{} pad{} s{}", self.bm, self.bn, self.pm, self.pn, self.pad, splits)
    }
}

struct Shape {
    m: usize,
    k: usize,
    n: usize,
    ta: bool,
    tb: bool,
    acc: bool,
}

/// One cmma product (plus the sum launch when split or accumulating) into `out` (scratch holds the partials).
fn cmma_run(sh: &Shape, cfg: Cfg, splits: usize, plane: u32, a: &Handle, b: &Handle, out: &Handle, scratch: &Handle) {
    let (m, k, n) = (sh.m, sh.k, sh.n);
    let stages = k / 32;
    let slice = stages.div_ceil(splits);
    let splits = stages.div_ceil(slice);
    let threads = plane * cfg.pm * cfg.pn;
    let lda = if sh.ta { m } else { k } as u32;
    let ldb = if sh.tb { k } else { n } as u32;
    let direct = splits == 1 && !sh.acc;
    let dst = if direct { out } else { scratch };
    k_cmma::launch(
        client(),
        CubeCount::Static(n as u32 / cfg.bn, m as u32 / cfg.bm, splits as u32),
        CubeDim::new_2d(plane, cfg.pm * cfg.pn),
        arg(a, m * k),
        arg(b, k * n),
        arg(dst, splits * m * n),
        m as u32,
        n as u32,
        lda,
        ldb,
        slice as u32,
        stages as u32,
        sh.ta,
        sh.tb,
        cfg.bm,
        cfg.bn,
        cfg.pm,
        cfg.pn,
        cfg.pad,
        threads,
    );
    if !direct {
        k_sum::launch(client(), CubeCount::Static((m * n).div_ceil(256) as u32, 1, 1), CubeDim::new_1d(256), arg(scratch, splits * m * n), arg(out, m * n), (m * n) as u32, splits as u32, sh.acc);
    }
}

/// (best, median) ms per launch.
fn time(mut launch: impl FnMut(), sync: &Handle) -> (f64, f64) {
    launch();
    launch();
    let _ = read(sync);
    let mut t: Vec<f64> = (0..ROUNDS)
        .map(|_| {
            let t0 = Instant::now();
            for _ in 0..REPS {
                launch();
            }
            let _ = read(sync);
            t0.elapsed().as_secs_f64() * 1e3 / REPS as f64
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (t[0], t[ROUNDS / 2])
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs()));
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max) / scale
}

fn main() {
    let _lease = gpu_lease::hold(Kind::Shared, "scratchtape matmul_cmma_spike", Duration::from_secs(15 * 60));
    let plane = client().properties().hardware.plane_size_max;
    println!("adapter/backend: default client (DiscreteGpu(0)); plane size {plane}; {ROUNDS} rounds x {REPS} launches; shared GPU, best-of-N");
    // (m, k, n, ta, tb, accumulate), per-step sites, each launched 4x.
    let shapes = [
        Shape { m: 256, k: 2048, n: 512, ta: true, tb: false, acc: true },
        Shape { m: 2048, k: 768, n: 256, ta: false, tb: true, acc: false },
        Shape { m: 256, k: 2048, n: 768, ta: true, tb: false, acc: true },
        Shape { m: 512, k: 2048, n: 256, ta: true, tb: false, acc: true },
        Shape { m: 2048, k: 256, n: 768, ta: false, tb: false, acc: false },
        Shape { m: 2048, k: 512, n: 256, ta: false, tb: true, acc: false },
        Shape { m: 2048, k: 256, n: 512, ta: false, tb: true, acc: false },
        Shape { m: 2048, k: 512, n: 256, ta: false, tb: false, acc: false },
        Shape { m: 256, k: 2048, n: 256, ta: true, tb: false, acc: true },
        Shape { m: 2048, k: 256, n: 512, ta: false, tb: false, acc: false },
        // Ceiling probes (not training shapes): big enough that launch cost is negligible.
        Shape { m: 4096, k: 4096, n: 4096, ta: false, tb: false, acc: false },
        Shape { m: 4096, k: 4096, n: 4096, ta: true, tb: false, acc: false },
        Shape { m: 4096, k: 4096, n: 4096, ta: false, tb: true, acc: false },
    ];
    // Launch floor: a trivial kernel, same batching.
    {
        let (p, o) = (upload_f32(&[0.0; 256]), upload_f32(&[0.0; 256]));
        let tiny = upload_f32(&[0.0]);
        let (best, med) = time(|| k_sum::launch(client(), CubeCount::Static(1, 1, 1), CubeDim::new_1d(256), arg(&p, 256), arg(&o, 256), 256, 1, false), &tiny);
        println!("launch floor (256-element sum kernel): best {best:.4} ms  median {med:.4} ms per launch");
    }
    let cfgs = [
        Cfg { bm: 64, bn: 64, pm: 2, pn: 2, pad: 8 },
        Cfg { bm: 64, bn: 64, pm: 2, pn: 2, pad: 0 },
        Cfg { bm: 128, bn: 64, pm: 4, pn: 2, pad: 8 },
        Cfg { bm: 64, bn: 128, pm: 2, pn: 2, pad: 8 },
        Cfg { bm: 128, bn: 128, pm: 2, pn: 2, pad: 8 },
    ];
    let mut rng = Rng::new(11);
    let tiny = upload_f32(&[0.0]);
    let mut summary = vec![];
    for sh in &shapes {
        let (m, k, n) = (sh.m, sh.k, sh.n);
        let name = format!("[{m}x{k}]{}.[{k}x{n}]{}{}", if sh.ta { "T" } else { "N" }, if sh.tb { "T" } else { "N" }, if sh.acc { " +=" } else { "" });
        let av: Vec<f32> = (0..m * k).map(|_| rng.next_gaussian()).collect();
        let bv: Vec<f32> = (0..k * n).map(|_| rng.next_gaussian()).collect();
        let (ah, bh) = (upload_f32(&av), upload_f32(&bv));
        let zeros = vec![0.0f32; m * n];
        let flops = 2.0 * (m * k * n) as f64;
        fn mk(h: &Handle, trans: bool) -> MatRef<'_> {
            MatRef { trans, ..MatRef::new(h) }
        }
        let epi = Epilogue { accumulate: sh.acc, ..Default::default() };

        // f32 reference (into a zeroed out, so += is a plain product).
        let want_h = upload_f32(&zeros);
        matmul(mk(&ah, sh.ta), mk(&bh, sh.tb), MatRef::new(&want_h), 1, m, k, n, epi);
        let want = read(&want_h);
        let o32 = upload_f32(&zeros);
        let (f32_best, f32_med) = time(|| matmul(mk(&ah, sh.ta), mk(&bh, sh.tb), MatRef::new(&o32), 1, m, k, n, epi), &tiny);

        println!("\n{name}: f32 best {f32_best:.4} ms  median {f32_med:.4} ms  ({:.2} TFLOP/s)", flops / f32_best / 1e9);
        let scratch = client().empty(if sh.acc { 8 } else { 1 } * m * n * 4);
        let mut best: Option<(f64, f64, String, f32)> = None;
        for cfg in cfgs {
            if m % cfg.bm as usize != 0 || n % cfg.bn as usize != 0 {
                continue;
            }
            let split_opts: &[usize] = if sh.acc { &[1, 2, 4, 8] } else { &[1] };
            for &sp in split_opts {
                let o = upload_f32(&zeros);
                cmma_run(sh, cfg, sp, plane, &ah, &bh, &o, &scratch);
                let err = rel_err(&read(&o), &want);
                let (bt, mt) = time(|| cmma_run(sh, cfg, sp, plane, &ah, &bh, &o, &scratch), &tiny);
                let label = cfg.name(sp);
                println!("    cmma {label:<22} best {bt:.4} ms  median {mt:.4} ms  {:.2} TFLOP/s  err {err:.2e}", flops / bt / 1e9);
                if best.as_ref().is_none_or(|b| bt < b.0) {
                    best = Some((bt, mt, label, err));
                }
            }
        }
        summary.push((name, f32_best, f32_med, flops, best.unwrap()));
    }
    println!("\n{:<28} {:>9} {:>9} {:>8} {:>8} {:>8} {:>9}  best config", "shape", "f32 ms", "cmma ms", "speedup", "f32 TF", "cmma TF", "rel err");
    let (mut t32, mut tc) = (0.0, 0.0);
    for (i, (name, f32_best, _f32_med, flops, (bt, _mt, label, err))) in summary.iter().enumerate() {
        if i < 10 {
            t32 += f32_best;
            tc += bt;
        } else if i == 10 {
            println!("-- ceiling probes --");
        }
        println!("{name:<28} {f32_best:>9.4} {bt:>9.4} {:>7.2}x {:>8.2} {:>8.2} {err:>9.2e}  {label}", f32_best / bt, flops / f32_best / 1e9, flops / bt / 1e9);
    }
    println!("sum over the 10 sites: f32 {t32:.3} ms, cmma {tc:.3} ms ({:.2}x)", t32 / tc);
}
