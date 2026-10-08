// How fast is the kNN search's matrix-core product, and does register blocking help? knn.rs's `k_dots_cmma` has one plane
// compute one 16x16 output tile, loading every A and B fragment from global memory with no reuse. This times it against
// planes that hold 2x2 and 4x4 accumulator tiles (each loaded fragment feeds 2 or 4 products) at the real shape:
// keys [16384, 256] f16 against 1024 queries, out [16384, 1024] f32. Reports ms per product and TFLOP/s (best and median
// of ROUNDS), and the largest difference from the baseline output. Exclusive GPU lease.
// Run: cargo run --release --example cmma_blocking_check
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::gpu_step::{client, read, upload_f32};
use scratchtape::nn::Rng;
use std::time::{Duration, Instant};

const M: usize = 16384;
const N: usize = 1024;
const K: usize = 256;
const ROUNDS: usize = 7;
const REPS: usize = 8;

#[cube(launch)]
fn k_1x1(a: &[f16], b: &[f16], out: &mut [f32], #[comptime] k: u32, n: u32) {
    let row0 = CUBE_POS_Y * 16;
    let col0 = CUBE_POS_X * 16;
    let c = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    #[unroll]
    for kk in 0..k / 16 {
        let at = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &a[(row0 * k + kk * 16) as usize..a.len()], k);
        let bt = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &b[(kk * 16 * n + col0) as usize..b.len()], n);
        cmma::execute(&at, &bt, &c, &c);
    }
    let len = out.len();
    cmma::store(&mut out[(row0 * n + col0) as usize..len], &c, n, cmma::MatrixLayout::RowMajor);
}

#[cube(launch)]
fn k_2x2(a: &[f16], b: &[f16], out: &mut [f32], #[comptime] k: u32, n: u32) {
    let row0 = CUBE_POS_Y * 32;
    let col0 = CUBE_POS_X * 32;
    let c00 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c01 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c10 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c11 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    #[unroll]
    for kk in 0..k / 16 {
        let a0 = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &a[(row0 * k + kk * 16) as usize..a.len()], k);
        let a1 = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &a[((row0 + 16) * k + kk * 16) as usize..a.len()], k);
        let b0 = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &b[(kk * 16 * n + col0) as usize..b.len()], n);
        let b1 = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &b[(kk * 16 * n + col0 + 16) as usize..b.len()], n);
        cmma::execute(&a0, &b0, &c00, &c00);
        cmma::execute(&a0, &b1, &c01, &c01);
        cmma::execute(&a1, &b0, &c10, &c10);
        cmma::execute(&a1, &b1, &c11, &c11);
    }
    let len = out.len();
    cmma::store(&mut out[(row0 * n + col0) as usize..len], &c00, n, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut out[(row0 * n + col0 + 16) as usize..len], &c01, n, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut out[((row0 + 16) * n + col0) as usize..len], &c10, n, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut out[((row0 + 16) * n + col0 + 16) as usize..len], &c11, n, cmma::MatrixLayout::RowMajor);
}

fn arg(h: &Handle, len: usize) -> BufferArg {
    // Safety: every handle here holds at least `len` elements of the kernel's type.
    unsafe { BufferArg::from_raw_parts(h.clone(), len) }
}

fn main() {
    let _lease = gpu_lease::hold(Kind::Exclusive, "scratchtape cmma_blocking_check", Duration::from_secs(15 * 60));
    let mut rng = Rng::new(3);
    let a: Vec<f16> = (0..M * K).map(|_| f16::from_f32(rng.next_gaussian())).collect();
    let b: Vec<f16> = (0..K * N).map(|_| f16::from_f32(rng.next_gaussian())).collect();
    let (ah, bh) = (client().create_from_slice(f16::as_bytes(&a)), client().create_from_slice(f16::as_bytes(&b)));
    let out = client().empty(M * N * 4);
    let tiny = upload_f32(&[0.0]);
    let plane = client().properties().hardware.plane_size_max;
    println!("plane size {plane}");
    let flops = 2.0 * (M * N * K) as f64;
    let mut base: Vec<f32> = vec![];
    for (name, tile) in [("1x1", 1usize), ("2x2", 2)] {
        let launch = || {
            let (cx, cy) = ((N / (16 * tile)) as u32, (M / (16 * tile)) as u32);
            let (ar, br, or) = (arg(&ah, M * K), arg(&bh, K * N), arg(&out, M * N));
            if tile == 1 {
                k_1x1::launch(client(), CubeCount::Static(cx, cy, 1), CubeDim::new_1d(plane), ar, br, or, K as u32, N as u32);
            } else {
                k_2x2::launch(client(), CubeCount::Static(cx, cy, 1), CubeDim::new_1d(plane), ar, br, or, K as u32, N as u32);
            }
        };
        launch();
        let _ = read(&tiny);
        let mut t: Vec<f64> = (0..ROUNDS)
            .map(|_| {
                let t0 = Instant::now();
                for _ in 0..REPS {
                    launch();
                }
                // A read of a small buffer waits for the in-order queue (reading `out` per round would time the copy).
                let _ = read(&tiny);
                t0.elapsed().as_secs_f64() * 1e3 / REPS as f64
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let got = read(&out);
        let diff = if base.is_empty() { 0.0 } else { got.iter().zip(&base).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max) };
        if base.is_empty() {
            base = got;
        }
        println!("{name}: best {:.2} ms ({:.1} TFLOP/s)  median {:.2} ms  max |diff| vs 1x1 {diff}", t[0], flops / t[0] / 1e9, t[ROUNDS / 2]);
    }
}
