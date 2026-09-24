// Same measurement as examples/gpu/gpu_dispatch_overhead.rs, written once as
// cubecl kernels and run on each runtime:
//   cubecl_spike wgpu-dx12 | wgpu-vulkan | cpu
// Per shape: queued cost per dispatch (N chained matmuls, one sync), and
// queued cost per elementwise launch, vs scratchtape's single-threaded
// NdArray::matmul. Raw-wgpu reference (gpu_dispatch_overhead.rs, DX12):
// ~0.011 ms at 64^3, 0.022 ms at (512,128)@(128,128).
//
// Result 2026-09-24, cubecl 0.11.0-pre.3, RX 9060 XT eGPU idle, CPU idle
// (ms per queued launch; matmul GFLOP/s in brackets):
//   shape                 dx12            vulkan          cpu runtime     elementwise dx12/vk/cpu  NdArray
//   (64,64)@(64,64)       0.011 [49]      0.016 [33]      3.43 [0.2]      0.008/0.010/3.38         0.06
//   (512,128)@(128,128)   0.027 [620]     0.024 [703]     5.14 [3.3]      0.008/0.010/4.94         1.3-1.6
//   (512,256)@(256,256)   0.095 [707]     0.096 [702]     14.6 [4.6]      0.007/0.010/5.42         4.3-5.2
//   (2048,128)@(128,128)  0.095 [709]     0.075 [900]     9.48 [7.1]      0.008/0.009/5.85         5.0-5.4
// GPU: parity with raw wgpu (gpu_dispatch_overhead.rs), ~10 us/launch.
// First DX12 run: 19 s total, shader compile leaked into one timing (0.9 ms
// outlier); warm rerun clean. CPU runtime: a ~3.4 ms floor per launch even
// for a trivial elementwise kernel, and the naive matmul is slower than
// single-threaded NdArray::matmul - a ~2400-launch training step would cost
// ~8 s on it. Unusable at this granularity; "one source for GPU and CPU"
// doesn't hold here yet. JIT: 170 s total for the CPU run.
// Build friction: fresh resolve fails (pliron 0.17 vs pliron-derive 0.18,
// fixed by humble-cortex's Cargo.lock); the CPU runtime needed two local
// patches (see Cargo.toml [patch.crates-io] and PATCHES.md).
//
// Tuning (cpu-sweep, 2026-09-24). The CPU runtime exposes no speed options
// (only CUBECL_CPU_STACK_*). Its threadpool makes one task per *unit* of a
// cube, and each task loops over every cube, so units-per-cube is the
// parallelism, and each launch waits for the previous one to finish. Launch
// shape is the only lever. Elementwise over 65536 floats, matmul
// (512,128)@(128,128) with a 1D kernel, ms per queued launch:
//   units/cube   empty   elementwise  matmul   (stock 200 us idle poll)
//   1            0.001   0.051        6.43     single-threaded, ~native
//   8            5.0     3.17         5.92
//   64           1.07    3.85         6.90
// Handing work to multiple threads costs ms: idle workers park after
// 200 us and waking them stalls. With IDLE_POLL patched to 20 ms
// (PATCHES.md #3):
//   1            0.002   0.044        5.93
//   8            0.047   0.288        1.10
//   64           0.196   0.342        0.81     (NdArray: 1.3-1.6)
// Best case: small ops at 1 unit (us launch, one thread), matmuls at 8-64
// units (~1.6-2x faster than NdArray even with a naive, CPU-unfriendly loop
// order). That's a per-op launch policy plus a forked constant that burns
// idle CPU, for roughly what std::thread on NdArray gives directly.
//
// Extras (`cubecl_spike wgpu-vulkan extras`, 2026-09-24): comptime-k
// matmul, fresh output per launch, matrix cores.
// Matrix cores: vulkaninfo shows VK_KHR_cooperative_matrix on the RX 9060 XT
// (and the 780M). cubecl reports cmma only when built `--features vulkan`
// (its own SPIR-V compiler): 12 configs, f16x16x16->f32 included. The
// default WGSL path (DX12 or Vulkan) reports 0.
// Correctness (pre.4, SPIR-V): k_matmul_cmma matches the f32 CPU product to
// f16 precision (max err 3e-5 to 8e-5) at all four shapes that ran. Timings
// with selfplay-burn sharing the eGPU were floor-bound (~0.8 ms per launch
// for every kernel, against 0.01-0.1 ms idle), so they say nothing
// about cmma speed.
// Idle eGPU (12:30, after the other jobs exited), two rounds, ms per queued
// launch [cmma GFLOP/s]; round 1 / round 2:
//   (64,64)       naive 0.015/0.016  comptime 0.009/0.009  fresh 0.015/0.015  cmma 0.010/0.007
//   (512,128)     0.048/0.053        0.015/0.014           0.050/0.052        0.008/0.007 [2120/2456]
//   (512,256)     0.117/0.123        0.057/0.067           0.119/0.125        0.012/0.013 [5625/5301]
//   (2048,128)    0.110/0.109        0.068/0.060           0.110/0.110        0.012/0.012 [5809/5756]
//   (2048,512)    1.251/1.239        0.507/0.508           1.250/1.257        0.089/0.277 [12041/3878]
// cmma is 5-14x faster than naive and sits at the launch floor for step
// shapes. comptime k gives 2-3x on its own. Fresh allocation is free
// (pooled).
// Tiled f32 baseline (k_matmul_tiled: 64x64 tile per cube, 4x4 per unit,
// shared-memory slabs). Correct to 1e-3 at every shape. The eGPU was
// shared with selfplay-burn-head, so these are best of two rounds, ms
// [GFLOP/s]:
//   (64,64)       tiled 0.014 [39]     cmma 0.009 [62]
//   (512,128)     0.017 [970]          0.008 [2079]
//   (512,256)     0.055 [1211]         0.026 [2563]
//   (2048,128)    0.034 [1978]         0.011 [6259]
//   (2048,512)    0.226 [4748]         0.100 [10789]
// Tiled f32 is 1.1-5.5x faster than naive. cmma keeps a 2-3x edge over
// it, so most of the naive-vs-cmma gap was memory layout, not matrix
// cores. The tiled kernel is simple (no vector loads, no double
// buffering), so 2-3x is an upper bound on what cmma adds.
// Earlier: another session's long GPU job (humble-cortex) saturated
// the eGPU, and every launch loop - even plain `bench` - stalled
// indefinitely at ~0 CPU. It looked like a comptime-kernel hang until
// `bench` stalled too. On a shared eGPU a stall is contention first,
// not a bug.
use cubecl::prelude::*;
use cubecl::client::Client;
use cubecl::server::Handle;
use cubecl_runtime::runtime::Runtime;
use half::f16;
use scratchtape::nn::Rng;
use scratchtape::tensor::NdArray;
use std::time::Instant;

const N: usize = 500;
const REPS: usize = 5;

/// Naive matmul, one unit per output element: out[m,n] = a[m,k] @ b[k,n].
/// Same algorithm as src/matmul.wgsl, so differences are runtime, not kernel.
#[cube(launch)]
fn k_matmul(a: &[f32], b: &[f32], out: &mut [f32], m: u32, k: u32, n: u32) {
    let row = ABSOLUTE_POS_Y;
    let col = ABSOLUTE_POS_X;
    if row < m && col < n {
        let mut sum = 0.0f32;
        for p in 0..k {
            sum += a[(row * k + p) as usize] * b[(p * n + col) as usize];
        }
        out[(row * n + col) as usize] = sum;
    }
}

/// Elementwise stand-in for the non-matmul half of a step (bias + ReLU-ish).
#[cube(launch)]
fn k_elementwise(x: &[f32], out: &mut [f32], len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        let v = x[i] * 0.5f32 + 0.1f32;
        out[i] = f32::max(v, 0.0f32);
    }
}

/// 1D matmul: one unit per output element, for launch-shape sweeps.
#[cube(launch)]
fn k_matmul_1d(a: &[f32], b: &[f32], out: &mut [f32], m: u32, k: u32, n: u32) {
    let idx = ABSOLUTE_POS as u32;
    if idx < m * n {
        let row = idx / n;
        let col = idx % n;
        let mut sum = 0.0f32;
        for p in 0..k {
            sum += a[(row * k + p) as usize] * b[(p * n + col) as usize];
        }
        out[idx as usize] = sum;
    }
}

/// Does nothing: the pure launch floor.
#[cube(launch)]
fn k_empty(out: &mut [f32]) {
    if ABSOLUTE_POS == 0 {
        out[0] = out[0];
    }
}

/// CPU runtime launch-shape sweep. Its threadpool makes one task per unit of
/// the cube, and each task loops over every cube, so units-per-cube is the
/// parallelism and cube count is serial work per task.
fn sweep(client: Client) {
    let sync = |c: &Client| cubecl::future::block_on(c.sync()).unwrap();
    let n = 200;
    let mut rng = Rng::new(7);
    let (m, k) = (512usize, 128usize);
    let a = rand(&mut rng, m, k);
    let b = rand(&mut rng, k, k);
    let a_h = client.create_from_slice(f32::as_bytes(&a.data));
    let b_h = client.create_from_slice(f32::as_bytes(&b.data));
    let o_h = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * k]));
    let len = m * k;
    println!("units/cube | cubes | empty ms | elementwise(65536) ms | matmul (512,128)@(128,128) ms");
    for &d in &[1u32, 4, 8, 16, 32, 64, 256] {
        let cubes = (len as u32).div_ceil(d);
        let empty = best(&mut || {
            for _ in 0..n {
                k_empty::launch(&client, CubeCount::Static(cubes, 1, 1), CubeDim::new_1d(d), buf(&o_h, len));
            }
            sync(&client);
        }) / n as f64;
        let ew = best(&mut || {
            for _ in 0..n {
                k_elementwise::launch(&client, CubeCount::Static(cubes, 1, 1), CubeDim::new_1d(d), buf(&a_h, len), buf(&o_h, len), len as u32);
            }
            sync(&client);
        }) / n as f64;
        let mm_n = 20;
        let mm = best(&mut || {
            for _ in 0..mm_n {
                k_matmul_1d::launch(&client, CubeCount::Static(cubes, 1, 1), CubeDim::new_1d(d), buf(&a_h, m * k), buf(&b_h, k * k), buf(&o_h, len), m as u32, k as u32, k as u32);
            }
            sync(&client);
        }) / mm_n as f64;
        let got = f32::from_bytes(&client.read_one(o_h.clone()).unwrap())[..len].to_vec();
        let want = a.matmul(&b);
        let err = got.iter().zip(&want.data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-3, "1d matmul wrong at d={d}: {err}");
        println!("{d:>10} | {cubes:>6} | {:>8.3} | {:>8.3} | {:>8.3}", empty * 1e3, ew * 1e3, mm * 1e3);
    }
}

/// k as a comptime constant: the loop bound is baked into the compiled
/// kernel (one compile per distinct k), so it can be fully unrolled.
#[cube(launch)]
fn k_matmul_ct(a: &[f32], b: &[f32], out: &mut [f32], m: u32, n: u32, #[comptime] k: u32) {
    let row = ABSOLUTE_POS_Y;
    let col = ABSOLUTE_POS_X;
    if row < m && col < n {
        let mut sum = 0.0f32;
        #[unroll]
        for p in 0..k {
            sum += a[(row * k + p) as usize] * b[(p * n + col) as usize];
        }
        out[(row * n + col) as usize] = sum;
    }
}

/// Tiled f32 matmul, the fair non-matrix-core baseline: each 16x16 cube
/// computes a 64x64 output tile, each unit a 4x4 block held in registers,
/// with A and B staged through shared memory in 64x16 / 16x64 slabs.
/// Needs m % 64 == 0, n % 64 == 0, k % 16 == 0.
#[cube(launch)]
fn k_matmul_tiled(a: &[f32], b: &[f32], out: &mut [f32], #[comptime] k: u32, #[comptime] n: u32) {
    let tx = UNIT_POS_X;
    let ty = UNIT_POS_Y;
    let tid = ty * 16 + tx;
    let row0 = CUBE_POS_Y * 64;
    let col0 = CUBE_POS_X * 64;
    let mut a_s = Shared::<[f32]>::new_slice(1024usize);
    let mut b_s = Shared::<[f32]>::new_slice(1024usize);
    let mut acc = Array::<f32>::new(16usize);
    let mut bv = Array::<f32>::new(4usize);
    for kk in 0..k / 16 {
        #[unroll]
        for i in 0..4u32 {
            let e = tid + 256 * i;
            a_s[e as usize] = a[((row0 + e / 16) * k + kk * 16 + e % 16) as usize];
            b_s[e as usize] = b[((kk * 16 + e / 64) * n + col0 + e % 64) as usize];
        }
        sync_cube();
        #[unroll]
        for p in 0..16u32 {
            #[unroll]
            for j in 0..4u32 {
                bv[j as usize] = b_s[(p * 64 + tx * 4 + j) as usize];
            }
            #[unroll]
            for i in 0..4u32 {
                let av = a_s[((ty * 4 + i) * 16 + p) as usize];
                #[unroll]
                for j in 0..4u32 {
                    acc[(i * 4 + j) as usize] += av * bv[j as usize];
                }
            }
        }
        sync_cube();
    }
    #[unroll]
    for i in 0..4u32 {
        #[unroll]
        for j in 0..4u32 {
            out[((row0 + ty * 4 + i) * n + col0 + tx * 4 + j) as usize] = acc[(i * 4 + j) as usize];
        }
    }
}

/// Matrix-core matmul: out[M,N] (f32) = a[M,K] @ b[K,N] (f16, row-major).
/// One plane per cube computes one 16x16 output tile, stepping K by 16.
#[cube(launch)]
fn k_matmul_cmma(a: &[f16], b: &[f16], out: &mut [f32], #[comptime] k: u32, #[comptime] n: u32) {
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

/// Extra measurements beyond `bench`: comptime specialization, per-launch
/// fresh output allocation (what a GPU-resident tape does per op), and
/// matrix cores where the runtime reports them.
fn extras(client: Client) {
    let sync = |c: &Client| cubecl::future::block_on(c.sync()).unwrap();
    let f16_cfg = cubecl::features::MmaConfig {
        a_type: cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16),
        b_type: cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16),
        cd_type: cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        m: 16,
        k: 16,
        n: 16,
    };
    let cmma_ok = client.features().matmul.cmma.contains(&f16_cfg);
    let plane = client.properties().hardware.plane_size_max;
    println!("cmma f16x16x16->f32 supported: {cmma_ok}; plane size max {plane}; all cmma configs: {}", client.features().matmul.cmma.len());
    println!("shape | runtime-k ms | comptime-k ms | tiled ms [GFLOP/s] | fresh-alloc ms (runtime k) | cmma ms [GFLOP/s]");
    let mut rng = Rng::new(7);
    for &(m, k) in &[(64usize, 64usize), (512, 128), (512, 256), (2048, 128), (2048, 512)] {
        let n = k;
        let a = rand(&mut rng, m, k);
        let b = rand(&mut rng, k, n);
        let a_h = client.create_from_slice(f32::as_bytes(&a.data));
        let b_h = client.create_from_slice(f32::as_bytes(&b.data));
        let o_h = client.empty(m * n * 4);
        let dim = CubeDim::new_2d(8, 8);
        let count = CubeCount::Static((n as u32).div_ceil(8), (m as u32).div_ceil(8), 1);
        let rt = best(&mut || {
            for _ in 0..N {
                k_matmul::launch(&client, count.clone(), dim, buf(&a_h, m * k), buf(&b_h, k * n), buf(&o_h, m * n), m as u32, k as u32, n as u32);
            }
            sync(&client);
        }) / N as f64;
        k_matmul_ct::launch(&client, count.clone(), dim, buf(&a_h, m * k), buf(&b_h, k * n), buf(&o_h, m * n), m as u32, n as u32, k as u32);
        let got = f32::from_bytes(&client.read_one(o_h.clone()).unwrap())[..m * n].to_vec();
        let want = a.matmul(&b);
        let err = got.iter().zip(&want.data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-3, "comptime matmul wrong: {err}");
        let ct = best(&mut || {
            for _ in 0..N {
                k_matmul_ct::launch(&client, count.clone(), dim, buf(&a_h, m * k), buf(&b_h, k * n), buf(&o_h, m * n), m as u32, n as u32, k as u32);
            }
            sync(&client);
        }) / N as f64;
        let tcount = CubeCount::Static((n / 64) as u32, (m / 64) as u32, 1);
        let tdim = CubeDim::new_2d(16, 16);
        let o_t = client.empty(m * n * 4);
        k_matmul_tiled::launch(&client, tcount.clone(), tdim, buf(&a_h, m * k), buf(&b_h, k * n), buf(&o_t, m * n), k as u32, n as u32);
        let got = f32::from_bytes(&client.read_one(o_t.clone()).unwrap())[..m * n].to_vec();
        let err = got.iter().zip(&want.data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-3, "tiled matmul wrong: {err}");
        let tiled = best(&mut || {
            for _ in 0..N {
                k_matmul_tiled::launch(&client, tcount.clone(), tdim, buf(&a_h, m * k), buf(&b_h, k * n), buf(&o_t, m * n), k as u32, n as u32);
            }
            sync(&client);
        }) / N as f64;
        // A tape allocates every op's output; handles drop back to the pool.
        let fresh = best(&mut || {
            let mut keep = Vec::with_capacity(N);
            for _ in 0..N {
                let o = client.empty(m * n * 4);
                k_matmul::launch(&client, count.clone(), dim, buf(&a_h, m * k), buf(&b_h, k * n), buf(&o, m * n), m as u32, k as u32, n as u32);
                keep.push(o);
            }
            sync(&client);
        }) / N as f64;
        let cm = if cmma_ok {
            let a16: Vec<f16> = a.data.iter().map(|&x| f16::from_f32(x)).collect();
            let b16: Vec<f16> = b.data.iter().map(|&x| f16::from_f32(x)).collect();
            let a16_h = client.create_from_slice(f16::as_bytes(&a16));
            let b16_h = client.create_from_slice(f16::as_bytes(&b16));
            let cc = CubeCount::Static((n / 16) as u32, (m / 16) as u32, 1);
            let cd = CubeDim::new_1d(plane);
            let launch = || unsafe {
                k_matmul_cmma::launch(&client, cc.clone(), cd, BufferArg::from_raw_parts(a16_h.clone(), m * k), BufferArg::from_raw_parts(b16_h.clone(), k * n), BufferArg::from_raw_parts(o_h.clone(), m * n), k as u32, n as u32)
            };
            launch();
            let got = f32::from_bytes(&client.read_one(o_h.clone()).unwrap())[..m * n].to_vec();
            let err = got.iter().zip(&want.data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
            let t = best(&mut || { for _ in 0..N { launch(); } sync(&client); }) / N as f64;
            format!("{:.4} [{:.0}] (f16 max err {err:.1e})", t * 1e3, 2.0 * (m * k * n) as f64 / t / 1e9)
        } else {
            "n/a".into()
        };
        println!("({m},{k})@({k},{n}) | {:.4} | {:.4} | {:.4} [{:.0}] | {:.4} | {cm}", rt * 1e3, ct * 1e3, tiled * 1e3, 2.0 * (m * k * n) as f64 / tiled / 1e9, fresh * 1e3);
    }
}

fn buf(h: &Handle, len: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), len) }
}

fn rand(rng: &mut Rng, rows: usize, cols: usize) -> NdArray {
    NdArray::new((0..rows * cols).map(|_| rng.next_f32() * 0.2 - 0.1).collect(), vec![rows, cols])
}

fn best(f: &mut dyn FnMut()) -> f64 {
    (0..REPS).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::MAX, f64::min)
}

fn bench(client: Client) {
    let sync = |c: &Client| cubecl::future::block_on(c.sync()).unwrap();
    println!("shape (m,k)@(k,k) | queued matmul ms/dispatch | GFLOP/s | queued elementwise ms/launch | cpu NdArray::matmul ms");
    let mut rng = Rng::new(7);
    for &(m, k) in &[(64usize, 64usize), (512, 128), (512, 256), (2048, 128)] {
        let a = rand(&mut rng, m, k);
        let b = rand(&mut rng, k, k);
        let ping = [client.create_from_slice(f32::as_bytes(&a.data)), client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * k]))];
        let b_h = client.create_from_slice(f32::as_bytes(&b.data));
        let dim = CubeDim::new_2d(8, 8);
        let count = CubeCount::Static((k as u32).div_ceil(8), (m as u32).div_ceil(8), 1);
        let launch = |i: usize| {
            k_matmul::launch(&client, count.clone(), dim, buf(&ping[i % 2], m * k), buf(&b_h, k * k), buf(&ping[1 - i % 2], m * k), m as u32, k as u32, k as u32);
        };

        // Check: one launch reproduces the CPU product.
        launch(0);
        let got = f32::from_bytes(&client.read_one(ping[1].clone()).unwrap())[..m * k].to_vec();
        let want = a.matmul(&b);
        let err = got.iter().zip(&want.data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-3, "cubecl matmul disagrees with CPU: max err {err}");

        let queued = best(&mut || { for i in 0..N { launch(i); } sync(&client); }) / N as f64;
        let len = m * k;
        let ew = best(&mut || {
            for i in 0..N {
                k_elementwise::launch(&client, CubeCount::Static((len as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256), buf(&ping[i % 2], len), buf(&ping[1 - i % 2], len), len as u32);
            }
            sync(&client);
        }) / N as f64;
        let cpu_calls = if m * k * k > 10_000_000 { 5 } else { 50 };
        let cpu = best(&mut || for _ in 0..cpu_calls { std::hint::black_box(a.matmul(&b)); }) / cpu_calls as f64;
        let flops = 2.0 * (m * k * k) as f64;
        println!("({m:>4},{k:>3})@({k},{k}) | {:>9.4} | {:>7.1} | {:>9.4} | {:>8.3}", queued * 1e3, flops / queued / 1e9, ew * 1e3, cpu * 1e3);
    }
}

/// Single-source check (2026-09-24): JIT cost per kernel, tiled-matmul
/// correctness, and a step-shaped chain of 144 launches (72 x tiled matmul
/// (512,128)@(128,128) then elementwise), one sync, against the same chain
/// on NdArray. `tiled` picks the shared-memory kernel (GPU shape); the CPU
/// runtime gets the barrier-free naive one, because it runs a barrier
/// kernel's cube as one spinning OS thread per unit: 256 threads on <=16
/// cores never finished a single tiled launch in 12 min (2026-09-24).
fn step(client: Client, tiled: bool) {
    let sync = |c: &Client| cubecl::future::block_on(c.sync()).unwrap();
    let (m, k) = (512usize, 128usize);
    let mut rng = Rng::new(11);
    let x = rand(&mut rng, m, k);
    let w = rand(&mut rng, k, k);
    // A fresh ping-pong pair starting at x (the CPU runtime's Bytes-based write is awkward).
    let fresh = || [client.create_from_slice(f32::as_bytes(&x.data)), client.empty(m * k * 4)];
    let w_h = client.create_from_slice(f32::as_bytes(&w.data));
    let tcount = CubeCount::Static((k / 64) as u32, (m / 64) as u32, 1);
    let tdim = CubeDim::new_2d(16, 16);
    let ecount = CubeCount::Static(((m * k) as u32).div_ceil(256), 1, 1);
    let ncount = CubeCount::Static((k as u32).div_ceil(8), (m as u32).div_ceil(8), 1);
    let mm = |p: &[Handle; 2], src: usize| {
        if tiled {
            k_matmul_tiled::launch(&client, tcount.clone(), tdim, buf(&p[src], m * k), buf(&w_h, k * k), buf(&p[1 - src], m * k), k as u32, k as u32)
        } else {
            k_matmul::launch(&client, ncount.clone(), CubeDim::new_2d(8, 8), buf(&p[src], m * k), buf(&w_h, k * k), buf(&p[1 - src], m * k), m as u32, k as u32, k as u32)
        }
    };
    let mm_name = if tiled { "tiled matmul" } else { "naive matmul" };
    let ew = |p: &[Handle; 2], src: usize| k_elementwise::launch(&client, ecount.clone(), CubeDim::new_1d(256), buf(&p[src], m * k), buf(&p[1 - src], m * k), (m * k) as u32);

    // JIT: the first launch of a kernel compiles it; the second doesn't.
    let scratch = fresh();
    for (name, f) in [(mm_name, &mm as &dyn Fn(&[Handle; 2], usize)), ("elementwise", &ew)] {
        let t = Instant::now();
        f(&scratch, 0);
        sync(&client);
        let first = t.elapsed().as_secs_f64();
        let t = Instant::now();
        f(&scratch, 0);
        sync(&client);
        println!("jit {name}: first launch {:.1} ms, second {:.3} ms", first * 1e3, t.elapsed().as_secs_f64() * 1e3);
    }

    // Correctness: one tiled matmul from x.
    let ping = fresh();
    mm(&ping, 0);
    let got = f32::from_bytes(&client.read_one(ping[1].clone()).unwrap())[..m * k].to_vec();
    let want = x.matmul(&w);
    let err = got.iter().zip(&want.data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
    assert!(err < 1e-3, "{mm_name} wrong on this runtime: max err {err}");
    println!("{mm_name} correct (max err {err:.2e})");

    // The chain, and its NdArray twin. Each pair returns to ping[0].
    let pairs = 72;
    let ping = fresh();
    let chain = || { for _ in 0..pairs { mm(&ping, 0); ew(&ping, 1); } sync(&client); };
    chain();
    let got = f32::from_bytes(&client.read_one(ping[0].clone()).unwrap())[..m * k].to_vec();
    let nd_chain = || {
        let mut v = x.clone();
        for _ in 0..pairs {
            v = v.matmul(&w);
            v.data.iter_mut().for_each(|e| *e = (*e * 0.5 + 0.1).max(0.0));
        }
        v
    };
    let want = nd_chain();
    let err = got.iter().zip(&want.data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
    assert!(err < 1e-3, "chain disagrees with NdArray: max err {err}");
    let dev = best(&mut || chain());
    let nd = best(&mut || { std::hint::black_box(nd_chain()); });
    println!("chain of {} launches, one sync: {:.2} ms ({:.1} us/launch); NdArray single-thread {:.2} ms; max err {err:.2e}", 2 * pairs, dev * 1e3, dev / (2 * pairs) as f64 * 1e6, nd * 1e3);
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "wgpu-dx12".into());
    let t0 = Instant::now();
    match which.as_str() {
        "wgpu-dx12" | "wgpu-vulkan" => {
            use cubecl::wgpu::{Dx12, RuntimeOptions, Vulkan, WgpuDevice, WgpuRuntime, init_setup};
            let device = WgpuDevice::default();
            let setup = if which == "wgpu-dx12" {
                init_setup::<Dx12>(&device, RuntimeOptions::default())
            } else {
                init_setup::<Vulkan>(&device, RuntimeOptions::default())
            };
            let info = setup.adapter.get_info();
            println!("runtime {which}: {} ({:?}, {:?})", info.name, setup.backend, info.device_type);
            if std::env::args().nth(2).as_deref() == Some("extras") {
                extras(<WgpuRuntime>::client(&device));
            } else if std::env::args().nth(2).as_deref() == Some("step") {
                step(<WgpuRuntime>::client(&device), true);
            } else {
                bench(<WgpuRuntime>::client(&device));
            }
        }
        #[cfg(feature = "cpu")]
        "cpu" => {
            use cubecl::cpu::{CpuDevice, CpuRuntime};
            println!("runtime cpu ({} logical cores)", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));
            bench(CpuRuntime::client(&CpuDevice));
        }
        #[cfg(feature = "cpu")]
        "cpu-step" => {
            use cubecl::cpu::{CpuDevice, CpuRuntime};
            println!("runtime cpu ({} logical cores)", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));
            step(CpuRuntime::client(&CpuDevice), false);
        }
        #[cfg(feature = "cpu")]
        "cpu-sweep" => {
            use cubecl::cpu::{CpuDevice, CpuRuntime};
            sweep(CpuRuntime::client(&CpuDevice));
        }
        other => panic!("unknown runtime {other} (cpu needs --features cpu)"),
    }
    println!("total {:.1}s (includes JIT compile)", t0.elapsed().as_secs_f64());
}
