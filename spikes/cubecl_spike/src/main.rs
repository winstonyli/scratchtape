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
use cubecl::prelude::*;
use cubecl::server::Handle;
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
fn sweep<R: Runtime>(client: ComputeClient<R>) {
    let sync = |c: &ComputeClient<R>| cubecl::future::block_on(c.sync()).unwrap();
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
                k_empty::launch::<R>(&client, CubeCount::Static(cubes, 1, 1), CubeDim::new_1d(d), buf(&o_h, len));
            }
            sync(&client);
        }) / n as f64;
        let ew = best(&mut || {
            for _ in 0..n {
                k_elementwise::launch::<R>(&client, CubeCount::Static(cubes, 1, 1), CubeDim::new_1d(d), buf(&a_h, len), buf(&o_h, len), len as u32);
            }
            sync(&client);
        }) / n as f64;
        let mm_n = 20;
        let mm = best(&mut || {
            for _ in 0..mm_n {
                k_matmul_1d::launch::<R>(&client, CubeCount::Static(cubes, 1, 1), CubeDim::new_1d(d), buf(&a_h, m * k), buf(&b_h, k * k), buf(&o_h, len), m as u32, k as u32, k as u32);
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

fn buf<R: Runtime>(h: &Handle, len: usize) -> BufferArg<R> {
    unsafe { BufferArg::from_raw_parts(h.clone(), len) }
}

fn rand(rng: &mut Rng, rows: usize, cols: usize) -> NdArray {
    NdArray::new((0..rows * cols).map(|_| rng.next_f32() * 0.2 - 0.1).collect(), vec![rows, cols])
}

fn best(f: &mut dyn FnMut()) -> f64 {
    (0..REPS).map(|_| { let t = Instant::now(); f(); t.elapsed().as_secs_f64() }).fold(f64::MAX, f64::min)
}

fn bench<R: Runtime>(client: ComputeClient<R>) {
    let sync = |c: &ComputeClient<R>| cubecl::future::block_on(c.sync()).unwrap();
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
            k_matmul::launch::<R>(&client, count.clone(), dim, buf(&ping[i % 2], m * k), buf(&b_h, k * k), buf(&ping[1 - i % 2], m * k), m as u32, k as u32, k as u32);
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
                k_elementwise::launch::<R>(&client, CubeCount::Static((len as u32).div_ceil(256), 1, 1), CubeDim::new_1d(256), buf(&ping[i % 2], len), buf(&ping[1 - i % 2], len), len as u32);
            }
            sync(&client);
        }) / N as f64;
        let cpu_calls = if m * k * k > 10_000_000 { 5 } else { 50 };
        let cpu = best(&mut || for _ in 0..cpu_calls { std::hint::black_box(a.matmul(&b)); }) / cpu_calls as f64;
        let flops = 2.0 * (m * k * k) as f64;
        println!("({m:>4},{k:>3})@({k},{k}) | {:>9.4} | {:>7.1} | {:>9.4} | {:>8.3}", queued * 1e3, flops / queued / 1e9, ew * 1e3, cpu * 1e3);
    }
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
            bench::<WgpuRuntime>(WgpuRuntime::client(&device));
        }
        #[cfg(feature = "cpu")]
        "cpu" => {
            use cubecl::cpu::{CpuDevice, CpuRuntime};
            println!("runtime cpu ({} logical cores)", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));
            bench::<CpuRuntime>(CpuRuntime::client(&CpuDevice));
        }
        #[cfg(feature = "cpu")]
        "cpu-sweep" => {
            use cubecl::cpu::{CpuDevice, CpuRuntime};
            sweep::<CpuRuntime>(CpuRuntime::client(&CpuDevice));
        }
        other => panic!("unknown runtime {other} (cpu needs --features cpu)"),
    }
    println!("total {:.1}s (includes JIT compile)", t0.elapsed().as_secs_f64());
}
