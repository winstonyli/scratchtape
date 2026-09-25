// Does scratchtape's GPU path need Normal CPU priority? (LONG_RUNS.md's
// GPU-feeder exception, 2026-09-24.) Another session found burn training
// ran >=19x slower at BelowNormal under CPU load: its GPU threads wait by
// sleep-polling, and each wake-up queued behind busy threads. gpu.rs instead
// blocks in device.poll(wait_indefinitely) and a channel recv, so it may not
// suffer. This measures it rather than assuming.
//
// The probe is gpu.rs::gpu_matmul round trips (upload, one dispatch,
// readback) at 64 (overhead-bound) and 512. The load is a child copy of
// this binary spinning `t` threads at BelowNormal, as LONG_RUNS.md launches
// CPU jobs. For t in {0, 12, 16} (16 logical cores; 16 saturates them), the probe process
// alternates its own priority between Normal and BelowNormal, ROUNDS times,
// and reports min / median / p95 ms per round trip.
// Backend: WGPU_BACKEND as gpu.rs (DX12 by default). Windows only.
//
// Result 2026-09-24, DX12. The eGPU was shared with another session's burn
// training, which causes the ~168 ms p95 in every row. Medians, ms, over
// 3 rounds:
//   load  probe Normal     probe BelowNormal
//   0     5.1-7.3 / 7.8-8.2   5.1-5.9 / 7.8-8.2    (64 / 512)
//   12    5.2-6.4 / 6.4-8.1   6.5-7.2 / 9.0-10.0
//   16    5.2-6.5 / 7.6-8.1   138-159 / 163-171
// So blocking in device.poll doesn't protect us. Once every core is busy
// with equal-priority threads, each wake-up waits its turn and a round trip
// takes ~25x longer; at Normal the probe preempts the spinners and is
// unaffected. At 12 threads it was fine in this run. An earlier run, not
// checked for other sessions' CPU load, starved at 12 in 2 of 3 rounds
// (median 141-151 ms), so the 4 spare cores are no guarantee. GPU-feeding
// processes should run at Normal (LONG_RUNS.md's exception).
use scratchtape::gpu::gpu_matmul;
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::nn::Rng;
use scratchtape::tensor::NdArray;
use std::os::windows::io::AsRawHandle;
use std::process::Command;
use std::time::{Duration, Instant};

const ROUNDS: usize = 3;

type Handle = *mut std::ffi::c_void;
#[link(name = "kernel32")]
unsafe extern "system" {
    fn SetPriorityClass(process: Handle, class: u32) -> i32;
    fn GetCurrentProcess() -> Handle;
}
const NORMAL: u32 = 0x20;
const BELOW_NORMAL: u32 = 0x4000;

fn set_priority(h: Handle, class: u32) {
    // Safety: `h` is a live process handle (ours or our child's).
    assert!(unsafe { SetPriorityClass(h, class) } != 0, "SetPriorityClass failed");
}

/// Child mode: spin `t` threads; exits after 300 s so it can't be orphaned.
fn spin(t: usize) {
    std::thread::spawn(|| {
        std::thread::sleep(Duration::from_secs(300));
        std::process::exit(0);
    });
    let threads: Vec<_> = (0..t)
        .map(|i| {
            std::thread::spawn(move || {
                let mut x = i as u64 | 1;
                loop {
                    x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
                }
            })
        })
        .collect();
    for th in threads {
        th.join().unwrap();
    }
}

fn rand(rng: &mut Rng, n: usize) -> NdArray {
    NdArray::new((0..n * n).map(|_| rng.next_f32() * 0.2 - 0.1).collect(), vec![n, n])
}

/// `samples` round trips; returns (min, median, p95) in ms.
fn probe(a: &NdArray, b: &NdArray, samples: usize) -> (f64, f64, f64) {
    let mut t: Vec<f64> = (0..samples)
        .map(|_| {
            let s = Instant::now();
            std::hint::black_box(gpu_matmul(a, b));
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    (t[0], t[samples / 2], t[samples * 95 / 100])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("spin") {
        return spin(args[2].parse().unwrap());
    }
    let _lease = gpu_lease::hold(Kind::Exclusive, "scratchtape gpu_cpu_priority_check", Duration::from_secs(15 * 60));
    let mut rng = Rng::new(7);
    let cases: Vec<_> = [(64usize, 200usize), (512, 50)].into_iter().map(|(n, s)| (n, s, rand(&mut rng, n), rand(&mut rng, n))).collect();
    for (_, _, a, b) in &cases {
        let err = gpu_matmul(a, b).data.iter().zip(&a.matmul(b).data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-3, "gpu_matmul wrong: {err}");
        for _ in 0..10 {
            gpu_matmul(a, b);
        }
    }
    // Safety: the pseudo-handle for this process; always valid.
    let me = unsafe { GetCurrentProcess() };
    let exe = std::env::current_exe().unwrap();

    println!("load threads | probe prio  | size | ms min / median / p95");
    for t in [0usize, 12, 16] {
        let mut child = (t > 0).then(|| Command::new(&exe).args(["spin", &t.to_string()]).spawn().unwrap());
        if let Some(c) = &child {
            set_priority(c.as_raw_handle(), BELOW_NORMAL);
            std::thread::sleep(Duration::from_secs(1));
        }
        for round in 0..ROUNDS {
            for (class, name) in [(NORMAL, "Normal"), (BELOW_NORMAL, "BelowNormal")] {
                set_priority(me, class);
                for (n, samples, a, b) in &cases {
                    let (lo, med, p95) = probe(a, b, *samples);
                    println!("{t:>12} | {name:<11} | {n:>4} | {lo:6.2} / {med:6.2} / {p95:7.2}   (round {round})");
                }
            }
        }
        set_priority(me, NORMAL);
        if let Some(c) = &mut child {
            c.kill().unwrap();
            c.wait().unwrap();
        }
    }
}
