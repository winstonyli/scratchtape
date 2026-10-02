// Does Windows' per-process GPU priority protect a small GPU job from a
// heavy one on the shared eGPU? (LONG_RUNS.md, 2026-09-24.)
//
// The probe is gpu.rs::gpu_matmul at 64x64: one small submit plus
// readback, the kind of work that stalled behind other sessions' jobs.
// The load is a child copy of this binary. It loops submits of `n`
// chained (2048,128)@(128,128) matmul dispatches (~0.075 ms each on DX12),
// so n sets how long each submit holds the GPU. For n in {20, 500} and the
// load's GPU priority in {Normal, Idle}, set from outside through
// D3DKMTSetProcessSchedulingPriorityClass, time SAMPLES probe round trips
// (min / median / p95) and the load's own throughput.
// Backend: WGPU_BACKEND as gpu.rs (DX12 by default). Windows only.
//
// Result 2026-09-24, DX12. The eGPU was shared with another session's job
// (conv_pc2_isolated_16x16_anneal_check, GPU priority Normal). Two runs,
// probe ms min / median / p95:
//   no load               0.22-0.25 / 0.27-0.34 / 1.0-1.6 (0.5 / 640 in a
//                         third run)
//   n=20,  load Normal    0.5-0.7 / 2.2-6.6 / 675-929
//   n=20,  load Idle      0.4-1.2 / 2.1-2.2 / 642
//   n=500, load Normal    0.3 / 3.3-3.5 / 509-644
//   n=500, load Idle      0.2-0.3 / 3.2-3.5 / 510-644
// Setting the load's GPU priority to Idle made no measurable difference.
// The p95 sits at a recurring ~510 or ~640 ms stall whatever the priority
// or submit size. The load managed only 58-417 dispatches/s, against
// ~13,000/s alone, so the stalls look like the third job's long,
// unpreempted packets. Our two processes' relative priority can't fix that.
// Re-run on an idle eGPU before judging the priority class itself.
use scratchtape::gpu::gpu_matmul;
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::nn::Rng;
use scratchtape::tensor::NdArray;
use std::io::{BufRead, BufReader};
use std::os::windows::io::AsRawHandle;
use std::process::{Command, Stdio};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const SAMPLES: usize = 200;

#[link(name = "gdi32")]
unsafe extern "system" {
    fn D3DKMTSetProcessSchedulingPriorityClass(process: *mut std::ffi::c_void, class: i32) -> i32;
    fn D3DKMTGetProcessSchedulingPriorityClass(process: *mut std::ffi::c_void, class: *mut i32) -> i32;
}
const IDLE: i32 = 0;
const NORMAL: i32 = 2;

fn rand(rng: &mut Rng, rows: usize, cols: usize) -> NdArray {
    NdArray::new((0..rows * cols).map(|_| rng.next_f32() * 0.2 - 0.1).collect(), vec![rows, cols])
}

fn bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Child mode: submit `n` chained dispatches per submit, forever, printing
/// the cumulative dispatch count every 100 ms.
fn load(n: usize) {
    let preferred = wgpu::Backends::DX12;
    let backends = wgpu::Backends::from_env().unwrap_or(preferred);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).unwrap();
    let info = adapter.get_info();
    eprintln!("load: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(include_str!("../../src/matmul.wgsl").into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let (m, k) = (2048usize, 128usize);
    let mut rng = Rng::new(5);
    let storage = wgpu::BufferUsages::STORAGE;
    let mk = |data: &[f32], usage| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &bytes(data), usage });
    let ping = [mk(&rand(&mut rng, m, k).data, storage), mk(&vec![0.0; m * k], storage)];
    let b = mk(&rand(&mut rng, k, k).data, storage);
    let mut dims = [0u8; 16];
    for (i, v) in [m as u32, k as u32, k as u32].iter().enumerate() {
        dims[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let dims = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &dims, usage: wgpu::BufferUsages::UNIFORM });
    let layout = pipeline.get_bind_group_layout(0);
    let bind: Vec<_> = (0..2)
        .map(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ping[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: b.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ping[1 - i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: dims.as_entire_binding() },
                ],
            })
        })
        .collect();
    let (mut done, mut last) = (0u64, Instant::now());
    loop {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for d in 0..n {
                pass.set_bind_group(0, &bind[d % 2], &[]);
                pass.dispatch_workgroups((k as u32).div_ceil(8), (m as u32).div_ceil(8), 1);
            }
        }
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        done += n as u64;
        if last.elapsed() >= Duration::from_millis(100) {
            println!("{done}");
            last = Instant::now();
        }
    }
}

/// SAMPLES probe round trips; returns (min, median, p95) in ms.
fn probe(a: &NdArray, b: &NdArray) -> (f64, f64, f64) {
    let mut t: Vec<f64> = (0..SAMPLES)
        .map(|_| {
            let s = Instant::now();
            std::hint::black_box(gpu_matmul(a, b));
            s.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    (t[0], t[SAMPLES / 2], t[SAMPLES * 95 / 100])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("load") {
        return load(args[2].parse().unwrap());
    }
    let _lease = gpu_lease::hold(Kind::Exclusive, "scratchtape gpu_priority_check", Duration::from_secs(15 * 60));
    let mut rng = Rng::new(7);
    let (a, b) = (rand(&mut rng, 64, 64), rand(&mut rng, 64, 64));
    let want = a.matmul(&b);
    let err = gpu_matmul(&a, &b).data.iter().zip(&want.data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
    assert!(err < 1e-4, "probe matmul wrong: {err}");
    for _ in 0..20 {
        gpu_matmul(&a, &b);
    }

    println!("load          | load GPU prio | probe ms min / median / p95 | load dispatches/s");
    let (lo, med, p95) = probe(&a, &b);
    println!("none          | -             | {lo:6.2} / {med:6.2} / {p95:7.2} | -");
    let exe = std::env::current_exe().unwrap();
    for n in [20usize, 500] {
        for (prio, name) in [(NORMAL, "Normal"), (IDLE, "Idle")] {
            let mut child = Command::new(&exe).args(["load", &n.to_string()]).stdout(Stdio::piped()).spawn().unwrap();
            let count = Arc::new(AtomicU64::new(0));
            let c = count.clone();
            let out = child.stdout.take().unwrap();
            std::thread::spawn(move || {
                for line in BufReader::new(out).lines().map_while(Result::ok) {
                    c.store(line.trim().parse().unwrap_or(0), Ordering::Relaxed);
                }
            });
            // Wait until the load is actually running, then 1 s of warm-up.
            let start = Instant::now();
            while count.load(Ordering::Relaxed) == 0 && start.elapsed() < Duration::from_secs(60) {
                std::thread::sleep(Duration::from_millis(50));
            }
            // Only now: a process with no GPU device yet is refused with
            // STATUS_INVALID_PARAMETER (0xc000000d).
            let h = child.as_raw_handle();
            let mut got = -1;
            // Safety: `h` is the live child's process handle.
            let status = unsafe { D3DKMTSetProcessSchedulingPriorityClass(h, prio) | D3DKMTGetProcessSchedulingPriorityClass(h, &mut got) };
            assert!(status == 0 && got == prio, "GPU priority not applied: status {status:#x}, class {got}");
            std::thread::sleep(Duration::from_secs(1));
            let (c0, t0) = (count.load(Ordering::Relaxed), Instant::now());
            let (lo, med, p95) = probe(&a, &b);
            let rate = (count.load(Ordering::Relaxed) - c0) as f64 / t0.elapsed().as_secs_f64();
            child.kill().unwrap();
            child.wait().unwrap();
            println!("{n:>3}/submit    | {name:<13} | {lo:6.2} / {med:6.2} / {p95:7.2} | {rate:8.0}");
        }
    }
}
