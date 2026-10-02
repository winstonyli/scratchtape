// Would a device-resident GPU training step beat a multithreaded CPU one?
// That turns on one unmeasured number: the cost of a dispatch that is
// queued behind others in one submit, with no readback, rather than
// gpu.rs::gpu_matmul's per-call cost (fresh buffers + upload + submit +
// map/readback, 0.3-0.5 ms on DX12). The RX 9060 XT is a USB4 eGPU, so
// every sync crosses the tunnel; a resident design syncs once per step.
//
// For each square-ish shape (m x k) @ (k x k), chained ping-pong so each
// dispatch depends on the previous one (as a forward pass's ops do), time:
//   roundtrip: gpu_matmul per call (today's design)
//   queued:    N dispatches recorded into one compute pass, one submit,
//              one wait, buffers resident; reported per dispatch
//   cpu:       NdArray::matmul (single-threaded) for the same shape
// Best-of-REPS wall time: the machine is shared (see LONG_RUNS.md).
// Backend: WGPU_BACKEND=dx12|vulkan (default dx12 on Windows, as gpu.rs).
//
// Result 2026-09-24 (measured on wgpu 23; ported to wgpu 30 since, not yet
// re-timed), RX 9060 XT eGPU idle (2-3% util), CPU shared with a
// 17-core job, naive 8x8 WGSL kernel (ms):
//   shape             roundtrip dx12/vk   queued dx12/vk     cpu (1 thread)
//   (64,64)@(64,64)     0.43 / 2.51       0.011 / 0.009       0.08
//   (512,128)@(128,128) 1.18 / 2.49       0.022 / 0.031       1.6
//   (512,256)@(256,256) 1.85 / 3.03       0.079 / 0.141       4.6
//   (2048,128)@(128,128)2.44 / 4.01       0.075 / 0.089       6.4
// Queued dispatches cost ~10 us, 40-100x less than a round trip, and the
// naive kernel already reaches 750-900 GFLOP/s (DX12) at training shapes.
// DX12 >= Vulkan at every shape here, queued too. The per-op round trip,
// not the kernel or the USB4 link's bandwidth, is what lost the old branch.
use scratchtape::gpu::gpu_matmul;
use scratchtape::gpu_lease::{self, Kind};
use scratchtape::nn::Rng;
use scratchtape::tensor::NdArray;
use std::time::Instant;
use wgpu::util::DeviceExt;

const N: usize = 500;
const REPS: usize = 5;

fn rand(rng: &mut Rng, rows: usize, cols: usize) -> NdArray {
    NdArray::new((0..rows * cols).map(|_| rng.next_f32() * 0.2 - 0.1).collect(), vec![rows, cols])
}

fn bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn main() {
    let _lease = gpu_lease::hold(Kind::Exclusive, "scratchtape gpu_dispatch_overhead", std::time::Duration::from_secs(15 * 60));
    let preferred = if cfg!(windows) { wgpu::Backends::DX12 } else { wgpu::Backends::all() };
    let backends = wgpu::Backends::from_env().unwrap_or(preferred);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }))
        .expect("no adapter for requested backend");
    let info = adapter.get_info();
    println!("device: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
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
    let layout = pipeline.get_bind_group_layout(0);

    println!("shape (m,k)@(k,k) | roundtrip ms/call | queued ms/dispatch (N={N}) | cpu ms/call | queued GFLOP/s");
    let mut rng = Rng::new(7);
    for &(m, k) in &[(64, 64), (512, 128), (512, 256), (2048, 128)] {
        let a = rand(&mut rng, m, k);
        let b = rand(&mut rng, k, k);
        let flops = 2.0 * (m * k * k) as f64;

        // Correctness of the queued path's first link: out0 = a @ b.
        let mk_buf = |data: &[f32], usage| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &bytes(data), usage });
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let ping = [mk_buf(&a.data, storage), mk_buf(&vec![0.0; m * k], storage)];
        let b_buf = mk_buf(&b.data, wgpu::BufferUsages::STORAGE);
        let mut dims = [0u8; 16];
        for (i, v) in [m as u32, k as u32, k as u32].iter().enumerate() {
            dims[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &dims, usage: wgpu::BufferUsages::UNIFORM });
        // bind[i] reads ping[i], writes ping[1-i].
        let bind: Vec<_> = (0..2)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: ping[i].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: ping[1 - i].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
                    ],
                })
            })
            .collect();
        let run_queued = |n: usize| {
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
        };

        // Check: one queued dispatch reproduces the CPU product.
        run_queued(1);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (m * k * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&ping[1], 0, &staging, 0, (m * k * 4) as u64);
        queue.submit(Some(enc.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let got: Vec<f32> = staging.slice(..).get_mapped_range().unwrap().as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        let want = a.matmul(&b);
        let err = got.iter().zip(&want.data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-3, "queued dispatch disagrees with CPU: max err {err}");

        let best = |f: &mut dyn FnMut()| {
            (0..REPS)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_secs_f64()
                })
                .fold(f64::MAX, f64::min)
        };
        let rt_calls = 20;
        let roundtrip = best(&mut || {
            for _ in 0..rt_calls {
                std::hint::black_box(gpu_matmul(&a, &b));
            }
        }) / rt_calls as f64;
        let queued = best(&mut || run_queued(N)) / N as f64;
        let cpu_calls = if m * k * k > 10_000_000 { 5 } else { 50 };
        let cpu = best(&mut || {
            for _ in 0..cpu_calls {
                std::hint::black_box(a.matmul(&b));
            }
        }) / cpu_calls as f64;
        println!("({m:>4},{k:>3})@({k},{k}) | {:>8.3} | {:>8.4} | {:>8.3} | {:>7.1}", roundtrip * 1e3, queued * 1e3, cpu * 1e3, flops / queued / 1e9);
    }
}
