//! First GPU backend work: a single hand-written WGSL matmul kernel, checked
//! against the trusted CPU implementation (same discipline as every new
//! backward rule this project has added being gradient-checked against
//! numerical differences before being trusted). Not wired into Tape/NdArray
//! yet - this module only proves the kernel is correct and measures it in
//! isolation, matching how new capabilities (k-means, graph extraction) were
//! built standalone before any integration.

use crate::tensor::NdArray;
use std::sync::OnceLock;

/// Device/queue/pipeline setup is a one-time cost (adapter negotiation,
/// shader compilation) - measured to dominate any per-call comparison if
/// repeated on every gpu_matmul call, the GPU-side analog of the CPU
/// persistent-thread-pool-vs-spawn-per-call tradeoff surveyed earlier.
/// Cached once via OnceLock, reused across calls; only per-call buffers and
/// the bind group (which references them) are created fresh each time.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

static CONTEXT: OnceLock<GpuContext> = OnceLock::new();

fn context() -> &'static GpuContext {
    CONTEXT.get_or_init(|| pollster::block_on(init_context()))
}

async fn init_context() -> GpuContext {
    // DX12 by default: on the RX 9060 XT it had far lower per-call overhead
    // than wgpu's default pick, Vulkan (128x128 matmul round trip 1.1 ms vs
    // 7.8 ms; 512: 6.6 vs 30 ms; roughly even at 1024), 2026-09-23, driver
    // 32.0.31041. That's this project's regime: many short, overhead-bound
    // dispatches. For long compute-bound kernels Vulkan wins instead
    // (humble-cortex: 1.4-1.6x), so re-measure if the workload grows. Off
    // Windows, or if DX12 is missing, fall back to any backend.
    // WGPU_BACKEND=vulkan|dx12|... overrides.
    let preferred = if cfg!(windows) { wgpu::Backends::DX12 } else { wgpu::Backends::all() };
    let backends = wgpu::Backends::from_env().unwrap_or(preferred);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    // HighPerformance, not default(): on a laptop-style iGPU + dGPU machine
    // default() picked the integrated Radeon 780M over the discrete RX 9060
    // XT - every GPU benchmark before this fix ran on the iGPU. The adapter
    // actually chosen is logged below so this can't go unnoticed again.
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
        .await;
    // Preferred backend unavailable: retry across all backends rather than fail.
    let adapter = match adapter {
        Ok(a) => a,
        Err(_) => wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle())
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .expect("gpu_matmul: no GPU adapter found"),
    };
    let info = adapter.get_info();
    eprintln!("gpu: using {} ({:?}, {:?})", info.name, info.device_type, info.backend);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("gpu_matmul: failed to get GPU device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul"),
        source: wgpu::ShaderSource::Wgsl(include_str!("matmul.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group_layout = pipeline.get_bind_group_layout(0);

    GpuContext { device, queue, pipeline, bind_group_layout }
}

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// GPU dims uniform must be std140-ish aligned - padded to 16 bytes (4 u32s)
/// even though only 3 are used, since a uniform buffer struct's size needs
/// to be a multiple of its largest member's alignment (4 bytes) rounded up
/// to WGSL's own alignment rules for the uniform address space.
fn dims_bytes(m: u32, k: u32, n: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&m.to_le_bytes());
    out[4..8].copy_from_slice(&k.to_le_bytes());
    out[8..12].copy_from_slice(&n.to_le_bytes());
    out
}

pub fn gpu_matmul(a: &NdArray, b: &NdArray) -> NdArray {
    assert_eq!(a.shape.len(), 2, "gpu_matmul lhs must be 2D, got {:?}", a.shape);
    assert_eq!(b.shape.len(), 2, "gpu_matmul rhs must be 2D, got {:?}", b.shape);
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "gpu_matmul inner dims must match: {:?} vs {:?}", a.shape, b.shape);

    let ctx = context();
    let device = &ctx.device;
    let queue = &ctx.queue;

    let a_bytes = f32_slice_to_bytes(&a.data);
    let b_bytes = f32_slice_to_bytes(&b.data);
    let out_size = (m * n * 4) as u64;

    use wgpu::util::DeviceExt;
    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a"),
        contents: &a_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b"),
        contents: &b_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let dims_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: &dims_bytes(m as u32, k as u32, n as u32),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_bind_group"),
        layout: &ctx.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dims_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&ctx.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(8), (m as u32).div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("gpu_matmul: map_async channel closed");
    });
    device.poll(wgpu::PollType::wait_indefinitely()).expect("gpu_matmul: device poll failed");
    rx.recv().expect("gpu_matmul: map_async never responded").expect("gpu_matmul: buffer map failed");

    let mapped = slice.get_mapped_range().expect("gpu_matmul: mapped range unavailable");
    let out_data = bytes_to_f32_vec(&mapped);
    drop(mapped);
    staging_buf.unmap();

    NdArray::new(out_data, vec![m, n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_matmul_matches_cpu() {
        let a = NdArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = NdArray::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let expected = a.matmul(&b);
        let actual = gpu_matmul(&a, &b);
        assert_eq!(actual.shape, expected.shape);
        for (x, y) in actual.data.iter().zip(expected.data.iter()) {
            assert!((x - y).abs() < 1e-4, "gpu {x} vs cpu {y}");
        }
    }

    #[test]
    fn gpu_matmul_matches_cpu_larger() {
        let mut rng = crate::nn::Rng::new(42);
        let (m, k, n) = (37, 53, 29);
        let a = NdArray::new((0..m * k).map(|_| rng.next_f32() * 2.0 - 1.0).collect(), vec![m, k]);
        let b = NdArray::new((0..k * n).map(|_| rng.next_f32() * 2.0 - 1.0).collect(), vec![k, n]);
        let expected = a.matmul(&b);
        let actual = gpu_matmul(&a, &b);
        assert_eq!(actual.shape, expected.shape);
        for (x, y) in actual.data.iter().zip(expected.data.iter()) {
            assert!((x - y).abs() < 1e-3, "gpu {x} vs cpu {y}");
        }
    }
}
