use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use scratchtape::gpu::gpu_matmul;
use scratchtape::tensor::NdArray;

/// First real GPU-vs-CPU comparison point - CPU side already has
/// cache-blocking + AVX2/FMA behind it, GPU side is a naive (unblocked,
/// unshared-memory) WGSL kernel, checked correct in gpu.rs's own tests
/// before ever being benched. Sizes go beyond the CPU bench's 256 cap
/// specifically to see whether GPU's fixed per-call dispatch/readback
/// overhead is amortized away at larger sizes, where the actual GPU-vs-CPU
/// decision (re-profiled at model-scale-up) said matmul dominance was
/// heading.
fn bench_matmul_cpu_vs_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_cpu_vs_gpu");
    for &n in &[128usize, 256, 512, 1024] {
        let a = NdArray::new(vec![0.5f32; n * n], vec![n, n]);
        let b = NdArray::new(vec![0.3f32; n * n], vec![n, n]);

        group.bench_with_input(BenchmarkId::new("cpu", n), &n, |bencher, _| {
            bencher.iter(|| a.matmul(&b));
        });
        group.bench_with_input(BenchmarkId::new("gpu", n), &n, |bencher, _| {
            bencher.iter(|| gpu_matmul(&a, &b));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul_cpu_vs_gpu);
criterion_main!(benches);
