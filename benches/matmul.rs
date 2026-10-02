use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use scratchtape::tensor::NdArray;

/// Baseline for the current naive i-k-j triple-loop matmul, to compare
/// against once a blocked/tiled or SIMD version exists - there's nothing to
/// compare against yet, so this is the first bench, not a retrofit onto
/// something already optimized.
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &n in &[32usize, 64, 128, 256] {
        let a = NdArray::new(vec![0.5f32; n * n], vec![n, n]);
        let b = NdArray::new(vec![0.3f32; n * n], vec![n, n]);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| a.matmul(&b));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
