use criterion::{criterion_group, criterion_main, Criterion};
use engine::nn::{Rng, TransformerBlock};
use engine::tape::Tape;
use engine::tensor::NdArray;

/// Baseline for a realistic COMPOSITE workload, not an isolated primitive
/// like benches/matmul.rs - one forward+backward pass through a full
/// transformer block at tiny_lm.rs's actual scale, matching exactly what
/// one real training step does (fresh Tape, leaf input, forward, backward).
/// Established specifically because the first real samply profile of a
/// training run showed ~68-71% of wall-clock time inside the heap
/// allocator, not in any numerical code - this is the "before" number any
/// allocation-reduction change gets measured against, not a number checked
/// in isolation with nothing to compare it to.
fn bench_transformer_block_forward_backward(c: &mut Criterion) {
    let (d_model, n_heads, d_ff, seq_len) = (32usize, 4usize, 64usize, 16usize);
    let mut rng = Rng::new(1);
    let block = TransformerBlock::new(&mut rng, d_model, n_heads, d_ff);
    let x_data: Vec<f32> = (0..seq_len * d_model).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();

    c.bench_function("transformer_block_forward_backward", |b| {
        b.iter(|| {
            let mut tape = Tape::new();
            let x = tape.leaf(NdArray::new(x_data.clone(), vec![seq_len, d_model]));
            let out = block.forward(&mut tape, x);
            let loss = tape.sum(out.y);
            tape.backward(loss);
        });
    });
}

criterion_group!(benches, bench_transformer_block_forward_backward);
criterion_main!(benches);
