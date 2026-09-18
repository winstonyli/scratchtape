// Naive GPU matmul kernel - correctness first, mirrors the CPU side's own
// history (naive i-k-j loop before cache-blocking/AVX2 were added). One
// thread computes one output element via a straight dot product; no
// workgroup-shared-memory tiling yet (the GPU analog of CPU cache blocking) -
// that's a measured follow-up, not a speculative one, same discipline used
// throughout this project's CPU optimization pass.

struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    var sum: f32 = 0.0;
    for (var p: u32 = 0u; p < dims.k; p = p + 1u) {
        sum = sum + a[row * dims.k + p] * b[p * dims.n + col];
    }
    out_buf[row * dims.n + col] = sum;
}
