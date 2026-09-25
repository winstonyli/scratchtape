//! split_heads / merge_heads (milestone 2): the permutations between the
//! residual stream's [B·T, H·w] columns and batched attention's
//! [B·H·T, w] rows, as `Tape::split_heads` / `merge_heads`. Each is the
//! other's backward. One unit per element.
use super::{EW_DIM, buf, client};
use cubecl::prelude::*;
use cubecl::server::Handle;

fn whole(h: &Handle) -> BufferArg {
    buf(h, h.size_in_used() as usize / 4)
}

/// Columns col0..col0 + H·w of src [B·T, cols] as [B·H·T, w]: row
/// (b·H + h)·T + t holds src row b·T + t, columns col0 + h·w ...
pub fn split_heads(src: &Handle, cols: usize, col0: usize, n_heads: usize, width: usize, batch: usize, t: usize) -> Handle {
    let len = batch * n_heads * t * width;
    let out = client().empty(len * 4);
    super::count_launch();
    k_split::launch(client(), CubeCount::Static((len as u32).div_ceil(EW_DIM), 1, 1), CubeDim::new_1d(EW_DIM), whole(src), whole(&out), len as u32, cols as u32, col0 as u32, n_heads as u32, width as u32, t as u32);
    out
}

/// The inverse: src [B·H·T, w] into columns col0..col0 + H·w of dst
/// [B·T, cols], written or accumulated. With col0 = 0 and cols = H·w it's
/// `Tape::merge_heads`; with an offset it's split_heads' backward into a
/// wider gradient (dQKV).
#[allow(clippy::too_many_arguments)]
pub fn merge_heads(src: &Handle, dst: &Handle, cols: usize, col0: usize, n_heads: usize, width: usize, batch: usize, t: usize, accumulate: bool) {
    let len = batch * n_heads * t * width;
    super::count_launch();
    k_merge::launch(client(), CubeCount::Static((len as u32).div_ceil(EW_DIM), 1, 1), CubeDim::new_1d(EW_DIM), whole(src), whole(dst), len as u32, cols as u32, col0 as u32, n_heads as u32, width as u32, t as u32, accumulate);
}

/// Where element i of the [B·H·T, w] side lives on the [B·T, cols] side.
#[cube]
fn stream_index(i: u32, cols: u32, col0: u32, n_heads: u32, width: u32, t: u32) -> u32 {
    let j = i % width;
    let row = i / width;
    let tt = row % t;
    let bh = row / t;
    let h = bh % n_heads;
    let b = bh / n_heads;
    (b * t + tt) * cols + col0 + h * width + j
}

#[cube(launch)]
fn k_split(src: &[f32], out: &mut [f32], len: u32, cols: u32, col0: u32, n_heads: u32, width: u32, t: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        out[i as usize] = src[stream_index(i, cols, col0, n_heads, width, t) as usize];
    }
}

#[cube(launch)]
fn k_merge(src: &[f32], dst: &mut [f32], len: u32, cols: u32, col0: u32, n_heads: u32, width: u32, t: u32, #[comptime] accumulate: bool) {
    let i = ABSOLUTE_POS as u32;
    if i < len {
        let o = stream_index(i, cols, col0, n_heads, width, t) as usize;
        let mut v = src[i as usize];
        if accumulate {
            v += dst[o];
        }
        dst[o] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_step::{read, upload};
    use crate::nn::Rng;
    use crate::tape::Tape;
    use crate::tensor::NdArray;

    /// Needs the discrete GPU. Splitting Q, K and V out of a fused
    /// [B·T, 3D] must equal `Tape::split_heads` exactly (it only moves
    /// values); merging must equal `Tape::merge_heads`, and merging all
    /// three back into a [B·T, 3D] gradient at their column offsets must
    /// rebuild the original, accumulated on top of what was there.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn split_and_merge_match_tape() {
        let (batch, n_heads, width, t) = (2, 3, 4, 5);
        let d = n_heads * width;
        let mut rng = Rng::new(4);
        let x: Vec<f32> = (0..batch * t * 3 * d).map(|_| rng.next_gaussian()).collect();
        let mut tape = Tape::new();
        let xv = tape.leaf(NdArray::new(x.clone(), vec![batch * t, 3 * d]));
        let xh = upload(&x);
        let back = upload(&x);
        for part in 0..3 {
            let want = tape.split_heads(xv, part * d..(part + 1) * d, n_heads, batch);
            let got = split_heads(&xh, 3 * d, part * d, n_heads, width, batch, t);
            assert_eq!(read(&got), tape.value(want).data, "split part {part}");
            let merged = tape.merge_heads(want, n_heads, batch);
            let out = upload(&vec![0.0f32; batch * t * d]);
            merge_heads(&got, &out, d, 0, n_heads, width, batch, t, false);
            assert_eq!(read(&out), tape.value(merged).data, "merge part {part}");
            merge_heads(&got, &back, 3 * d, part * d, n_heads, width, batch, t, true);
        }
        let twice: Vec<f32> = x.iter().map(|v| 2.0 * v).collect();
        assert_eq!(read(&back), twice, "merge into dQKV columns, accumulated");
    }
}
