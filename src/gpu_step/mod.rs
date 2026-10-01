//! Device-resident training step on the discrete eGPU (cubecl, Vulkan
//! SPIR-V). Design and milestones: docs/gpu_step_design.md. Milestone 1:
//! device selection, the flat parameter/gradient buffers, SGD. Milestone 2:
//! the kernels, one module each, each with a GPU parity test.
use crate::nn::{Embedding, LayerNorm, Linear, TransformerBlock};
use cubecl::client::Client;
use cubecl::prelude::*;
pub use cubecl::server::Handle;
use cubecl::wgpu::{Dx12, RuntimeOptions, Vulkan, WgpuDevice, WgpuDeviceKind, WgpuRuntime, init_setup};
use cubecl_runtime::runtime::Runtime;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod matmul;
mod profile;
pub mod rows;
pub mod tape;
pub mod tokens;

pub use profile::{host_profile_cut, host_profile_start, host_profile_take, profile_start, profile_take};

/// Units per cube for 1-D elementwise kernels.
const EW_DIM: u32 = 256;

/// Cubes for a 1-D elementwise kernel over `units` elements (EW_DIM units each).
fn cubes(units: usize) -> CubeCount {
    CubeCount::Static((units as u32).div_ceil(EW_DIM), 1, 1)
}

static CLIENT: OnceLock<Client> = OnceLock::new();

/// Launches per queue submission (cubecl's default is 32). Each submit
/// costs ~60 µs of host time, and GPU-side per submission on Vulkan;
/// pipelined steps took 2.35 ms at 32, 1.97 at 128, 1.86 at 512. 128
/// keeps the GPU fed early when a step ends in a readback.
const SUBMIT_TASKS: usize = 128;

/// Kernel launches queued so far, for counting launches per step.
pub static LAUNCHES: AtomicUsize = AtomicUsize::new(0);

#[track_caller]
fn count_launch() {
    count_launch_as(String::new);
}

/// `count_launch` with a tag appended to the profiler's site key (say, a
/// matmul's shape); `tag` only runs while profiling.
#[track_caller]
fn count_launch_as(tag: impl FnOnce() -> String) {
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
    if profile::active() {
        profile::mark_launch(std::panic::Location::caller(), tag());
    }
}

/// The first discrete GPU, on Vulkan, or DX12 with `WGPU_BACKEND=dx12` (as
/// `gpu.rs`). DX12 needs DXC (`dxcompiler.dll` on PATH, e.g. the Windows
/// SDK's `bin/<ver>/x64`): without it wgpu silently falls back to FXC,
/// whose shaders ran the step ~170× slower, so that panics here. cubecl
/// submits every `SUBMIT_TASKS` launches unless `CUBECL_WGPU_MAX_TASKS`
/// says otherwise. Never falls back to the iGPU or CPU:
/// cubecl panics with "No Discrete GPU device found" if there's none, and
/// its `CUBECL_WGPU_DEFAULT_DEVICE` override only applies to the default
/// device kind, not this one. Logs the adapter once. Cached: cubecl
/// registers one client per device.
pub fn client() -> &'static Client {
    CLIENT.get_or_init(|| {
        let device = WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0));
        let dx12 = std::env::var("WGPU_BACKEND").is_ok_and(|b| b.eq_ignore_ascii_case("dx12"));
        let mut options = RuntimeOptions::default();
        if std::env::var_os("CUBECL_WGPU_MAX_TASKS").is_none() {
            options.tasks_max = SUBMIT_TASKS;
        }
        let setup = if dx12 {
            let on_path = std::env::var_os("PATH").is_some_and(|p| std::env::split_paths(&p).any(|d| d.join("dxcompiler.dll").is_file()));
            assert!(on_path, "WGPU_BACKEND=dx12 needs dxcompiler.dll on PATH (Windows SDK bin/<ver>/x64); without it wgpu falls back to FXC");
            init_setup::<Dx12>(&device, options)
        } else {
            init_setup::<Vulkan>(&device, options)
        };
        let info = setup.adapter.get_info();
        eprintln!("gpu_step: {} ({:?}, {:?}, driver {})", info.name, info.device_type, setup.backend, info.driver_info);
        <WgpuRuntime>::client(&device)
    })
}

/// Every trainable parameter of a plain tiny_lm, flattened in this order:
/// token embedding, position embedding, each block's `to_flat`, final
/// LayerNorm, output projection. Gradients use the same offsets.
pub fn pack(token_emb: &Embedding, pos_emb: &Embedding, blocks: &[TransformerBlock], final_ln: &LayerNorm, output_proj: &Linear) -> Vec<f32> {
    let mut out = token_emb.to_flat();
    out.extend(pos_emb.to_flat());
    for b in blocks {
        out.extend(b.to_flat());
    }
    out.extend(final_ln.to_flat());
    out.extend(output_proj.to_flat());
    out
}

/// K same-shape models trained in the same launches (horizontal fusion,
/// HFTA): model m's parameters and gradients sit at m·stride in the flat
/// buffers, and its rows are the m-th of K equal slices of every
/// activation. Kernels that read parameters by row, or reduce rows into a
/// parameter gradient, take one; the rest see K models' rows as one batch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Models {
    pub k: usize,
    pub stride: usize,
}

impl Models {
    pub const ONE: Self = Self { k: 1, stride: 0 };
}

/// Parameters and their gradients, resident on the device as two flat
/// buffers with matching offsets.
pub struct DeviceParams {
    pub params: Handle,
    pub grads: Handle,
    pub len: usize,
    pub models: Models,
}

impl DeviceParams {
    /// Uploads `flat` (from `pack`); gradients start at zero.
    pub fn upload(flat: &[f32]) -> Self {
        Self::upload_models(flat, 1)
    }

    /// Uploads K models' `pack`s, concatenated, for fused training (see
    /// `Models`); gradients start at zero.
    pub fn upload_models(flat: &[f32], k: usize) -> Self {
        assert!(k >= 1 && flat.len().is_multiple_of(k), "{} parameters don't split into {k} models", flat.len());
        let c = client();
        let params = c.create_from_slice(f32::as_bytes(flat));
        let grads = c.create_from_slice(f32::as_bytes(&vec![0.0f32; flat.len()]));
        Self { params, grads, len: flat.len(), models: Models { k, stride: flat.len() / k } }
    }

    /// One launch: p -= lr * g over every parameter.
    pub fn sgd(&self, lr: f32) {
        count_launch();
        k_sgd::launch(client(), cubes(self.len), CubeDim::new_1d(EW_DIM), buf(&self.params, self.len), buf(&self.grads, self.len), lr, self.len as u32);
    }

    /// One launch of heavy-ball momentum: v = mu * v + g, then p -= lr * v.
    /// `v` holds the velocity (`len` f32s, zero at the start of training).
    pub fn momentum(&self, lr: f32, mu: f32, v: &Handle) {
        count_launch();
        k_momentum::launch(client(), cubes(self.len), CubeDim::new_1d(EW_DIM), buf(&self.params, self.len), buf(&self.grads, self.len), buf(v, self.len), lr, mu, self.len as u32);
    }

    /// One launch: p *= 1 - shrink * mask over every parameter (mask is
    /// 0 or 1 per parameter, e.g. `Config::decay_mask`). Weight decay at
    /// rate wd is `decay(lr * wd, mask)` before `sgd(lr)`.
    pub fn decay(&self, shrink: f32, mask: &Handle) {
        count_launch();
        k_decay::launch(client(), cubes(self.len), CubeDim::new_1d(EW_DIM), buf(&self.params, self.len), buf(mask, self.len), shrink, self.len as u32);
    }

    /// One launch: g = 0.
    pub fn zero_grads(&self) {
        count_launch();
        k_fill::launch(client(), cubes(self.len), CubeDim::new_1d(EW_DIM), buf(&self.grads, self.len), 0.0f32, self.len as u32);
    }

    /// One launch: every model's parameters become the mean over the K
    /// models (local SGD's sync; momentum buffers are left alone).
    pub fn average_models(&self) {
        let Models { k, stride } = self.models;
        count_launch();
        k_average_models::launch(client(), cubes(stride), CubeDim::new_1d(EW_DIM), buf(&self.params, self.len), stride as u32, k as u32);
    }

    /// Blocking readback (crosses USB4; for tests and checkpoints only).
    pub fn read(&self, h: &Handle) -> Vec<f32> {
        f32::from_bytes(&client().read_one(h.clone()).unwrap())[..self.len].to_vec()
    }

}

/// The first f32 of a buffer, blocking: a training step's one readback
/// (the loss).
pub fn read_f32(h: &Handle) -> f32 {
    f32::from_bytes(&client().read_one(h.clone()).unwrap())[0]
}

/// A new device buffer holding `v` (crosses USB4; not per step).
pub fn upload_f32(v: &[f32]) -> Handle {
    client().create_from_slice(f32::as_bytes(v))
}

fn buf(h: &Handle, len: usize) -> BufferArg {
    // Safety: every handle here is created with at least `len` f32s.
    unsafe { BufferArg::from_raw_parts(h.clone(), len) }
}

#[cube(launch)]
fn k_sgd(p: &mut [f32], g: &[f32], lr: f32, len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        p[i] -= lr * g[i];
    }
}

#[cube(launch)]
fn k_momentum(p: &mut [f32], g: &[f32], v: &mut [f32], lr: f32, mu: f32, len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        v[i] = mu * v[i] + g[i];
        p[i] -= lr * v[i];
    }
}

#[cube(launch)]
fn k_decay(p: &mut [f32], mask: &[f32], shrink: f32, len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        p[i] *= 1.0 - shrink * mask[i];
    }
}

#[cube(launch)]
fn k_average_models(p: &mut [f32], stride: u32, k: u32) {
    let i = ABSOLUTE_POS as u32;
    if i < stride {
        let mut s = 0.0f32;
        for m in 0..k {
            s += p[(m * stride + i) as usize];
        }
        let mean = s / k as f32;
        for m in 0..k {
            p[(m * stride + i) as usize] = mean;
        }
    }
}

#[cube(launch)]
fn k_fill(x: &mut [f32], v: f32, len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        x[i] = v;
    }
}

#[cfg(test)]
use upload_f32 as upload;

/// A whole buffer, blocking (crosses USB4; not per step).
pub fn read(h: &Handle) -> Vec<f32> {
    f32::from_bytes(&client().read_one(h.clone()).unwrap()).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Rng;
    use crate::optim::Sgd;
    use crate::tensor::NdArray;

    /// Needs the discrete GPU. Three models of 5 parameters average to
    /// their elementwise mean, in every model's slot.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn average_models_means_every_slot() {
        let flat: Vec<f32> = (0..15).map(|i| (i * i) as f32).collect();
        let dev = DeviceParams::upload_models(&flat, 3);
        dev.average_models();
        let got = dev.read(&dev.params);
        for i in 0..5 {
            let mean = (flat[i] + flat[5 + i] + flat[10 + i]) / 3.0;
            for m in 0..3 {
                assert!((got[m * 5 + i] - mean).abs() < 1e-4, "slot {m} param {i}: {} vs {mean}", got[m * 5 + i]);
            }
        }
    }

    /// Needs the discrete GPU, so it's opt-in: `cargo test -- --ignored`.
    /// One SGD launch over a packed model must equal `optim::Sgd` on the
    /// same flat vector; `zero_grads` must clear the gradient buffer.
    #[test]
    #[ignore = "needs the discrete GPU"]
    fn device_sgd_matches_optim_sgd() {
        let (vocab, d, heads, d_ff, seq) = (256, 128, 8, 256, 64);
        let mut rng = Rng::new(3);
        let tok = Embedding::new(&mut rng, vocab, d);
        let pos = Embedding::new(&mut rng, seq, d);
        let blocks: Vec<_> = (0..4).map(|_| TransformerBlock::new(&mut rng, d, heads, d_ff)).collect();
        let flat = pack(&tok, &pos, &blocks, &LayerNorm::new(d), &Linear::new(&mut rng, d, vocab));
        let grads: Vec<f32> = (0..flat.len()).map(|_| rng.next_gaussian()).collect();

        let mut dev = DeviceParams::upload(&flat);
        dev.grads = client().create_from_slice(f32::as_bytes(&grads));
        dev.sgd(0.3);
        let got = dev.read(&dev.params);

        let mut want = NdArray::new(flat.clone(), vec![flat.len()]);
        Sgd { lr: 0.3 }.step(&mut want, &NdArray::new(grads, vec![flat.len()]));
        let err = got.iter().zip(&want.data).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-6, "max |gpu - cpu| = {err}");

        dev.zero_grads();
        assert!(dev.read(&dev.grads).iter().all(|&g| g == 0.0));

        let mask: Vec<f32> = (0..flat.len()).map(|i| (i % 3 == 0) as u32 as f32).collect();
        dev.decay(0.5, &upload_f32(&mask));
        let decayed = dev.read(&dev.params);
        let exact = decayed.iter().zip(&got).zip(&mask).all(|((d, g), m)| *d == if *m == 1.0 { 0.5 * g } else { *g });
        assert!(exact, "decay(0.5) must halve exactly the masked parameters and leave the rest");
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn device_momentum_matches_reference() {
        let mut rng = Rng::new(5);
        let n = 10_000;
        let mut p: Vec<f32> = (0..n).map(|_| rng.next_gaussian()).collect();
        let mut dev = DeviceParams::upload(&p);
        let v = upload_f32(&vec![0.0; n]);
        let mut vel = vec![0.0f32; n];
        // Two steps, so the second one exercises the carried velocity.
        for _ in 0..2 {
            let g: Vec<f32> = (0..n).map(|_| rng.next_gaussian()).collect();
            dev.grads = upload(&g);
            dev.momentum(0.1, 0.9, &v);
            for i in 0..n {
                vel[i] = 0.9 * vel[i] + g[i];
                p[i] -= 0.1 * vel[i];
            }
        }
        let err = dev.read(&dev.params).iter().zip(&p).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 1e-6, "max |gpu - cpu| = {err}");
    }
}
