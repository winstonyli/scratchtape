//! Device-resident training step on the discrete eGPU (cubecl, Vulkan
//! SPIR-V). Design and milestones: docs/gpu_step_design.md. Milestone 1:
//! device selection, the flat parameter/gradient buffers, SGD.
use crate::nn::{Embedding, LayerNorm, Linear, TransformerBlock};
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{RuntimeOptions, Vulkan, WgpuDevice, WgpuDeviceKind, WgpuRuntime, init_setup};
use cubecl_runtime::runtime::Runtime;
use std::sync::OnceLock;

/// Units per cube for 1-D elementwise kernels.
const EW_DIM: u32 = 256;

static CLIENT: OnceLock<Client> = OnceLock::new();

/// The first discrete GPU, on Vulkan. Never falls back to the iGPU or CPU:
/// cubecl panics with "No Discrete GPU device found" if there's none, and
/// its `CUBECL_WGPU_DEFAULT_DEVICE` override only applies to the default
/// device kind, not this one. Logs the adapter once. Cached: cubecl
/// registers one client per device.
pub fn client() -> &'static Client {
    CLIENT.get_or_init(|| {
        let device = WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0));
        let setup = init_setup::<Vulkan>(&device, RuntimeOptions::default());
        let info = setup.adapter.get_info();
        eprintln!("gpu_step: {} ({:?}, {:?}, driver {})", info.name, info.device_type, setup.backend, info.driver_info);
        <WgpuRuntime>::client(&device)
    })
}

/// Every trainable parameter of a plain tiny_lm, flattened in this order:
/// token embedding, position embedding, each block's `to_fused_flat`, final
/// LayerNorm, output projection. Gradients use the same offsets.
pub fn pack(token_emb: &Embedding, pos_emb: &Embedding, blocks: &[TransformerBlock], final_ln: &LayerNorm, output_proj: &Linear) -> Vec<f32> {
    let mut out = token_emb.to_flat();
    out.extend(pos_emb.to_flat());
    for b in blocks {
        out.extend(b.to_fused_flat());
    }
    out.extend(final_ln.to_flat());
    out.extend(output_proj.to_flat());
    out
}

/// Parameters and their gradients, resident on the device as two flat
/// buffers with matching offsets.
pub struct DeviceParams {
    pub params: Handle,
    pub grads: Handle,
    pub len: usize,
}

impl DeviceParams {
    /// Uploads `flat` (from `pack`); gradients start at zero.
    pub fn upload(flat: &[f32]) -> Self {
        let c = client();
        let params = c.create_from_slice(f32::as_bytes(flat));
        let grads = c.create_from_slice(f32::as_bytes(&vec![0.0f32; flat.len()]));
        Self { params, grads, len: flat.len() }
    }

    /// One launch: p -= lr * g over every parameter.
    pub fn sgd(&self, lr: f32) {
        k_sgd::launch(client(), self.cubes(), CubeDim::new_1d(EW_DIM), buf(&self.params, self.len), buf(&self.grads, self.len), lr, self.len as u32);
    }

    /// One launch: g = 0.
    pub fn zero_grads(&self) {
        k_fill::launch(client(), self.cubes(), CubeDim::new_1d(EW_DIM), buf(&self.grads, self.len), 0.0f32, self.len as u32);
    }

    /// Blocking readback (crosses USB4; for tests and checkpoints only).
    pub fn read(&self, h: &Handle) -> Vec<f32> {
        f32::from_bytes(&client().read_one(h.clone()).unwrap())[..self.len].to_vec()
    }

    fn cubes(&self) -> CubeCount {
        CubeCount::Static((self.len as u32).div_ceil(EW_DIM), 1, 1)
    }
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
fn k_fill(x: &mut [f32], v: f32, len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        x[i] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Rng;
    use crate::optim::Sgd;
    use crate::tensor::NdArray;

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
    }
}
