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
pub mod rows;
pub mod tape;
pub mod tokens;

/// Units per cube for 1-D elementwise kernels.
const EW_DIM: u32 = 256;

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
    let mut host = HOST.lock().unwrap();
    if PROFILE.get().is_none() && host.is_none() {
        return;
    }
    let at = std::panic::Location::caller();
    let site = format!("{}:{} {}", at.file(), at.line(), tag());
    if let Some(h) = host.as_mut() {
        h.mark(Some(site.clone()));
    }
    if let Some(p) = PROFILE.get() {
        p.lock().unwrap().mark(Some(site));
    }
}

/// Host time per launch site, for finding where queueing a step goes: the
/// time from one launch's `count_launch` to the next is charged to the
/// first (its launch call, plus the host work before the next launch). Off
/// unless `host_profile_start`. Includes building the site key, ~1 µs.
static HOST: std::sync::Mutex<Option<HostProfile>> = std::sync::Mutex::new(None);

#[derive(Default)]
struct HostProfile {
    open: Option<(String, std::time::Instant)>,
    sites: std::collections::HashMap<String, (usize, f64)>,
}

impl HostProfile {
    fn mark(&mut self, next: Option<String>) {
        let now = std::time::Instant::now();
        if let Some((site, t)) = self.open.take() {
            let e = self.sites.entry(site).or_default();
            e.0 += 1;
            e.1 += (now - t).as_secs_f64();
        }
        self.open = next.map(|n| (n, now));
    }
}

/// Starts charging host time to launch sites (see `HOST`).
pub fn host_profile_start() {
    *HOST.lock().unwrap() = Some(HostProfile::default());
}

/// Charges the open site up to now and stops charging until the next
/// launch; call it before a blocking readback so the wait isn't charged.
pub fn host_profile_cut() {
    if let Some(h) = HOST.lock().unwrap().as_mut() {
        h.mark(None);
    }
}

/// (site, launches, total host s), most expensive first; stops the
/// host profile.
pub fn host_profile_take() -> Vec<(String, usize, f64)> {
    let mut h = HOST.lock().unwrap().take().expect("host_profile_start first");
    h.mark(None);
    let mut v: Vec<_> = h.sites.into_iter().map(|(k, (n, t))| (k, n, t)).collect();
    v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    v
}

/// Per-launch-site GPU time from device timestamps, for finding slow
/// kernels. Each launch runs in its own profile window (no host sync; the
/// window only flushes the queue and brackets its compute pass with
/// timestamp writes), charged to the line that launched it. Off unless
/// `profile_start` is called. A sync-per-launch profiler was tried first
/// and couldn't rank kernels: its ~0.5 ms round trip swamped them.
static PROFILE: OnceLock<std::sync::Mutex<Profile>> = OnceLock::new();

#[derive(Default)]
struct Profile {
    open: Option<(String, cubecl_runtime::client::ProfileWindow)>,
    done: Vec<(String, cubecl::profile::ProfileDuration)>,
}

impl Profile {
    fn mark(&mut self, next: Option<String>) {
        if let Some((site, w)) = self.open.take() {
            self.done.push((site, client().profile_end(w).unwrap()));
        }
        self.open = next.map(|n| (n, client().profile_start().unwrap()));
    }
}

/// Starts charging GPU time to launch sites (see `PROFILE`). Panics if
/// the device can't report timestamps.
pub fn profile_start() {
    let method = client().properties().timing_method;
    assert_eq!(method, cubecl::profile::TimingMethod::Device, "no device timestamps");
    PROFILE.get_or_init(Default::default);
}

/// Closes the open window and returns (site, launches, total GPU s),
/// most expensive first; clears the tally. Windows that carried no
/// measurement are counted as launches but add no time.
pub fn profile_take() -> Vec<(String, usize, f64)> {
    let mut p = PROFILE.get().expect("profile_start first").lock().unwrap();
    p.mark(None);
    let mut sites: std::collections::HashMap<String, (usize, f64)> = Default::default();
    for (site, d) in p.done.drain(..) {
        let e = sites.entry(site).or_default();
        e.0 += 1;
        if let Some(t) = pollster::block_on(d.resolve()) {
            e.1 += t.duration().as_secs_f64();
        }
    }
    let mut v: Vec<_> = sites.into_iter().map(|(k, (n, t))| (k, n, t)).collect();
    v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    v
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
        count_launch();
        k_sgd::launch(client(), self.cubes(), CubeDim::new_1d(EW_DIM), buf(&self.params, self.len), buf(&self.grads, self.len), lr, self.len as u32);
    }

    /// One launch: p *= 1 - shrink * mask over every parameter (mask is
    /// 0 or 1 per parameter, e.g. `Config::decay_mask`). Weight decay at
    /// rate wd is `decay(lr * wd, mask)` before `sgd(lr)`.
    pub fn decay(&self, shrink: f32, mask: &Handle) {
        count_launch();
        k_decay::launch(client(), self.cubes(), CubeDim::new_1d(EW_DIM), buf(&self.params, self.len), buf(mask, self.len), shrink, self.len as u32);
    }

    /// One launch: g = 0.
    pub fn zero_grads(&self) {
        count_launch();
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
fn k_decay(p: &mut [f32], mask: &[f32], shrink: f32, len: u32) {
    let i = ABSOLUTE_POS;
    if (i as u32) < len {
        p[i] *= 1.0 - shrink * mask[i];
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
fn upload(v: &[f32]) -> Handle {
    client().create_from_slice(f32::as_bytes(v))
}

#[cfg(test)]
fn read(h: &Handle) -> Vec<f32> {
    f32::from_bytes(&client().read_one(h.clone()).unwrap()).to_vec()
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

        let mask: Vec<f32> = (0..flat.len()).map(|i| (i % 3 == 0) as u32 as f32).collect();
        dev.decay(0.5, &upload_f32(&mask));
        let decayed = dev.read(&dev.params);
        let exact = decayed.iter().zip(&got).zip(&mask).all(|((d, g), m)| *d == if *m == 1.0 { 0.5 * g } else { *g });
        assert!(exact, "decay(0.5) must halve exactly the masked parameters and leave the rest");
    }
}
