// wgpu 30's types nest deeply enough that proving gpu.rs's
// `OnceLock<GpuContext>` is Send + Sync overflows the default limit (128),
// a future-incompat warning (rust-lang/rust#159228) slated to become an error.
#![recursion_limit = "256"]
pub mod tensor;
pub mod tape;
pub mod optim;
pub mod nn;
pub mod gpu;
pub mod gpu_step;
