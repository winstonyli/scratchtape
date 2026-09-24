# Local patches for `--features cpu` (cubecl 0.11.0-pre.4, Windows)

`patches/` is gitignored (vendored crates.io sources). To recreate, extract
`tracel-llvm-bundler-23.1.0-3.crate` from `$CARGO_HOME/registry/cache/` into
`patches/tracel-llvm-bundler` and apply:

1. **`src/config.rs`, `get_libs`**: call `llvm_config(prefix_os, "--libnames")`
   instead of `"--libs"`. On Windows, `--libs` prints full paths, and
   `split_whitespace` breaks them at the space in
   `C:\Users\First Last\AppData\Local\tracel\...`. The symptom is
   `could not find native static library 'First'`. Every bundler version
   through 23.1.0-3 still has this bug.

Only the CPU runtime compiles the bundler. The `wgpu` and `vulkan` builds
never touch it. The build script downloads its own LLVM
(`tracel-llvm-23.1.0-3`, 0.66 GB unpacked) into `%LOCALAPPDATA%\tracel`.
With this one patch, `--features cpu,vulkan` builds in 6 minutes. The
unused 22.1.4-6 copy (pre.3's) was deleted.

pre.4 CPU smoke run (`cubecl_spike cpu`, stock `IDLE_POLL`, BelowNormal
priority on a CPU shared with other jobs, 227 s including JIT), ms per
queued launch:

| shape | matmul pre.4 | matmul pre.3 | elementwise pre.4 | elementwise pre.3 |
|---|---|---|---|---|
| 64 | 17.2 | 3.43 | 0.20 | 3.38 |
| 512x128 | 1.72 | 5.14 | 0.42 | 4.94 |
| 512x256 | 6.13 | 14.6 | 0.50 | 5.42 |
| 2048x128 | 12.8 | 9.48 | 16.4 | 5.85 |

The runtime works. The numbers swing 0.1-10x in both directions, which
looks like contention more than a pre.4 change (`IDLE_POLL` is unchanged;
the rest of the scheduler wasn't diffed). Either way, per-launch cost stays in milliseconds, against
~10 us on the GPU, so the CPU runtime verdict stands.

## Moving from pre.3 to pre.4 (2026-09-24)

- A fresh resolve works now: pliron 0.18 has no version skew, so the
  lockfile borrowed from humble-cortex is gone.
- The cubecl-cpu `bytemuck` patch is no longer needed, because pre.4
  declares the dependency.
- The IDLE_POLL experiment below is not re-applied. It targeted pre.3's
  cubecl-cpu. pre.4's constant is still 200 us.
- API changes:
  - `ComputeClient<R>` became the non-generic `cubecl::client::Client`,
    and `K::launch::<R>` became `K::launch`.
  - `BufferArg` lost its runtime parameter.
  - The `Runtime` trait is no longer re-exported, so the spike takes a
    direct `cubecl-runtime` dependency.
  - `WgpuRuntime::client` has to be written `<WgpuRuntime>::client`,
    because the default compiler type parameter isn't inferred in
    expression position.

## Version survey (2026-09-24)

No version removes patch 1. Every tracel-llvm bundler through 23.1.0-3
(pre.4) still splits `--libs` output on whitespace.

| cubecl | date | wgpu | CPU runtime | Windows patches | burn |
|---|---|---|---|---|---|
| 0.9.0 | 2026-01 | 26 | MLIR, same launch path as 0.10 | spaces | 0.20 |
| 0.10.0 (stable) | 2026-05 | 29 | MLIR, one blocking worker per core; one message per *unit*; no idle poll; spins at `sync_cube`; spawns extra workers when the cube is larger than the core count | spaces | 0.21 |
| 0.11.0-pre.3 (was pinned) | 2026-08 | 30 | pliron + LLVM, scheduler with 200 us `IDLE_POLL` | spaces + bytemuck | 0.22-pre.3 |
| 0.11.0-pre.4 (pinned) | 2026-09-22 | 30 | same as pre.3, pliron 0.18 | spaces | 0.22-pre.4 |

- Cooperative matrix: in 0.10 and 0.11 alike, the `spirv` feature queries
  `VK_KHR_cooperative_matrix` the same way. pre.3 adds only NVIDIA
  coopmat2.
- Launch-path work between 0.10 and pre.3 could lower GPU overhead:
  metadata uniform caching (#1504) and graph capture (#1505).
- Main after pre.4: `IDLE_POLL` is unchanged, with no AMD or Windows
  fixes. There is no date for 0.11.0 stable. Pre-releases come every 2-4
  weeks.
- Open issues:
  - #1336: thread-private arrays race above ~544 B on Vulkan SPIR-V.
  - #1635: the CPU runtime assumes a 512-bit load width.
  - #104: Windows CI is still off.

## Optional experiment patch (pre.3 only, not re-applied)

**Patch 3 (historical): cubecl-cpu 0.11.0-pre.3**, `src/compute/threadpool/scheduler/dispatcher.rs`:
   `IDLE_POLL` raised from `from_micros(200)` to `from_micros(20_000)`.
   Idle workers then keep polling instead of parking between launches.
   This cut the parallel per-launch floor from 1-5 ms to 0.05-0.4 ms
   (`cubecl_spike cpu-sweep`), at the cost of 16 threads spinning with
   `yield_now` for up to 20 ms after every launch, which is bad on a shared
   machine. It's a source constant, not a runtime option.
   Upstream went the other way (survey 2026-09-24, `docs/gpu_step_design.md`):
   #1545 made idle polls yield, and the open #1658/#1566 park the client
   and block drained workers. They report 1.3–5.3× on models, but none of
   it is merged or tested on Windows.
