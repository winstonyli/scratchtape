# Local patches for `--features cpu` (cubecl 0.11.0-pre.3, Windows)

`patches/` is gitignored (vendored crates.io sources). To recreate, copy
both crates from `$CARGO_HOME/registry/src/index.crates.io-*/` into
`patches/` and apply:

1. **`patches/tracel-llvm-bundler`** (22.1.4-6), `src/config.rs`, `get_libs`:
   call `llvm_config(prefix_os, "--libnames")` instead of `"--libs"`.
   `--libs` prints full paths on Windows, and `split_whitespace` breaks them
   when the path has a space (`C:\Users\First Last\AppData\Local\tracel\...`).
   The symptom is `could not find native static library 'First'`.
2. **`patches/cubecl-cpu`** (0.11.0-pre.3), `Cargo.toml`: add
   `[target.'cfg(target_os = "windows")'.dependencies.bytemuck] version = "1"`.
   `src/compute/affinity/windows.rs` uses `bytemuck` but never declares it.
   pre.4's manifest declares it.

The build script also downloads `tracel-llvm-22.1.4-6-windows-x64.tar.xz`
(about 77 MB, from github.com/tracel-ai/tracel-llvm releases) into
`%LOCALAPPDATA%\tracel`.

## Version survey (2026-09-24)

No version removes patch 1. Every tracel-llvm bundler through 23.1.0-3
(pre.4) still splits `--libs` output on whitespace.

| cubecl | date | wgpu | CPU runtime | Windows patches | burn |
|---|---|---|---|---|---|
| 0.9.0 | 2026-01 | 26 | MLIR, same launch path as 0.10 | spaces | 0.20 |
| 0.10.0 (stable) | 2026-05 | 29 | MLIR, one blocking worker per core; one message per *unit*; no idle poll; spins at `sync_cube`; spawns extra workers when the cube is larger than the core count | spaces | 0.21 |
| 0.11.0-pre.3 (pinned) | 2026-08 | 30 | pliron + LLVM, scheduler with 200 us `IDLE_POLL` | spaces + bytemuck | 0.22-pre.3 |
| 0.11.0-pre.4 | 2026-09-22 | 30 | same as pre.3, pliron 0.18 | spaces | 0.22-pre.4 |

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

## Optional experiment patch (not needed to build)

3. **`patches/cubecl-cpu`**, `src/compute/threadpool/scheduler/dispatcher.rs`:
   `IDLE_POLL` raised from `from_micros(200)` to `from_micros(20_000)`.
   Idle workers then keep polling instead of parking between launches.
   This cut the parallel per-launch floor from 1-5 ms to 0.05-0.4 ms
   (`cubecl_spike cpu-sweep`), at the cost of 16 threads spinning with
   `yield_now` for up to 20 ms after every launch, which is bad on a shared
   machine. It's a source constant, not a runtime option.
