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
