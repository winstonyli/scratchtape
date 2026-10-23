# scratchtape

kind: experiment
version: `Cargo.toml` `[package] version` (0.6.0; tags v0.1.0-v0.5.0 are retrospective on the milestone commits, see `git tag -n99`). No changelog: tag messages are the history (`git tag -n99`). Tags only when the user asks.

- Deferred work lives in `PARKED.md`; live research lines in `docs/open_threads.md` and `docs/tiers_design.md`.
- Long runs, benchmarks, profiles and sweeps launch through `~/.claude/runs/run.ps1` (`~/.claude/RUNS.md`); rows land in `docs/runs.md`. Pass `-Priority Normal` for GPU-feeding jobs.
- Decision rule for model/tier changes: judged on the fresh split (`corpus=fresh22 memory_corpus=fresh150`), `part=0/4` and `part=1/4`; a trained model's number carries ~0.002 of seed noise (`docs/tiers_design.md`).
- A longer variant of a finished constant-lr run continues from its checkpoint with `init=<ckpt>` and a new seed instead of retraining.
