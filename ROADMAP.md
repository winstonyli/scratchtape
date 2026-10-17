# Roadmap

Ordered milestones (format: `~/.claude/VERSIONING.md`). Tagging is the user's call; `kind: experiment`, so a tag marks a step.

- **v0.2 - does the dyn gain keep growing with training budget?** novels40 at 64M windows, continued from the 32M checkpoint (`scripts/train_novels40_64m.sh`, running 2026-10-17); 16M -> 32M gained 0.005 under dyn on both fresh-split parts. Result goes in `docs/tiers_design.md`.
- **v0.3 - is the stack worth scaling, or is it at its floor?** Decide between a bigger model / corpus (changes the project's size) and closing the thread. Inputs: the 64M result, `PARKED.md` model-side entries.
