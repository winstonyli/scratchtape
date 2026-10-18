# Roadmap

Ordered milestones (format: `~/.claude/VERSIONING.md`). Tagging is the user's call; `kind: experiment`, so a tag marks a step.

- **v0.2 - does the dyn gain keep growing with training budget?** Answered 2026-10-17: it halves per doubling (16M -> 32M -0.005, 32M -> 64M -0.0025; dyn stack 0.9765 / 1.0131 on parts 0 / 1, `docs/tiers_design.md` "novels40 at 64M windows"). Tag is the user's call.
- **v0.3 - is the stack worth scaling, or is it at its floor?** Decide between a bigger model / corpus (changes the project's size) and closing the thread. Inputs: the 64M result, `PARKED.md` model-side entries.
