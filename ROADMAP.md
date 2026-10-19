# Roadmap

Ordered milestones (format: `~/.claude/VERSIONING.md`). Tagging is the user's call; `kind: experiment`, so a tag marks a step.

- **v0.2 - does the dyn gain keep growing with training budget?** Answered 2026-10-17: it halves per doubling (16M -> 32M -0.005, 32M -> 64M -0.0025; dyn stack 0.9765 / 1.0131 on parts 0 / 1, `docs/tiers_design.md` "novels40 at 64M windows"). Tag is the user's call.
- **v0.3 - is the stack worth scaling, or is it at its floor?** Answered 2026-10-18: not at its floor. d=384 beats d=256 by 0.014-0.018 in the stack at matched memory (`docs/tiers_design.md` "Width test"). Next milestone below.
- **v0.4 - how far does width go, and can the full memory be kept?** Partly answered 2026-10-18: d=384 with keys narrowed to 256 dims fits the full memory and gives the best stack (dyn 0.9689 / 1.0085). Seed repeat done (3 seeds mean 0.9694 / 1.0087, spread 0.0015): the win stands. d=512 (one seed) reaches dyn 0.9568 / 0.9928, -0.013 / -0.016 vs d=384. d=640 (one seed) reaches 0.9459 / 0.9850, gains per width step now 0.011 / 0.008. Open: d=768 (needs dispatch fix, PARKED.md) / more blocks, a seed repeat of d=384, longer d=384 training.
