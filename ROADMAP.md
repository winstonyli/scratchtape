# Roadmap

Ordered milestones (format: `~/.claude/VERSIONING.md`). Tagging is the user's call; `kind: experiment`, so a tag marks a step.

- **v0.2 - does the dyn gain keep growing with training budget?** Answered 2026-10-17: it halves per doubling (16M -> 32M -0.005, 32M -> 64M -0.0025; dyn stack 0.9765 / 1.0131 on parts 0 / 1, `docs/tiers_design.md` "novels40 at 64M windows"). Tag is the user's call.
- **v0.3 - is the stack worth scaling, or is it at its floor?** Answered 2026-10-18: not at its floor. d=384 beats d=256 by 0.014-0.018 in the stack at matched memory (`docs/tiers_design.md` "Width test"). Next milestone below.
- **v0.4 - how far does width go, and can the full memory be kept?** d=512 and/or more blocks on novels40; make the 125M-key memory fit at wider d (a compressed key store or fewer key dimensions); compare to the full-memory d=256 best (0.9765 / 1.0131).
