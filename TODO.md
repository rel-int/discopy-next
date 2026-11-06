> Scan through the discopy-next repo history and come up with skills that would improve future development, speed them up, use less tokens etc.

> all of them sound good let's go all at once

Skills in `.claude/skills/`, measured against the history 10-14 → 11-06: 142 of 151
conflicted merges of `main` into a branch conflicted on `CHANGELOG.md` alone.

- [ ] `merge-main`: a `CHANGELOG.md` merge driver (main's file plus the branch's own lines, links
  in number order), registered by the session-start hook, and a rename check of what `main`
  removed since the branch point; replayed on the 142 historical conflicts
- [ ] `next-checks`: every gate on the merged tree, one line per gate, ruff/ty diffed against
  `main`, pylint's raw score on 3.14, failures diffed against the merge base
- [ ] `cubic-round`: unresolved threads compactly, a round's `TODO.md`, replies and resolves
- [ ] `checkboxes`: start, claim, done and finish of `TODO.md` boxes as one pushed commit each, finish
  checking the changelog bullet and its link definition
- [ ] changelog line under Project
