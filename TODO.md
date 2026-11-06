> Review round on #462 (cubic-dev-ai[bot]), head 8313f130:
>
> 1. .claude/skills/checkboxes/boxes.py:352 (thread 4492855874)
> [P2] A `git add` failure bypasses `commit()`'s rollback handler, leaving the newly written `TODO.md` edit behind after the command fails. Wrap the staging/removal preparation in the same restoration path so every commit failure restores the prior TODO state.
>
> 2. .claude/skills/merge-main/rename-check.py:62 (thread 4492855882)
> [P2] `rename-check` misses names bound by top-level loops and context managers. Because `top_names` only descends through `if`/`try` and assignment nodes, removing `for exported in ...` does not create a removed-name entry, so stale imports pass silently. Include all top-level binding constructs, including loop targets and `with`/`match` bindings, when collecting names.
>
> 3. .claude/skills/next-checks/SKILL.md:37 (thread 4492855891)
> [P2] `pylint-head.json` is not guaranteed to exist in the printed log directory: cache hits skip the measurement that creates it. Preserve the cached messages in the log directory or document the actual cache file path so the suggested follow-up command works.
>
> 4. .claude/skills/next-checks/SKILL.md:28 (thread 4492855902)
> [P2] The `changelog` gate does not actually skip when no bullet was added; any non-empty added line can produce PASS with zero bullets. Require at least one bullet before passing, or document the narrower condition, so a user-facing change cannot bypass the changelog requirement.
>
> 5. .claude/skills/merge-main/rename-check.py:193 (thread 4492855908)
> [P3] The attribute pattern's lookbehind `(?<![\w])` lets a dot precede the match, so `self.foo.Bar` or `x.foo.Bar` gets flagged as a reference to `foo.Bar`. Use `(?<![\w.])` to match the module-path pattern above it.

- [x] .claude/skills/checkboxes/boxes.py:352 A `git add` failure bypasses `commit()`'s rollback handler, leaving the newly written `TODO.md` edit behind a…
- [x] .claude/skills/merge-main/rename-check.py:62 `rename-check` misses names bound by top-level loops and context managers.
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:23 .claude/skills/next-checks/SKILL.md:37 `pylint-head.json` is not guaranteed to exist in the printed log directory: cache hits skip the measurement t…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:23 .claude/skills/next-checks/SKILL.md:28 The `changelog` gate does not actually skip when no bullet was added; any non-empty added line can produce PA…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:23 .claude/skills/merge-main/rename-check.py:193 The attribute pattern's lookbehind `(?<![\w])` lets a dot precede the match, so `self.foo.Bar` or `x.foo.Bar`…
