---
name: next-checks
description: Run every gate of CONTRIBUTING.md's "Run the unit tests" on the current tree of rel-int/discopy-next, one line per gate, ruff/ty/pylint diffed against the merge base. Use before every push and after merging main into a branch.
---

# next-checks

```bash
python3 -I .claude/skills/next-checks/checks.py [--only flake8,github,pylint,ruff,ty,tests,notebooks,changelog] [--full] [--all-notebooks] [--base REF]
```

Run from the root of the checkout or worktree under check; uncommitted and untracked files count.
Base is `merge-base(HEAD, origin/main)`; `git fetch origin main` first if it may be stale.

One line per gate, `PASS|FAIL|SKIP <gate>: <fact>`, then `base <sha>, <seconds>, logs <dir>`;
exit 1 on any FAIL. What the lines do not say:

- ruff, ty and pylint fail only on findings new since base, matched with line numbers ignored.
- pylint's score is the raw `json2` one, on a cached 3.14 venv synced like CI's lint job.
  `3.14 sync failed` means another interpreter: the score may differ from CI's by ~0.02.
  `FAIL lock` means `uv.lock` is stale, which CI rejects: run `uv lock`.
- `tests` runs each changed module's test file and doctests; it names modules with no test file.
  A change to `test/plugin.py`, `test/fixtures/` or a conftest only gets a collection check:
  run `--full` (~10 min locally, the whole suite under coverage, failures diffed with base).
- `changelog` SKIPs when no bullet was added: add one if the change is user-facing.
  It names any of base's lines the branch removes, since a bad merge drops main's bullets.
- SKIP is not PASS: say which gates skipped when you report.

Cost: pylint takes ~5 min, serially, whenever `discopy/` changed since its last run (results
are cached under `~/.cache/next-checks/` by content). While iterating, use
`--only flake8,ruff,ty,tests`; run everything once before pushing.

On FAIL the line names the first finding. For more, grep the log, never cat it:
`grep -n -m 20 '^FAILED\|^ERROR' <dir>/tests.log`, or filter `<dir>/pylint-head.json` by symbol.

Local green is evidence for the checks that ran, not for CI: CI on the pushed branch, with `main`
merged in, remains the authority.
