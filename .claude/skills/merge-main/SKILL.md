---
name: merge-main
description: Merge origin/main into the current PR branch of discopy-next in one command, with CHANGELOG.md resolved by a merge driver (main's file plus the branch's own bullets and links) and a check for imports of modules or names main removed or renamed. Use whenever a branch is behind main, CI asks for main merged in, or a merge conflicts on CHANGELOG.md.
---

# merge-main

```shell
.claude/skills/merge-main/merge-main.sh ["<what came in>"]
```

Fast-forwards to `origin/<branch>` if behind, merges `origin/main` (CHANGELOG.md by the driver),
checks renamed names, commits `Merge origin/main into <branch>: <#N of each PR in main>` plus the
`next.trailer` lines; never pushes or rebases. Then run `python3 -I .claude/skills/next-checks/checks.py`
and push with `git push` (RULES.md 4: on a rejection, fetch, merge the remote branch, retry).

Output:
- `merge-main: <sha> Merge origin/main into … (N commits, CHANGELOG.md merged|untouched, rename-check clean); not pushed…`:
  committed.
- `already contains origin/main`: nothing to do.
- `conflicts left in:` + files: merge in progress. In `CHANGELOG.md` markers surround only a bullet
  both sides edited, or one deleted and the other reworded; a release on main (a `## ` heading
  changed) leaves the whole file to git's markers. Resolve, run `rename-check.py`, `git commit --no-edit`.
- `path:line old → new?` (`→ ?`: no successor found): the merged tree still uses a module or
  top-level name main deleted or renamed, e.g. `discopy.axioms → discopy.testable?`. Not a git
  conflict; fix it, then `git commit --no-edit` (the message waits in MERGE_MSG).
- exit 2: detached HEAD, uncommitted changes, branch diverged from `origin/<branch>`, fetch, merge
  or rename-check failed.

Edge cases: a plain `git merge`/`git pull` uses the driver once the SessionStart hook has set
`merge.changelog.driver`; without it (no hook, or GitHub's "Update branch" button) git merges
CHANGELOG.md as text. The driver keeps CRLF and odd bytes; identical bullets added on both sides
appear once.
