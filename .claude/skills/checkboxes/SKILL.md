---
name: checkboxes
description: Start, claim, tick, extend and finish a branch's TODO.md on discopy-next (RULES.md 1-3), one command per step, each one commit pushed; finish checks the CHANGELOG.md bullet and its link definitions. Use instead of editing TODO.md by hand for a mutex claim, a ticked box, a review round or the deletion that hands a PR to review.
---

# checkboxes

```shell
T=".claude/skills/checkboxes/boxes.py"
python3 -I $T start --prompt "<human prompt>" --box "<point>" ...  # RULES.md 1 (--from-file F)
python3 -I $T start --round --from-file feedback.md --box ...      # RULES.md 2, a review round
python3 -I $T list                  # "N [ |WIP @owner Nh [stale]|x|filed] text"
python3 -I $T claim N               # RULES.md 3: [ ] -> [WIP] @<session>-<yyyy-MM-dd HH:mm> UTC
python3 -I $T done N [--filed 471]  # -> [x], or "[x] …, filed as #471"
python3 -I $T add "<point>" [--claim]
python3 -I $T finish [--no-changelog]  # RULES.md 2: git rm TODO.md once every box is [x]
```

Prints `boxes: <sha> <subject>; pushed`. Exit 1 is a refusal, nothing committed; exit 2 means
another session holds the box (or started a TODO.md): pick another. `--dry-run` prints the
commit and diff. Trailers come from `git config --get-all next.trailer` (set once per session,
next-coordinator's `new-worktree.sh` does it) plus `--trailer LINE`; the session id is
`$CLAUDE_CODE_REMOTE_SESSION_ID` (`cse_` read as `session_`) or `--session ID`.

- It pushes to the same-named branch on the remote, and refuses on main.
- A non-fast-forward push is undone, the remote branch merged (RULES.md 4), the step redone
  and pushed once more; any other push or commit failure is undone and reported.
- A claim older than 12 h is stale and reclaimable; one with no date needs `--force`, as does
  `done` on someone's fresh claim.
- `finish` needs a new bullet under `## [Unreleased]` in HEAD's CHANGELOG.md since the merge
  base with `--base` (default `origin/main`), a definition for each ref it cites, new
  definitions in number order. `--no-changelog` (no user-facing change) only warns.
