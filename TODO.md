> Review round on #462 (cubic-dev-ai[bot]), head 929b872d:
>
> 1. .claude/skills/merge-main/rename-check.py:89 (thread 4492489825)
> [P2] `git diff --name-status` quotes non-ASCII paths, so `endswith('.py')` fails and those modules are skipped silently. Run git with `core.quotePath=false` so these paths are seen.
>
> 2. .claude/skills/merge-main/replay.py:37 (thread 4492489839)
> [P2] `git()` decodes strictly, so one commit with non-UTF-8 bytes in CHANGELOG.md aborts the whole replay. Decode with surrogateescape here, and make `run_driver` write and read with the same error handler so the bytes round-trip.
>
> 3. .claude/skills/merge-main/replay.py:163 (thread 4492489846)
> [P2] A release on main makes `DRIVER.merge` return `None`, and the `or ("",)` fallback then compares an empty string to the committed file. That counts a correct merge as "units alone differs" and reports 8 false "driver lost" lines. Handle `None` as its own "declined" count and skip the unit comparison.
>
> ```suggestion
>             merged = DRIVER.merge(base, ours, theirs)
>             clean["driver " + compare(got, want)] += 1
>             if merged is None:  # a release: the unit merge declines it
>                 clean["units alone declined"] += 1
>                 continue
>             units = merged[0]
>             clean["units alone " + compare(units, want)] += 1
>             if compare(units, want) == "differs":
>                 notes.append("clean units alone "
>                              + detail(commit, units, want, *texts))
> ```
>
> 4. .claude/skills/merge-main/changelog-merge.py:133 (thread 4492489874)
> [P2] `resolve` loses the branch-position anchor when main rewords a base bullet that the branch left unchanged, so later branch additions move before that bullet. Preserve `at_ours[o_kept[i]]` when `i in t_reworded` before continuing.
>
> ```suggestion
>         if i in o_kept:
>             if i in t_reworded:
>                 at_ours[o_kept[i]] = result[t_reworded[i]]
>             continue
> ```
>
> 5. .claude/skills/cubic-round/threads.py:152 (thread 4492489882)
> [P2] `reviewed_sha` fails to parse the documented `cubic:review-post:<sha>:...` marker and silently falls back to GitHub’s `commit_id`, so cubic status can report the wrong reviewed commit. Match the SHA directly, while optionally accepting the extra field if older markers use it.
>
> ```suggestion
>         r"cubic:review-post:(?:[^:]*:)?([0-9a-f]{40}):",
> ```
>
> 6. .claude/skills/next-checks/checks.py:293 (thread 4492489893)
> [P2] This normalization changes rule and file identity, so a new finding in a numbered rule or filename can match an unrelated baseline finding and be suppressed. Preserve the rule, file, and message values; line numbers are already outside the comparison key.
>
> ```suggestion
>         return tuple(str(k) for k in key)
> ```
>
> 7. .claude/skills/next-checks/checks.py:429 (thread 4492489910)
> [P2] The tests gate discards pytest’s exit status and can report PASS when pytest aborts without a parseable `FAILED`/`ERROR` line. Capture the code in both `gate_tests` and `gate_full`, and fail unparseable nonzero runs instead of treating them as green.
>
> 8. .claude/skills/checkboxes/boxes.py:279 (thread 4492489916)
> [P2] `finish` misses a newly added duplicate definition when the name already existed on the base, because `name in old_names` suppresses every duplicate for that name. Track old definition occurrences (not just names) so only duplicates that were entirely pre-existing are exempt.
>
> 9. .claude/skills/checkboxes/boxes.py:559 (thread 4492489922)
> [P2] `done --filed` accepts invalid text such as `abc`, writes `filed as #abc`, then `FILED` cannot recognize it. `list` and `finish` therefore treat a filed task as an ordinary `[x]`; validate `ISSUE` before committing.
>
> 10. .claude/skills/checkboxes/boxes.py:478 (thread 4492489931)
> [P2] `start --from-file` accepts arbitrary nonblank lines after the first checkbox, but `Todo` ignores unindented non-box lines. Reject malformed input so `list`, `done`, and `finish` cannot silently skip work.
>
> 11. .claude/skills/checkboxes/boxes.py:599 (thread 4492489935)
> [P2] `add` rejects a prompt-only `TODO.md` instead of adding its first box: the no-box branch inserts a separator, then this unconditional header comparison treats it as a prompt change. Skip the comparison for an append when no box existed, or compare the original header as a prefix.
>
> 12. .claude/skills/merge-main/rename-check.py:205 (thread 4492489941)
> [P3] The line number is computed with `find(item)`, which returns the first occurrence in the import block. An item that appears as text in an earlier alias or name reports the wrong line. Search from a running cursor so each item resolves to its own position.
>
> 13. .claude/skills/next-checks/SKILL.md:3 (thread 4492489950)
> [P3] The description overstates coverage: a default run executes `tests` only for changed modules, not the full suite CONTRIBUTING requires. Say that the default run covers changed-module tests and that `--full` runs the whole suite.
>
> ```suggestion
> description: Run the gates of CONTRIBUTING.md's "Run the unit tests" on the current tree of rel-int/discopy-next (the full suite only with --full), one line per gate, ruff/ty/pylint diffed against the merge base. Use before every push and after merging main into a branch.
> ```
>
> 14. .claude/skills/checkboxes/SKILL.md:19 (thread 4492489963)
> [P3] This says a refusal commits nothing, but the non-fast-forward retry path leaves a local merge commit behind when the second push fails, and `start` can exit 2 after that merge too. Reword it so users know a local merge commit may remain (unpushed).
>
> ```suggestion
> Prints `boxes: <sha> <subject>; pushed`. Exit 1 is a refusal and nothing is pushed; after a rejected push the merge of the remote branch may stay as an unpushed local commit. Exit 2 means
> ```
>
> 15. .claude/skills/cubic-round/threads.py:35 (thread 4492489971)
> [P3] This hardcoded `/root/.ccr/ca-bundle.crt` path is specific to one sandbox. When it exists, it replaces the default trust store for every request. Drop this branch and rely on the `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` overrides.
>
> 16. .claude/skills/merge-main/replay.py:48 (thread 4492489979)
> [P3] If `git log` fails, `git()` returns `None` and `.split()` raises `AttributeError`. Check for `None` and exit with a message naming the ref.
>
> 17. .claude/skills/merge-main/replay.py:59 (thread 4492489987)
> [P3] Reading the driver output with universal newlines turns CRLF into LF, so a byte-identical CRLF result is reported as not identical. Open the output with `newline=""` so the comparison sees the raw bytes.
>
> ```suggestion
>         with open(paths[1], encoding="utf-8", newline="") as file:
> ```
>
> 18. .claude/skills/cubic-round/SKILL.md:32 (thread 4492489995)
> [P3] The `&&` does not observe `threads.py` because it runs inside process substitution, so `boxes.py start` still runs when `--todo` exits 1. Capture the Python command with command substitution or an explicit `if` before starting the TODO.

- [x] .claude/skills/merge-main/rename-check.py:89 `git diff --name-status` quotes non-ASCII paths, so `endswith('.py')` fails and those modules are skipped sil…
- [x] .claude/skills/merge-main/replay.py:37 `git()` decodes strictly, so one commit with non-UTF-8 bytes in CHANGELOG.md aborts the whole replay.
- [x] .claude/skills/merge-main/replay.py:163 A release on main makes `DRIVER.merge` return `None`, and the `or ("",)` fallback then compares an empty stri…
- [x] .claude/skills/merge-main/changelog-merge.py:133 `resolve` loses the branch-position anchor when main rewords a base bullet that the branch left unchanged, so…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/cubic-round/threads.py:152 `reviewed_sha` fails to parse the documented `cubic:review-post:<sha>:...` marker and silently falls back to …
- [x] .claude/skills/next-checks/checks.py:293 This normalization changes rule and file identity, so a new finding in a numbered rule or filename can match …
- [x] .claude/skills/next-checks/checks.py:429 The tests gate discards pytest’s exit status and can report PASS when pytest aborts without a parseable `FAIL…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/checkboxes/boxes.py:279 `finish` misses a newly added duplicate definition when the name already existed on the base, because `name i…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/checkboxes/boxes.py:559 `done --filed` accepts invalid text such as `abc`, writes `filed as #abc`, then `FILED` cannot recognize it.
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/checkboxes/boxes.py:478 `start --from-file` accepts arbitrary nonblank lines after the first checkbox, but `Todo` ignores unindented …
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/checkboxes/boxes.py:599 `add` rejects a prompt-only `TODO.md` instead of adding its first box: the no-box branch inserts a separator,…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/merge-main/rename-check.py:205 The line number is computed with `find(item)`, which returns the first occurrence in the import block.
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/next-checks/SKILL.md:3 The description overstates coverage: a default run executes `tests` only for changed modules, not the full su…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/checkboxes/SKILL.md:19 This says a refusal commits nothing, but the non-fast-forward retry path leaves a local merge commit behind w…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/cubic-round/threads.py:35 This hardcoded `/root/.ccr/ca-bundle.crt` path is specific to one sandbox.
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/merge-main/replay.py:48 If `git log` fails, `git()` returns `None` and `.split()` raises `AttributeError`.
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/merge-main/replay.py:59 Reading the driver output with universal newlines turns CRLF into LF, so a byte-identical CRLF result is repo…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 12:40 .claude/skills/cubic-round/SKILL.md:32 The `&&` does not observe `threads.py` because it runs inside process substitution, so `boxes.py start` still…
