> Review round on #462 (cubic-dev-ai[bot]), head 0a27d0a5:
>
> 1. .claude/skills/next-checks/SKILL.md:18 (thread 4492664550)
> [P2] This line says pylint fails only on new findings, but the pylint gate also fails on a score below `fail-under` and on any `unused-import` in head even if base has it. Scope the diff-only rule to ruff and ty, and list the pylint absolute conditions.
>
> ```suggestion
> - ruff and ty fail only on findings new since base, matched with line numbers ignored. pylint also fails on a score below `fail-under` and on any `fail-on` message (`unused-import`) in head, even when base has it.
> ```
>
> 2. .claude/skills/merge-main/replay.py:163 (thread 4492664565)
> [P2] This `DRIVER.merge` call is unguarded, unlike the conflicted branch, which wraps it and counts `crashed`. If `merge` raises on a clean-merge commit, the replay aborts and the remaining counts are lost. Wrap it the same way and count the crash.
>
> ```suggestion
>             try:
>                 merged = DRIVER.merge(base, ours, theirs)
>             except Exception:  # pylint: disable=broad-exception-caught
>                 clean["crashed"] += 1
>                 continue
> ```
>
> 3. .claude/skills/merge-main/rename-check.py:153 (thread 4492664575)
> [P2] `still_there` treats any directory named after a removed module as proof that the module still exists. Deleting `discopy/old/__init__.py` while retaining a submodule then suppresses stale imports from `discopy.old`; only accept a `.py` module or a package directory containing `__init__.py`.
>
> 4. .claude/skills/cubic-round/threads.py:287 (thread 4492664580)
> [P2] When the ccr route fails, every thread has unknown resolution, but `--todo` still treats them all as open. Already-resolved threads then go into TODO.md with no warning, because the "resolution unknown" note is only printed in the normal output. Refuse to write the TODO when resolution is unknown.
>
> ```suggestion
>     if args.todo and any(t["resolved"] is None for t in threads):
>         sys.exit("threads.py: resolution unknown (ccr route failed), no TODO.md written")
>     if args.todo:
> ```
>
> 5. .claude/skills/cubic-round/threads.py:68 (thread 4492664593)
> [P2] The summary group `(.*?)</summary>` can run past the outer `</summary>` when a `<details>` block contains a nested `<details>`. The whole outer block is then matched and dropped or truncated, so the finding text disappears. Restrict the group so it cannot cross a `</summary>`.
>
> <details>
> <summary>Prompt for AI agents</summary>
>
> ```text
> Check if this issue is valid — if so, understand the root cause and fix it. When an issue isn't valid or won't be fixed in this PR, reply in its thread with the reason and then resolve the thread. At .claude/skills/cubic-round/threads.py, line 68:
>
> <comment>The summary group `(.*?)</summary>` can run past the outer `</summary>` when a `<details>` block contains a nested `<details>`. The whole outer block is then matched and dropped or truncated, so the finding text disappears. Restrict the group so it cannot cross a `</summary>`.</comment>
>
> <file context>
> @@ -0,0 +1,327 @@
> +
> +def strip_details(body):
> +    pattern = re.compile(
> +        r"<details>\s*<summary>(.*?)</summary>((?:(?!<details>).)*?)</details>",
> +        re.S | re.I,
> +    )
> </file context>
> ```
>
> </details>
>
> ```suggestion
>         r"<details>\s*<summary>((?:(?!</summary>).)*?)</summary>((?:(?!<details>).)*?)</details>",
> ```
>
> 6. .claude/skills/checkboxes/boxes.py:556 (thread 4492664605)
> [P2] `done` can tick an unclaimed `[ ]` box, bypassing the required mutex claim and its pushed lock. Require the box to be claimed by this session (or explicitly reject an unclaimed box) before allowing `[x]`.
>
> 7. .claude/skills/merge-main/merge-main.sh:30 (thread 4492664612)
> [P3] `tac` is not available on default macOS, so the PR-number list is lost there and the merge subject silently falls back to `N commits`. Use `awk` to reverse the lines instead, which works on both GNU and BSD systems.
>
> ```suggestion
> what="${1:-$(git log --first-parent --format=%s HEAD..origin/main | grep -o '^Merge pull request #[0-9]*' | grep -o '#[0-9]*' | awk '{ a[NR] = $0 } END { for (i = NR; i > 0; i--) print a[i] }' | paste -sd ' ' -)}"
> ```
>
> 8. .claude/skills/merge-main/replay.py:99 (thread 4492664618)
> [P3] A committed unit that ours kept, main deleted, and the driver dropped matches no category, so it is reported as `blank lines`. Add a `hand` case for committed units that one side deleted, so the summary does not hide real differences.
>
> 9. .claude/skills/cubic-round/threads.py:227 (thread 4492664623)
> [P3] The normal compact output drops the `[addressed <sha>]` marker for unresolved threads that are not `ours`, even though the skill documents this flag. Include the marker in the compact header and avoid adding it a second time in `block()`, otherwise users cannot see that the finding was already addressed.

- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:01 .claude/skills/next-checks/SKILL.md:18 This line says pylint fails only on new findings, but the pylint gate also fails on a score below `fail-under…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:01 .claude/skills/merge-main/replay.py:163 This `DRIVER.merge` call is unguarded, unlike the conflicted branch, which wraps it and counts `crashed`.
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:01 .claude/skills/merge-main/rename-check.py:153 `still_there` treats any directory named after a removed module as proof that the module still exists.
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:01 .claude/skills/cubic-round/threads.py:287 When the ccr route fails, every thread has unknown resolution, but `--todo` still treats them all as open.
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:01 .claude/skills/cubic-round/threads.py:68 The summary group `(.*?)</summary>` can run past the outer `</summary>` when a `<details>` block contains a n…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:01 .claude/skills/checkboxes/boxes.py:556 `done` can tick an unclaimed `[ ]` box, bypassing the required mutex claim and its pushed lock.
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:01 .claude/skills/merge-main/merge-main.sh:30 `tac` is not available on default macOS, so the PR-number list is lost there and the merge subject silently f…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:01 .claude/skills/merge-main/replay.py:99 A committed unit that ours kept, main deleted, and the driver dropped matches no category, so it is reported …
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:01 .claude/skills/cubic-round/threads.py:227 The normal compact output drops the `[addressed <sha>]` marker for unresolved threads that are not `ours`, ev…
