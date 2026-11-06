> Review round on #462 (cubic-dev-ai[bot]), head f28a8bf3:
>
> 1. .claude/skills/checkboxes/boxes.py:208 (thread 4493078440)
> [P2] `read_todo(required=False)` misses uncommitted deletions of a tracked `TODO.md`, so `start` can replace and commit over dirty state. Check TODO's porcelain status before returning for a missing file.
>
> 2. .claude/skills/checkboxes/boxes.py:489 (thread 4493078452)
> [P2] `start --prompt` misclassifies checkbox text inside the human prompt as TODO work, violating the verbatim prompt header rule. Treat `args.prompt` as the complete prompt and split embedded boxes only for `--from-file`.
>
> 3. .claude/skills/next-checks/SKILL.md:33 (thread 4493078464)
> [P3] The Cost note says pylint reruns only when `discopy/` changed. The cache key also covers `.pylintrc`, `pyproject.toml` and `uv.lock`, so changes to those files also trigger the ~5 min run. Name all four inputs so iterating authors are not surprised.

- [x] .claude/skills/checkboxes/boxes.py:208 `read_todo(required=False)` misses uncommitted deletions of a tracked `TODO.md`, so `start` can replace and c…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:49 .claude/skills/checkboxes/boxes.py:489 `start --prompt` misclassifies checkbox text inside the human prompt as TODO work, violating the verbatim pro…
- [WIP] @session_01MEJxy43kWBLATXt6UT6mUn-2026-11-06 13:49 .claude/skills/next-checks/SKILL.md:33 The Cost note says pylint reruns only when `discopy/` changed.
