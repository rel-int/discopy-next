> Task: fix https://github.com/rel-int/discopy-next/issues/251: `marimo export html` crashes on notebooks whose contraction triggers cotengra's process-pool (loky) optimizer; the same notebook runs as a script. USER's decision, recorded on the issue (comment 6955727504): "the fix goes in `docs/export_notebooks.py`, for every notebook it runs (no process pool for cotengra, or a safe start method), not in one notebook." A previous worker died in a container restart with nothing pushed; check `git ls-remote origin | grep -i 251` for a leftover branch before starting. Branch `claude/251-export-cotengra`, worktree /home/user/wt/251.
>
> Read the issue body and comments (mcp__github__issue_read) for the reproduction. Also related: https://github.com/rel-int/discopy-next/issues/435 (KNOWN_BROKEN in docs/export_notebooks.py names notebooks that fail on #251): if your fix makes those notebooks pass `uv run python docs/export_notebooks.py --check <name>`, remove them from KNOWN_BROKEN in the same PR and say so; do not decide #435's open question (whether a stale exemption fails --check). Test: `.github/tests` or a test of the script showing the environment/setting is applied; and the real `--check` of the affected notebooks run locally (report the result).

- [x] a test that a notebook starting a spawn process pool exports
- [x] `export` runs every notebook so that a spawned worker can re-import its main module
- [x] CHANGELOG.md bullet
- [ ] `--check optyx/readme_example`, and KNOWN_BROKEN if it passes
