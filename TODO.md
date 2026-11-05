> Follow toumix/desire/EVENING.md

Fixes #439: the `test` job's 25-minute backstop cancels healthy 3.12 runs, which take up to 24½ minutes on `main`.

- [ ] `build.yml`: `test` backstop 25 → 40 minutes, its comment saying what healthy runs take now
