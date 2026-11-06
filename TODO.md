> Tests in discopy-next are taking an unbearable 13-15min, let's cut this time in 3 and make it below 5 minutes. It should be possible to remove a lot of useless unit tests now that we have property testing. Also we don't need to run all the notebooks at every commit, checking them on merge commits should be enough.

- [x] measure where the time goes: on #448's `main` run, `test` is ~7 min of pytest and ~5.5 min of notebooks, `docs` 20 min; locally the tests of 1 s or more take 418 of 573 s
- [x] run the suite on one xdist worker per core, coverage following them with `patch = ["subprocess"]`
- [x] run the notebooks on `main` only, and the `docs` build of a pull request without them
- [x] build the docs in parallel: 13 min 47 s → 5 min 38 s locally without the notebooks
- [x] run the property matrix on every pull request, and delete the unit tests that repeat its laws
- [x] trim the slowest test
- [x] changelog entry
- [x] CI on the PR: build wall 21 → 7 min, test 3.13 still 7:16, docs 7:01, lint 6:01

> yes go with all three suggestions

- [x] a pull request tests on 3.14 only, `main` on 3.12, 3.13 and 3.14
- [x] pylint in parallel: not taken, `--jobs=4` scores 9.319 against serial 9.348 on the same tree (14 false `inconsistent-mro`), under #456's 9.33
- [x] the 26 tests of 4 s or more marked `slow`, run on `main` only: the rest takes 1:58 locally, coverage 98.187%
- [ ] CI under 5 min on the pull request
