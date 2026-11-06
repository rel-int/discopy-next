> Tests in discopy-next are taking an unbearable 13-15min, let's cut this time in 3 and make it below 5 minutes. It should be possible to remove a lot of useless unit tests now that we have property testing. Also we don't need to run all the notebooks at every commit, checking them on merge commits should be enough.

- [x] measure where the time goes: on #448's `main` run, `test` is ~7 min of pytest and ~5.5 min of notebooks, `docs` 20 min; locally the tests of 1 s or more take 418 of 573 s
- [x] run the suite on one xdist worker per core, coverage following them with `patch = ["subprocess"]`
- [x] run the notebooks on `main` only, and the `docs` build of a pull request without them
- [x] build the docs in parallel
- [x] run the property matrix on every pull request, and delete the unit tests that repeat its laws
- [x] trim the slowest test
- [ ] changelog entry
- [ ] CI under 5 min on the pull request
