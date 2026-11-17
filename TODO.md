> Fix rel-int/discopy-next#596: `build`'s Coverage step read 36% when `coverage combine` found no xdist worker data. Do what the issue says closes it: make the step say what it combined with `coverage combine --debug=dataio`, and before `coverage report`, assert the number of data files combined so that missing worker data fails as its own error, legible as such, rather than as a fail-under regression.

- [x] the Coverage step of build.yml combines with --debug=dataio and fails with its own error when it combined the controller's data file alone
- [WIP] @session_01MEJxy5Y19SnBwRFf22vqRj-2026-11-17 17:29 a test in .github/tests runs the step's snippet with worker data and without it
- [WIP] @session_01MEJxy5Y19SnBwRFf22vqRj-2026-11-17 17:29 CHANGELOG.md bullet under Project
