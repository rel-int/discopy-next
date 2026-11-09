> TikZ drawing does not compile for a Matplotlib-only colour (tab:blue, C0, 0.5)
>
> Evening: next-repo bug, no design decision (desire OPERATIONS.md 'Bugs in a next repo')

- [WIP] @session_01MEJxy65KogNCEukoKHtHe1-2026-11-09 00:21 Failing-first test: a Matplotlib-only colour (tab:blue, C0, 0.5, an RGB tuple) is written as an xcolor rgb spec, with and without use_tikzstyles
- [ ] Fix TikZ.format_color and its callers, keeping xcolor names and existing baselines
- [ ] CHANGELOG.md bullet, next-checks
