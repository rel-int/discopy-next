> ask, one 🚀: pin `runs-on: ubuntu-24.04` on the jobs that install Graphviz (`test` and `docs` in `build.yml`), with a line in `.github/tests/workflows.py` that keeps it. option 1 above.
>
> not taken: stay on `ubuntu-latest` and add the archive to the image's sources (option 2), which picks a cause the log does not give.
>
> default: the pin. asked 2026-11-20, due 11-22: with no word an agent opens it then, and one word reverses it.

- [WIP] @session_01C7YfAR6V7pj9o6FNkELjMG-2026-11-22 01:21 Pin runs-on: ubuntu-24.04 on build.yml's test and docs jobs, the two that pass graphviz: 'true' to .github/actions/setup
- [ ] A test in .github/tests/workflows.py that keeps the pin: every job asking the setup action for Graphviz pins its image
- [ ] Run the .github gates: actionlint, pflake8 .github, .github/tests
- [ ] Changelog bullet under Project citing this pull request
