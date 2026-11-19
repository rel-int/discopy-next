> 🐦 Birdsong, 2026-11-18 05:0x–05:4xZ, on the day PR of MEMORY_REPO — no interactive session has
> run since 11-17, so this is the plan this branch was cut from:
>
> > Both halves are filed as [next#625](https://github.com/rel-int/discopy-next/issues/625), and both
> > are upstream's bugs too. The retry-less `uv sync` sits directly below an `Install Graphviz` step
> > carrying `Acquire::Retries=3`, two timeouts and a second attempt, so the action already says what
> > a network step here should do; and `build.yml` already states the principle the cancelled
> > siblings break — *"let every commit on `main` keep the run that proves it green"*. Neither needs
> > your word; only the knob is an implementation choice. **Takeable tonight.**

- [x] fail-fast: false on build.yml's test matrix, so a red sibling does not cancel the Pythons that were measuring the commit
- [x] Install outlasts uv's 30-second default timeout, the knob uv's own error message names, with the second attempt the apt step beside it has
- [x] Pin both in .github/tests: a push to main measures every Python of its matrix, and the Install step's timeout is not uv's default
- [ ] Changelog bullet under Project, and the gates of CONTRIBUTING.md for a .github change: actionlint, pflake8 .github, .github/tests
