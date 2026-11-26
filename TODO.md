> The published docs show "This build does not run this notebook" for nearly every marimo notebook. That is expected given the setup: `docs/conf.py` sets `DISCOPY_SKIP_NOTEBOOKS=1` on Read the Docs, because running the notebooks there exceeds its 15-minute limit. We have also already crashed RTD's small machines by running too many notebooks there, so do not try to run them on RTD, in sequence or in parallel. Meanwhile, the `docs` job in `.github/workflows/build.yml` renders every notebook on `main` (via `docs/export_notebooks.py` from the `builder-inited` hook) but only uploads `docs/_build/html` as a workflow artifact. Nothing publishes it.
>
> Goal: the rendered notebooks computed by GitHub Actions on `main` should appear in the published docs. Two possible shapes:
> (a) RTD keeps hosting, but its build uses notebook HTML that Actions rendered (for example, from a branch, a release asset, or an artifact) instead of executing the notebooks itself.
> (b) Deploy the Actions docs build itself, for example to GitHub Pages.
>
> Prefer whichever keeps the current docs URL and hosting unless that turns out to be unreasonable. Changing where the docs are hosted is toumix's decision, so if (b) looks necessary, explain why rather than switching on your own. Keep `optyx/readme_example` unexecuted as it is now. Follow the repo's AGENTS.md.

- [x] publish the notebooks the `docs` job renders on `main` to a branch, from a job of its own with write access
- [WIP] @T-dfc2c559-2026-11-26 16:59 embed that branch's HTML in the Read the Docs build instead of the link pages, without running any notebook there
- [WIP] @T-dfc2c559-2026-11-26 16:59 docs, tests and changelog
