> Replace the sequential CHANGELOG.md and numbered ADRs in discopy-next, which serialise merges (Carbon's audit #761, item 4.1). Do the work yourself in one PR; don't create another thread.
>
> Changelog: every PR adds a fragment `changelog.d/<slug>.<section>.md` (section = added / changed / fixed / removed, slug = branch name, the fragment carries its `next#NNN` link). Compile with towncrier (or a short script if towncrier fits badly) at release; CHANGELOG.md keeps released versions only, and `[Unreleased]` points at `changelog.d/`. Migrate the current Unreleased entries into fragments. Add a CI check that a PR touching `discopy/` adds a fragment, with a label to opt out.
>
> ADRs: no numbers on next. A record is `docs/adr/YYYY-MM-DD-<slug>.md`, dated when accepted, so files sort chronologically and `index.md` keeps a glob toctree. A number is assigned only when a record is sliced upstream to discopy/discopy. Rename 0001–0011 to dated slugs (date from their Status line), update `index.md`'s description and supersession wording ("Superseded by `<file>`"), and fix every reference to the old numbers across the repo. While there, fix 0010/0011 still calling #694 and #693 open.
>
> This reverses #695's ruling, which kept numbers (#695 is closed): say so in the PR. #750 checks distinct ADR numbers. Repurpose it, or replace it in this PR, into a test that ADR filenames match the dated-slug pattern and every cross-reference to an ADR resolves. Open PRs that add 0008/0012 records (#679, #690, #706, #734, #737) will need renaming after this lands: list them in the PR description and leave a short comment on each.

- [WIP] @T-e5f32d46-2026-11-27 19:10 changelog.d/ fragments compiled by towncrier: config, template, CHANGELOG.md keeps released versions
- [WIP] @T-e5f32d46-2026-11-27 19:10 Migrate every [Unreleased] bullet of CHANGELOG.md into a fragment
- [ ] CI check: a PR touching discopy/ adds a fragment, label no-changelog opts out
- [ ] Repoint the tooling that reads [Unreleased]: checkboxes finish, next-checks changelog gate, merge-main driver, CONTRIBUTING, AGENTS
- [ ] Rename ADRs 0001-0011 to YYYY-MM-DD-slug.md, update index.md and every reference
- [ ] Repurpose test/adr.py: dated-slug filenames, every ADR cross-reference resolves
- [ ] Open the PR saying it reverses #695's ruling; comment on #679, #690, #706, #734, #737
