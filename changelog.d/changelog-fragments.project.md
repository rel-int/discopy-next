Each pull request adds its changelog entry as a fragment,
`changelog.d/<slug>.<section>.md`, compiled into CHANGELOG.md by towncrier
at release, and the `changelog` workflow fails one that changes `discopy/`
without a fragment unless it is labelled `no-changelog`. Architecture decision
records are named `docs/adr/YYYY-MM-DD-<slug>.md` after the day they are
accepted, with no number, and `test/adr.py` checks every link to one resolves
([next#772](https://github.com/rel-int/discopy-next/pull/772)).
