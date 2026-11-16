> > Restore what upstream already does: ship `.github/scripts/pylint_axioms.py`, set `load-plugins=pylint_axioms`, re-measure, raise `fail-under` to the new score. Cuts drift instead of adding it.
>
> (rel-int/discopy-next#547, a next-repo bug fixed without asking: no design decision)

- [WIP] @session_01MEJxy5XXNWvvBFsNYRgcmW-2026-11-16 00:24 Measure main's pylint score in several module orders and with a raised limit-inference-results
- [WIP] @session_01MEJxy5XXNWvvBFsNYRgcmW-2026-11-16 00:24 Restore .github/scripts/pylint_axioms.py, adapted to next's axiom and rule, with init-hook and load-plugins in .pylintrc
- [ ] Test the plugin in .github/tests
- [ ] Re-measure in several orders and raise fail-under to the stable score
- [ ] Fix CONTRIBUTING.md's 'with no plugin' sentence and add a CHANGELOG.md bullet
