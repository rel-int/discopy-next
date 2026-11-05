> Read desire/EVENING.md and run a 🌙 Evening turn for tonight.

Taken from [#438](https://github.com/rel-int/discopy-next/issues/438), first half: the two ruff
ratchet rules with one instance each, re-measured on `42945740` rather than read off the issue,
which measured `b0e874a6`.

- [WIP] @session_01C7YfAQc4aimLRHXsanFC5u-2026-11-05 01:25 Re-measure `E714` and `RUF022` with the two rules dropped from `ignore`
- [ ] `discopy/quantum/gates.py`: `not state["_array"] is None` → `is not None`
- [ ] `discopy/traced.py`: sort `__all__`, keeping the compact layout rather than ruff's one-name-per-line autofix
- [ ] `pyproject.toml`: drop `E714` and `RUF022` from `lint.ignore`, so the list only shrinks
- [ ] `pflake8 discopy`, `ruff check discopy`, `pylint discopy` on the change, and the test files that cover both sites
