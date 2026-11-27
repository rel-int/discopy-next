`proptest` groups a push to `main` by commit, as `build`, `benchmark`,
`lean` and `paper` do, and a scheduled or dispatched run by its own run id:
one group per ref keeps one running and one pending run, so a third merge
cancelled the pending one before it ever reached a runner.
`.github/tests/workflows.py` checks every one of the five ([next#591](https://github.com/rel-int/discopy-next/pull/591)).
