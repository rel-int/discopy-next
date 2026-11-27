`build`'s `test` job has a 40-minute backstop instead of 20, above the
9.7-20.3 minutes it runs in on `main` rather than inside it: a job over
`timeout-minutes` reports `cancelled`, so the six jobs it cut over five
runs of `main` in fifteen hours read as nothing having happened
([next#532](https://github.com/rel-int/discopy-next/pull/532)).
