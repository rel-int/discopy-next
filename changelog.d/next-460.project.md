CI runs the suite on one worker per core and the docs on every core, the
notebooks, the tests marked `slow` and Python 3.12 and 3.13 on `main` only,
and the property matrix on every pull request, whose laws the unit tests
stop repeating ([next#460](https://github.com/rel-int/discopy-next/pull/460)).
