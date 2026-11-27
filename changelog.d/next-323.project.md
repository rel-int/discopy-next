`CONTRIBUTING.md`'s benchmark command runs the `zx` suite, which needs PyZX,
and `test/pivotal.py` skips its sympy test instead of failing on it, so
`--skip-extra` holds there too ([next#323](https://github.com/rel-int/discopy-next/issues/323), [next#350](https://github.com/rel-int/discopy-next/issues/350)).
