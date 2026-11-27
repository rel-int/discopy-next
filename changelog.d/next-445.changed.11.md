A law declared `failing` raises `AxiomFailure` only when its equation is
false or its terms refuse to build, and returns the equation otherwise: a
law fixed under its declaration XPASSes in the property matrix, and
`falsify()` returns a real counterexample ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
