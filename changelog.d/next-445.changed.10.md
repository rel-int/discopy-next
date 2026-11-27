`Equation` compares by value: `==` compares type, terms, symbols, space and
`up_to`, a callable `up_to` by identity, and its hash agrees, so an equation
is hashable when its terms are; its repr prints the symbols unless they are
all `=`, and a `monoidal.Equation`'s space unless it is 1, so that
`eval(repr(e)) == e` unless `e` has an `up_to` of its own ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
