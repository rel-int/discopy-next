`tensor.Functor` evaluates `dtype=Expr` diagrams exactly and functorially:
its boundary and loop identities are integers, not floats, so
`F(f >> f) == F(f) >> F(f)` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
