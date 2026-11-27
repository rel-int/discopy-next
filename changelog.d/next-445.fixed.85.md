`matrix.Matrix.permutation` and `Matrix.copy` hold integers again, so a
`Matrix[Expr]`, and any symbolic optyx path matrix with a swap, creation or
selection, composes instead of raising `TypeError`; optyx's
`Permutation.to_path` and `Swap.to_path` return the dtype they are asked for
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
