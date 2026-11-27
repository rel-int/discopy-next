`abc.Nat` prints as `Nat(2)` rather than `2`, as `monoidal.Nat` does, so
`eval(str(n)) == n` holds on the nose; a `Matrix` prints its boundaries as
`dom=Nat(2), cod=Nat(2)`, so reading its repr back needs `Nat` in scope, as
`Matrix.environment()` gives ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
