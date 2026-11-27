`python.Function.uncurry` takes `abc`'s `n`, splitting a nested exponential
as every other closed category does, where `uncurry(0)` bound `0` to `left`
and uncurried with its arguments swapped; `curry(0)` and `uncurry(0)` are
the identity, and `uncurry(n)` raises `ValueError` when the exponent has
more than `n` types ([next#477](https://github.com/rel-int/discopy-next/issues/477)).
