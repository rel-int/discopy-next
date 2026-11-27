`hash` of a `tensor.Elementwise`, a `tensor.Diff` and a
`testable.Declaration` depends on no address, so it is the same in every
interpreter with `PYTHONHASHSEED` fixed: the function each holds is compared
by `==` but left out of the hash, and a declaration's category is hashed by
its `factory_key` ([next#506](https://github.com/rel-int/discopy-next/issues/506)).
