`discopy.enriched`, the change of base of a category `C` along a monad `M`:
an `enriched.Arrow[C, M]` from `A` to `B` is an element of `M(C(A, B))`,
composed by pairing then composing, with `tensor`, a monoidal product only
when `M` is commutative (not for `List`, where interchange fails), and, for
`C` the Python functions, `to_channel`. An `enriched.Functor` extends a functor `C -> D`
along the classmethod `D.algebra(M, dom, cod)`, given for the new list monad
`kleisli.monad.List` by `cat.Arrow`, which sends a list of arrows to their
`cat.Sum`, and by `Matrix`, which adds the matrices ([next#134](https://github.com/rel-int/discopy-next/issues/134)).
