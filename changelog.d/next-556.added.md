`discopy.csp`, a constraint satisfaction problem as a diagram and its
language as a functor: `instance` builds a `frobenius.Hypergraph` with a
spider per variable and a box with no input per constraint, plugged on its
scope, and `scopes` reads the constraints back, a box's inputs then its
outputs, so that one drawn in the plane is the same; a `Language` gives a
size to each sort and a predicate to each symbol, with `holds`, `solutions`
by backtracking, and `functor`, the `tensor.Functor` that sends an instance
to the number of its solutions over `int` and to their projection on its
outputs over `bool` (Chandra and Merlin, STOC 1977; Bonchi, Seeber and
Sobociński, CSL 2018) ([next#556](https://github.com/rel-int/discopy-next/pull/556)).
