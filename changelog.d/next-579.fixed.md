`tensor.Functor.contract` lifts NumPy's default bound on intermediates, the
size of the largest operand, for combinatorial maps as well as hypergraphs on
the `"greedy"` path: when every pairwise step exceeds the largest operand,
e.g. `hopf.Double(hopf.GroupAlgebra(Symmetric[3])).is_valid()`, NumPy no
longer falls back to one naive sum over all indices ([next#579](https://github.com/rel-int/discopy-next/issues/579), [#595](https://github.com/discopy/discopy/issues/595)).
