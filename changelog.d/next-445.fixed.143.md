`Hypergraph.make_left_monogamous` reads the legs of each merge in port
order, as `make_right_monogamous` already did, where it read them in the
iteration order of a `set` of port indices, so the wires, `==` and `repr` of
an affected hypergraph can change — the merge of a commutative monoid is
symmetric, so what it evaluates to does not ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
