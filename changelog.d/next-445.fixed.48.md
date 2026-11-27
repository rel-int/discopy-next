`Hypergraph.__repr__` writes `offsets` when any is set, as `to_tree` does,
so a planar hypergraph read back from its repr decodes without swaps
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
