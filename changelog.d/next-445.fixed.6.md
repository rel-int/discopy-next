`neural.from_incidence`, `neural.from_relation` and `CMap.from_wiring` accept
numpy integer indices (an integer array, or tuples of `np.int64`), where they
used to say that an existing box, factor or node does not exist ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
