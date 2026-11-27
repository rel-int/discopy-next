`CMap.loops` are stored sorted by `repr`, so maps compare their loops as a
multiset: `Diagram.to_map()` is the fold of `then` and `CMap.dagger` is
contravariant on maps with loops ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
