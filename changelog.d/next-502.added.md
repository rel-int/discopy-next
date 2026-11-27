`egraph.Builder`, a mutable e-graph built as egg does, with a union-find, a
hash-cons and a worklist: `add` in amortised constant time, `rebuild`
repairing only what changed, started with `EGraph.builder` and frozen back
into an `EGraph` with `freeze`, where the law `EGraph.building` says it is
what the same operations give on the pure one, over `egraph.History`
([next#502](https://github.com/rel-int/discopy-next/pull/502)).
