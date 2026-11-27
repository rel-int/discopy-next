The law `EGraph.building` compares the outputs each `add` gives on the
builder and on the pure e-graph, not only the e-graphs they end on:
`egraph.History.run` gives both, so a hash-cons keeping the wrong node of a
key fails it ([next#531](https://github.com/rel-int/discopy-next/pull/531)).
