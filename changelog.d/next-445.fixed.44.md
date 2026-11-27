`Hypergraph.to_graph` attaches each input and output to the spider it is
wired to, so `==` sees where an adjoint boundary attaches
(`f @ g @ Cap(x, x.r)` no longer equals `g @ f @ Cap(x, x.r)` as compact
hypergraphs), `hash` no longer raises `KeyError` and
`from_graph(to_graph(h))` gains no spider ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
