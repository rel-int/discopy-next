`Hypergraph.tensor` shifts each factor's offsets past the codomains of the
factors before it, so `x @ s.to_hypergraph()` decodes in a category without
swaps, like `(x @ s).to_hypergraph()` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
