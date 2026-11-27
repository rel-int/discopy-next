`Hypergraph.explicit_trace` takes the number of wires to trace and takes
them in one pass, so `make_causal` and `make_cocausal` convert the graph
once rather than once per cut: 7.5x faster on the traces of a graph with
128 of them over a category whose trace is a box, and the same term, the
`n`-wire trace being the `n` nested one-wire ones on either kind of
`trace_factory`. Its signature is now `explicit_trace(n=1, left=False)`,
`Hypergraph.trace`'s own, so a call passing `left` positionally — which
used to read `explicit_trace(True)` — passes it as `n` and must be spelled
`explicit_trace(left=True)` ([next#534](https://github.com/rel-int/discopy-next/pull/534)).
