`Hypergraph.explicit_trace` over a pivotal category builds its onion of
cups and caps in one constructor call instead of composing one pair at a
time, so tracing *n* wires scans the boundary once rather than 2*n* times:
it is linear in the wires traced where it was quadratic, 260× at *n*=128,
and `make_causal` is 2.3× there ([next#539](https://github.com/rel-int/discopy-next/pull/539)).
