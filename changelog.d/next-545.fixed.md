`CMap.explicit_trace` over a pivotal category builds its onion of cups and
caps in one constructor call instead of composing one pair at a time, so
tracing *n* wires rebuilds the permutation on its ports once rather than
2*n* times: it is linear in the wires traced where it was quadratic, and
`make_causal` is 95× at *n*=128, 1.808 s to 0.019 ([next#545](https://github.com/rel-int/discopy-next/pull/545)).
