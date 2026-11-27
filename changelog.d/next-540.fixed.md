`CMap.explicit_trace` takes the number of wires to trace as its **first**
argument, like `Hypergraph.explicit_trace`, so `CMap.trace` and
`CMap.make_causal` hand it their count instead of looping: over a traced
category that is not symmetric the map is converted once rather than once
per wire, 4.9× at *n*=128. A positional `explicit_trace(True)` therefore
now means one trace on the right rather than one on the left, and a count
that is not a number of wires the map has raises where it used to return
the map untraced. The pivotal branch, where a cut becomes cups and caps,
composed one pair at a time until [next#545](https://github.com/rel-int/discopy-next/pull/545), see [next#541](https://github.com/rel-int/discopy-next/issues/541) ([next#540](https://github.com/rel-int/discopy-next/pull/540)).
