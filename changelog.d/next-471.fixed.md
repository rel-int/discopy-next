`Hypergraph.make_monogamous` inserts every cup and cap it needs in one
pass, where it rebuilt and re-validated the whole hypergraph once per pair
of ports: a hypergraph with 128 transposes to bend goes from 4.8 s to
0.060 s, and one with more pairs than the recursion limit allows no longer
raises `RecursionError` ([next#471](https://github.com/rel-int/discopy-next/issues/471)).
