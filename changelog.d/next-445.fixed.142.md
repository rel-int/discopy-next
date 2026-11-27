`Hypergraph.make_bijective`, `make_left_monogamous` and
`make_right_monogamous` insert every box they need in one pass, where each
rebuilt and re-validated the whole hypergraph once per box: a hypergraph
with 128 spiders to fix goes from 1.1 s to 0.012 s, and one with more of
them than the recursion limit allows no longer raises `RecursionError`
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
