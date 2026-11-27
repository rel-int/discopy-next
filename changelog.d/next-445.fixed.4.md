`neural.geometric.Geometric.cycle` with `rounds=0` now matches the reference
cycle: it keeps the carried units and cells and only refreshes the answer.
Before, the peer net's result differed and the factor net raised
`StopIteration` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
