`Hypergraph.from_callable` restores its category's own `__call__` and no
longer deletes it, so `Hypergraph[python.Function].from_callable` leaves
`Function` callable and a `from_callable` nested in another works
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
