`quantum.Measure`'s `destructive` and `Encode`'s `constructive` are
documented the right way round, `Circuit.from_tk` says it adds logical swaps
rather than SWAP gates, and `Controlled` says that a zero distance raises
`ValueError` ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
