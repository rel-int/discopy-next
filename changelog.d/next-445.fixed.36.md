`Swap(x @ y, z)` says a swap takes atomic types instead of reporting a
permutation of length 2, and `Swap` with a non-swap permutation and
`finset.Function` (so `markov.Function`) with an out-of-range or wrong-length
list raise a `ValueError` that says what is wrong ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
