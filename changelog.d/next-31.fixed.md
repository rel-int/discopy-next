A `Permutation` given an iterable that is not a sequence, e.g. an
iterator, a generator or a dict's keys, reads it once into a list, so
`Permutation(x @ y, iter([1, 0]))` is the `Swap` the list form gives
([next#31](https://github.com/rel-int/discopy-next/issues/31)).
