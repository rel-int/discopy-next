`zw.ZBox` equality is exact on its amplitudes with the trailing zeros dropped,
so it is an equivalence relation and a `ZBox` is hashable and equal to its
one-layer diagram: `[1, 2] == [1, 2, 0]`, `[1, 2] != [1, 2, 3]`, and a NaN
amplitude equals itself. The tolerant prefix comparison is `ZBox.is_close`.
`zw.W` compares and hashes through its setoid, and `IndexableAmplitudes` has
a `__hash__` ([next#782](https://github.com/rel-int/discopy-next/pull/782)).
