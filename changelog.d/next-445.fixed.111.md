`eval(repr(x)) == x` now holds for `Ket`, `Bra`, `Copy`, `Match` and
`MixedScalar` (delayed too), and for a `Scalar` with a custom name, so
circuits that prepare, post-select, copy or match round-trip their repr
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
