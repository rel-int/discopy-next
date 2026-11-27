A non-element on the left of a monoid's `@` raises `TypeError`, as one on
the right does: `Monoid.whisker` takes elements only, so `'x' @ Nat(2)` and
`None @ List[int](2)` no longer give back the element ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
