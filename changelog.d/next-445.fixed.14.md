`Box.free_symbols` follows an in-place change to the box's data, so `subs`
and `lambdify` no longer return the box unchanged after a symbol is added to
mutable data ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
