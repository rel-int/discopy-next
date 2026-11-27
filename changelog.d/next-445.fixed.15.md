`Box.free_symbols` reads a numeric array's dtype instead of visiting every
entry, so a no-op `subs` on large tensor boxes takes milliseconds instead of
seconds ([next#445](https://github.com/rel-int/discopy-next/issues/445)).
